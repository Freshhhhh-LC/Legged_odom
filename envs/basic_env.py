import os
import sys
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch

import torch

import numpy as np

from utils.terrain import Terrain


class BasicEnv:
    def __init__(self, num_envs, sim_device, headless, dyn_rand=True, use_neck=False):
        self.gym = gymapi.acquire_gym()

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        if sim_device_type == "cuda":
            self.device = sim_device
        else:
            self.device = "cpu"

        self.headless = headless
        self.graphics_device_id = self.sim_device_id

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.num_envs = num_envs
        self.dyn_rand = dyn_rand

        self.create_sim(use_gpu=(sim_device_type == "cuda"))
        self.terrain = Terrain(self.gym, self.sim, self.device, uneven=True)
        self.create_envs(use_neck=use_neck)
        self.gym.prepare_sim(self.sim)  # otherwise the buffer allocation will fail

        self.set_viewer()
        self.set_camera()  # in case GUI is unavailable

        self.allocate_buffers()

    def create_sim(self, use_gpu):
        # set sim params
        self.up_axis_idx = 2  # 2 for z
        self.sim_params = gymapi.SimParams()
        self.sim_params.physx.use_gpu = use_gpu
        self.sim_params.use_gpu_pipeline = use_gpu
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        # set sim freq
        self.sim_params.dt = 0.004
        self.sim_params.substeps = 2
        self.sim_steps = 5
        self.step_dt = self.sim_steps * self.sim_params.dt

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, gymapi.SIM_PHYSX, self.sim_params)

    def create_envs(self, use_neck):
        # load robot assets
        asset_root = "resources/T1"
        asset_file = "T1_locomotion_with_neck.urdf" if use_neck else "T1_locomotion.urdf"
        asset_options = gymapi.AssetOptions()
        # dof drive mode: 0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort
        asset_options.default_dof_drive_mode = 1
        # bug: if collapse_fixed_joints = false, foot doesn't collide with ground
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        env_spacing = 0.25
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        env_per_row = int(np.sqrt(self.num_envs))

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.base_init_state = torch.zeros(1, 13, device=self.device)
        self.base_init_state[:, 2] = 0.71
        self.base_init_state[:, 4] = np.sin(0.0 * (np.pi / 180) / 2)
        self.base_init_state[:, 6] = np.cos(0.0 * (np.pi / 180) / 2)
        # offset of first env's origin to global origin
        env_offset_0 = -env_spacing * (env_per_row - 1)
        self.base_init_state[:, 0] = env_offset_0
        self.base_init_state[:, 1] = env_offset_0
        self.foot_height = 0.0375

        # global pos = local pos + env offset
        self.env_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_offset[:, 0] = 2 * env_spacing * (torch.arange(self.num_envs, device=self.device) % env_per_row)
        self.env_offset[:, 1] = 2 * env_spacing * (torch.arange(self.num_envs, device=self.device) // env_per_row)

        # acquire asset properties, may be used later
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        dof_names = self.gym.get_asset_dof_names(robot_asset)
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)

        stiffness_base = np.zeros(self.num_dof)
        damping_base = np.zeros(self.num_dof)
        stiffness_base[:] = 200.0  # waist & legs
        stiffness_base[[5, 6, 11, 12]] = 100.0  # ankles
        damping_base[:] = 2.0  # waist & legs
        damping_base[[5, 6, 11, 12]] = 4.0  # ankles
        dof_props["stiffness"][:] = stiffness_base
        dof_props["damping"][:] = damping_base
        dof_props["effort"][[2, 8]] = 25.0
        if use_neck:
            stiffness_base[-1] = 10.0
            damping_base[-1] = 0.2

        self.torque_limit = torch.from_numpy(dof_props["effort"]).clone().unsqueeze(0).to(self.device)
        self.dof_pos_limit = torch.zeros(3, 1, self.num_dof)
        self.dof_pos_limit[0, 0, :] = torch.from_numpy(dof_props["lower"])
        self.dof_pos_limit[1, 0, :] = torch.from_numpy(dof_props["upper"])
        self.dof_pos_limit[2, 0, :] = torch.from_numpy(dof_props["upper"] - dof_props["lower"])
        self.dof_pos_limit = self.dof_pos_limit.to(self.device)
        self.feet_indices = [-1, -1]
        self.hand_indices = [-1, -1]
        self.unallowed_contact_bodies = []
        if use_neck:
            self.head_index = None
        torso_index = None  # for special dyn randomization
        for i in range(self.num_bodies):
            if body_names[i] == "Trunk":
                torso_index = i
            if body_names[i] == "left_foot_link":
                self.feet_indices[0] = i
            elif body_names[i] == "right_foot_link":
                self.feet_indices[1] = i
            elif body_names[i] == "left_hand_link":
                self.hand_indices[0] = i
            elif body_names[i] == "right_hand_link":
                self.hand_indices[1] = i
            else:
                self.unallowed_contact_bodies.append(i)
            if use_neck:
                if body_names[i] == "H1":
                    self.head_index = i
        assert self.feet_indices[0] > 0 and self.feet_indices[1] > 0
        # assert self.hand_indices[0] > 0 and self.hand_indices[1] > 0
        assert torso_index is not None
        if use_neck:
            assert self.head_index is not None

        feet_collision_ids = []
        hand_collision_ids = []
        for j in self.feet_indices:
            feet_collision_ids += list(range(rbs_list[j].start, rbs_list[j].start + rbs_list[j].count))
        for j in self.hand_indices:
            hand_collision_ids += list(range(rbs_list[j].start, rbs_list[j].start + rbs_list[j].count))

        # create robot for each environment
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[0, 0:3])
        self.robot_handles = []
        self.envs = []

        if self.dyn_rand:
            self.torso_offset_scaled = torch.rand(self.num_envs, 3, device=self.device) * 2.0 - 1.0
            self.torso_mass_scaled = torch.rand(self.num_envs, 1, device=self.device) * 2.0 - 1.0
            self.friction_scaled = torch.rand(self.num_envs, 1, device=self.device) * 2.0 - 1.0
        else:
            self.torso_offset_scaled = torch.zeros(self.num_envs, 3, device=self.device)
            self.torso_mass_scaled = torch.zeros(self.num_envs, 1, device=self.device)
            self.friction_scaled = torch.zeros(self.num_envs, 1, device=self.device)

        def rand_float(half_range):
            return (np.random.rand() * 2.0 - 1.0) * half_range

        def dof_props_randomization(dof_props):
            dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][:] = stiffness_base * (1.0 + np.random.rand(self.num_dof) * 0.1 - 0.05)
            dof_props["damping"][:] = damping_base * (1.0 + np.random.rand(self.num_dof) * 0.1 - 0.05)

        def rigid_body_randomization(rb_props):
            for k in range(self.num_bodies):
                rb_props[k].com.x += rand_float(0.005)
                rb_props[k].com.y += rand_float(0.005)
                rb_props[k].com.z += rand_float(0.005)
                rb_props[k].mass *= 1.0 + rand_float(0.02)
                rb_props[k].invMass = 1.0 / rb_props[k].mass
            # stronger torso randomization
            rb_props[torso_index].com.x += self.torso_offset_scaled[i, 0].item() * 0.1
            rb_props[torso_index].com.y += self.torso_offset_scaled[i, 1].item() * 0.1
            rb_props[torso_index].com.z += self.torso_offset_scaled[i, 2].item() * 0.1
            rb_props[torso_index].mass *= 1.0 + self.torso_mass_scaled[i, 0].item() * 0.2
            rb_props[torso_index].invMass = 1.0 / rb_props[torso_index].mass

        def collision_randomization(col_props):
            # do not randomize too many items, otherwise the PxPhysics will
            # create too many collision materials that exceeds the limits of materials.
            # for k in range(len(col_props)):
            for k in feet_collision_ids:
                col_props[k].friction = 1.5 + rand_float(0.2) + self.friction_scaled[i, 0].item() * 1.0
                col_props[k].compliance = 1.0 + rand_float(0.5)
                col_props[k].restitution = 0.5 + rand_float(0.4)

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, env_per_row)
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, start_pose, "T1", i, 0, 0)  # disable self collision
            if self.dyn_rand:
                dof_props_randomization(dof_props)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
            if self.dyn_rand:
                rigid_body_randomization(body_props)
            self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, body_props)

            collision_props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
            if self.dyn_rand:
                collision_randomization(collision_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, collision_props)

            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)

    def set_viewer(self):
        self.viewer = None
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.enable_viewer_sync = True
            cam_pos = gymapi.Vec3(1.0, 0.0, 0.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def set_camera(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 720
        camera_props.height = 1280
        camera_props.use_collision_geometry = False
        cam_pos = gymapi.Vec3(0.0, -0.5, 0.2)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
        self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], camera_props)
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], cam_pos, cam_target)

    def allocate_buffers(self):
        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # refresh them
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # wrap gym tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)

        # proprio-observation:
        # grav vec: 3
        # rot vel: 3
        # dof state: self.num_dof * 2
        # last_action: self.num_dof
        self.num_basic_obs = 3 * 2 + 3 * self.num_dof

        self.grav_vec = torch.tensor((0.0, 0.0, -1.0)).reshape(1, 3).expand(self.num_envs, 3).to(self.device)
        # grav randomization
        if self.dyn_rand:
            self.grav_vec[:] += torch.randn(self.num_envs, 3, device=self.device) * 0.01

        # randomized delay
        # self.update_at_substeps[i, j] = True: env j updates control target at ith sim step
        self.update_at_substeps = torch.randint(self.sim_steps, (self.num_envs,), device=self.device).unsqueeze(0) == torch.arange(self.sim_steps).to(
            self.device
        ).unsqueeze(-1)
        self.max_delay_cnt = 4
        self.delay_cnt = torch.randint(2, self.max_delay_cnt, (self.num_envs,), device=self.device)

        self.pushing_forces = torch.zeros(self.num_envs, self.num_dof + 1, 3, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_dof + 1, 3, device=self.device)

        # buffer
        self.last_dof_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.basic_act_history = torch.zeros(self.num_envs, self.max_delay_cnt, self.num_dof, device=self.device)

    def render(self, mode="human"):
        if mode == "human":
            assert self.viewer is not None, "Headless mode, no viewer to render"
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
        elif mode == "rgb_array":
            bx, by, bz = self.root_states[0, 0:3]
            cam_pos = gymapi.Vec3(bx + 0.8, by, bz)
            cam_target = gymapi.Vec3(bx, by, bz)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], cam_pos, cam_target)

            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
            w, h = img.shape
            return img.reshape([w, h // 4, 4])

    def env_reset_idx(self, idx):
        self.last_dof_target[idx] = self.dof_pos[idx]
        self.basic_act_history[idx, :, :] = 0.0

    def env_step(self, action):
        self.basic_act_history = torch.roll(self.basic_act_history, 1, dims=1)
        self.basic_act_history[:, 0, :] = action

        # build jpos target
        dof_pos_target = self.basic_act_history[torch.arange(self.num_envs, device=self.device), self.delay_cnt]

        # simulate
        for i in range(self.sim_steps):
            self.last_dof_target[self.update_at_substeps[i]] = dof_pos_target[self.update_at_substeps[i]]
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.last_dof_target))
            # apply_pushing_forces
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.pushing_forces), gymtorch.unwrap_tensor(self.pushing_torques), gymapi.LOCAL_SPACE
            )
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if not self.headless:
            self.render(mode="human")
