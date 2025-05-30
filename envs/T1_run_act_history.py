from envs.basic_env import BasicEnv
import torch
import numpy as np

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *


def _build_t1_mirror_obs_mat():
    mat = torch.zeros(83, 83)
    mat[0:14, 0:14] = torch.eye(14)
    mat[14:20, 20:26] = torch.eye(6)
    mat[20:26, 14:20] = torch.eye(6)
    mat[26, 26] = 1
    mat[27:33, 33:39] = torch.eye(6)
    mat[33:39, 27:33] = torch.eye(6)
    mat[39, 39] = 1
    mat[40:45, 45:50] = torch.eye(5)
    mat[45:50, 40:45] = torch.eye(5)
    mat[50, 50] = 1
    mat[51:56, 56:61] = torch.eye(5)
    mat[56:61, 51:56] = torch.eye(5)
    mat[61, 61] = 1
    mat[62:67, 67:72] = torch.eye(5)
    mat[67:72, 62:67] = torch.eye(5)
    mat[72, 72] = 1
    mat[73:78, 78:83] = torch.eye(5)
    mat[78:83, 73:78] = torch.eye(5)
    flip_val = torch.ones(83)
    inverse_ids = [
        # inverse vectors
        1,
        3,
        5,
        7,
        8,
        9,
        10,
        11,
        # inverse waist, hip roll & yaw, ankle roll
        13,
        15,
        16,
        19,
        21,
        22,
        25,  # jpos
        26,
        28,
        29,
        32,
        34,
        35,
        38,  # jvel
        39,
        41,
        42,
        46,
        47,  # actions
        50,
        52,
        53,
        57,
        58,  # last actions
        61,
        63,
        64,
        68,
        69,  # last 2nd actions
        72,
        74,
        75,
        79,
        80,  # last 3rd actions
    ]
    flip_val[inverse_ids] = -1
    flip_mat = torch.diag(flip_val)
    mirror_transform_mat = torch.matmul(mat, flip_mat)
    return mirror_transform_mat


def _build_t1_mirror_privileged_mat():
    flip_val = torch.ones(14)
    inverse_ids = [1, 5, 9, 11, 13]
    flip_val[inverse_ids] = -1
    flip_mat = torch.diag(flip_val)
    return flip_mat


def _build_t1_mirror_action_mat():
    mat = torch.zeros(11, 11)
    mat[0, 0] = 1.0
    mat[1:6, 6:11] = torch.eye(5)
    mat[6:11, 1:6] = torch.eye(5)
    flip_val = torch.ones(11)
    inverse_ids = [0, 2, 3, 7, 8]
    flip_val[inverse_ids] = -1
    flip_mat = torch.diag(flip_val)
    mirror_transform_mat = torch.matmul(mat, flip_mat)
    return mirror_transform_mat


class T1RunActHistoryEnv(BasicEnv):
    mirror_obs_mat = _build_t1_mirror_obs_mat()
    mirror_priv_mat = _build_t1_mirror_privileged_mat()
    mirror_act_mat = _build_t1_mirror_action_mat()

    def __init__(self, num_envs, sim_device, headless, dyn_rand=True, curriculum=True, change_cmd=False):
        super().__init__(num_envs, sim_device, headless, dyn_rand=dyn_rand)

        self.curriculum = curriculum
        self.need_change_cmd = change_cmd
        self.num_zero_vel_envs = int(self.num_envs * 0.25)
        if self.need_change_cmd:
            self.ep_len_max = 1000 # 20s
        else:
            self.ep_len_max = 5000  # 100 s

        # action:
        # jpos_inc : self.num_dof - 2, no ankle roll
        self.num_act = self.num_dof - 2
        # observation:
        # grav vec: 3
        # rot vel: 3
        # vel xy cmd: 2
        # rot vel cmd: 1
        # ori feedback: 1
        # phase: 2
        # phase rate: 1
        # dof pos: self.num_dof
        # dof vel: self.num_dof
        # last_action: self.num_act
        self.num_obs = 3 * 2 + 2 + 1 + 1 + 2 + 1 + 2 * self.num_dof + 4 * self.num_act
        # privileged obs:
        # torso CoM: 3
        # mass: 1
        # friction: 1
        # lin vel: 3
        # height: 1
        # pushing twist: 6
        self.num_privileged_obs = 3 + 1 + 3 + 1 + 6

        self.step_cnt = 0
        self.kick_interval = 100  # per 2s
        self.push_interval = 251  # per 5s
        self.push_dur = 50  # 1s

        self.grav_vec = torch.tensor((0.0, 0.0, -1.0)).reshape(1, 3).expand(self.num_envs, 3).to(self.device)
        # grav randomization
        if self.dyn_rand:
            self.grav_vec[:] += torch.randn(self.num_envs, 3, device=self.device) * 0.01

        self.q0 = torch.zeros(1, self.num_dof, device=self.device)
        # ankle
        self.q0[0, [1, 7]] = -0.2
        self.q0[0, [4, 10]] = 0.4
        self.q0[0, [5, 11]] = -0.25
        self.start_height = 0.72

        self.low_vel_envs = int(self.num_envs * 0.5)

        # curriculum
        if self.curriculum:
            self.forward_vel_max = 0.6
            self.lateral_vel_max = 0.4
            self.rot_vel_max = 1.0
        else:
            self.forward_vel_max = 2.2
            self.lateral_vel_max = 1.0
            self.rot_vel_max = 2.0

        # curriculum state
        self.last_ep_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_tracking_res = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_target = torch.zeros(self.num_envs, 3, device=self.device)
        # grid curriculum. vx resolution: 0.2m/s, rot_vel resolution: 0.1 rad/s
        self.grid_curriculum = torch.zeros(10 * 41 * 41, device=self.device)
        self.grid_curriculum.view(10, 41, 41)[:, 20, 20] = 1.0
        self.env_grid_idx = torch.full((self.num_envs,), fill_value=20 * 41 + 20, device=self.device, dtype=torch.long)

        self.pushing_forces = torch.zeros(self.num_envs, self.num_dof + 1, 3, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_dof + 1, 3, device=self.device)

        # control buffer
        self.dof_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)

        # state buffer
        self.action_history = torch.zeros(self.num_envs, 4, self.num_act, device=self.device)
        self.last_dof_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.last_root_vel = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.root_acc = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
        self.dof_acc = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.filtered_vel = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        self.feet_global_pos = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.feet_global_vel = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.last_feet_vel = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.toe_global_pos = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.heel_global_pos = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.last_toe_global_pos = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.toe_ground_height = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.heel_ground_height = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_heel_global_pos = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        self.toe_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.heel_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.feet_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.touch_down = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        self.vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_rate = torch.zeros(self.num_envs, device=self.device)
        self.cmd_change_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.q_inc = torch.zeros(1, 12, device=self.device)
        self.q_inc[0, [0, 6]] = -np.pi / 12  # hip Y
        self.q_inc[0, [3, 9]] = np.pi / 6  # knee
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)

        # curriculum state
        self.last_ep_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.last_forward_vel = torch.zeros(self.num_envs, device=self.device)
        self.last_forward_cmd = torch.zeros(self.num_envs, device=self.device)

        # return buffer
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.privileged_obs_buf[:, :3] = self.torso_offset_scaled
        self.privileged_obs_buf[:, 3:4] = self.torso_mass_scaled
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.done_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ep_len = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def update_grid_curriculum(self, update_env_mask):
        vel_err = (self.last_tracking_res[update_env_mask] - self.last_target[update_env_mask]).abs()
        tracking_good = (vel_err[:, 0] < 0.5) & (vel_err[:, 1] < 0.5) & (vel_err[:, 2] < 0.5)
        walk_stably = self.last_ep_len[update_env_mask] > (self.ep_len_max - 50)
        success = tracking_good & walk_stably
        succeeded_grid_idx = self.env_grid_idx[update_env_mask][success]
        success_cnt = torch.bincount(succeeded_grid_idx, minlength=self.grid_curriculum.numel())

        # expand grid
        grid_view = self.grid_curriculum.view(10, 41, 41)
        grid_inc = (success_cnt.float() * 0.1).reshape(10, 41, 41)
        grid_view += grid_inc
        grid_view[:, :, 1:] += grid_inc[:, :, :-1]
        grid_view[:, :, :-1] += grid_inc[:, :, 1:]
        grid_view[:, 1:, :] += grid_inc[:, :-1, :]
        grid_view[:, :-1, :] += grid_inc[:, 1:, :]

        self.grid_curriculum.clamp_(max=1.0)

    def get_grid_cmd(self, len_cmd):
        grid_idx = torch.multinomial(self.grid_curriculum, len_cmd, replacement=True)
        phase_rate_grid = torch.div(grid_idx, 41 * 41, rounding_mode="floor")
        vx_grid = (torch.div(grid_idx, 41, rounding_mode="floor") % 41) - 20
        rot_vel_grid = (grid_idx % 41) - 20
        vx = vx_grid * 0.2 + torch.rand(len_cmd, device=self.device) * 0.2 - 0.1
        vy = (torch.rand(len_cmd, device=self.device) * 2 - 1) * (vx_grid.abs() * 0.5)
        rot_vel = rot_vel_grid * 0.1 + torch.rand(len_cmd, device=self.device) * 0.1 - 0.05
        phase_rate = phase_rate_grid * 0.1 + torch.rand(len_cmd, device=self.device) * 0.1
        return torch.stack((vx, vy, rot_vel), dim=-1), phase_rate, grid_idx

    def reset(self):
        self.done_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs))
        self.change_cmd()
        self.compute_obs()
        self.rew_terms = {}
        return self.obs_buf, {"privileged_obs": self.privileged_obs_buf}

    def update_curriculum(self, idx):
        self.last_ep_len[idx] = self.ep_len[idx]
        self.last_tracking_res[idx, 0:2] = self.filtered_vel[idx, 0:2]
        self.last_tracking_res[idx, 2] = self.filtered_vel[idx, 3]
        self.last_target[idx] = self.vel_cmd[idx]
        self.update_grid_curriculum(idx)

    def reset_idx(self, idx):
        len_idx = len(idx)
        if len_idx == 0:
            return

        if self.curriculum:
            self.update_curriculum(idx)

        # reset sim state
        idx_int32 = idx.to(dtype=torch.int32, device=self.device)
        self.root_states[idx, :2] = self.base_init_state[:, :2]
        self.root_states[idx, 2] = self.start_height + self.terrain.local_max_heights(self.root_states[idx, :2] + self.env_offset[idx, :2])
        yaw = torch.rand(len_idx, device=self.device) * (2 * np.pi)
        self.root_states[idx, 3:7] = quat_from_euler_xyz(
            torch.zeros(len_idx, device=self.device), torch.full((len_idx,), fill_value=0.1, device=self.device), yaw
        )
        self.root_states[idx, 7:9] = torch.randn(len_idx, 2, device=self.device) * 0.1
        self.root_states[idx, 9:] = 0.0
        self.dof_pos[idx] = self.q0 + torch.randn(len_idx, self.num_dof, device=self.device) * 0.05
        self.dof_vel[idx] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(idx_int32), len_idx)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(idx_int32), len_idx)

        self.last_root_vel[idx] = self.root_states[idx, 7:13]
        self.last_dof_vel[idx] = self.dof_vel[idx]

        self.filtered_vel[idx] = 0.0
        self.ep_len[idx] = 0
        self.dof_pos_target[idx] = self.q0
        self.cmd_change_time[idx] = 0
        self.action_history[idx, :, :] = 0.0

        self.env_reset_idx(idx)

    def step(self, action):
        self.action_history[:] = torch.roll(self.action_history, 1, dims=1)
        self.action_history[:, 0, :] = torch.clip(action, -1.0, 1.0)

        # build ref traj
        stand_cmd = self.phase_rate < -0.5
        leg_phases = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        rescaled_phase = self.phase.unsqueeze(-1) / 0.4
        phase_offset = torch.tensor([0.05 / 0.4, 0.55 / 0.4]).reshape(1, 2).to(self.device)
        feet_phases = torch.clip(rescaled_phase - phase_offset, min=0.0, max=1.0)
        self.j_inc_scale = (1.0 - torch.cos(2 * np.pi * feet_phases)) / 2
        self.j_inc_scale[stand_cmd, :] = 0.0
        self.dof_pos_ref[:] = self.q0
        self.dof_pos_ref[:, 1:7] += self.q_inc[:, 0:6] * self.j_inc_scale[:, 0:1]
        self.dof_pos_ref[:, 7:13] += self.q_inc[:, 6:12] * self.j_inc_scale[:, 1:2]

        # build jpos target
        self.dof_pos_target[:] = self.dof_pos_ref
        self.dof_pos_target[:, 0:6] += self.action_history[:, 0, 0:6]
        self.dof_pos_target[:, 7:12] += self.action_history[:, 0, 6:11]

        # update phase
        phase_inc = (self.phase_rate * 1.0 + 1.0) * self.step_dt
        feet_td_ref = ((self.phase >= 0.5) & (self.phase - phase_inc < 0.5)) | ((self.phase >= 0.0) & (self.phase - phase_inc < 0.0))
        phase_inc[stand_cmd] = 0.0
        self.phase[:] = torch.fmod(self.phase + phase_inc, 1.0)

        # disturbance for push recovery
        if self.step_cnt % self.kick_interval == 0:
            self.kick_robots()

        if self.step_cnt % self.push_interval == 0:
            self.set_pushing_forces()
        elif self.step_cnt % self.push_interval == self.push_dur:
            self.reset_pushing_forces()

        # simulate
        self.env_step(self.dof_pos_target)
        self.refresh_feet_pos()

        self.teleport_robot()

        self.ep_len += 1
        self.step_cnt += 1
        self.root_acc[:] = (self.root_states[:, 7:13] - self.last_root_vel) / self.step_dt
        self.dof_acc[:] = (self.dof_vel - self.last_dof_vel) / self.step_dt
        local_vel = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        self.filtered_vel[:, 0:3] = self.filtered_vel[:, 0:3] * 0.9 + local_vel * 0.1
        self.filtered_vel[:, 3] = self.filtered_vel[:, 3] * 0.9 + self.root_states[:, 12] * 0.1
        self.compute_rew_and_reset()
        self.change_cmd()
        self.compute_obs()
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_dof_vel[:] = self.dof_vel
        self.last_dof_target[:] = self.dof_pos_target

        return (
            self.obs_buf,
            self.rew_buf,
            self.done_buf,
            {"time_outs": self.timeout_buf, "rew_terms": self.rew_terms, "privileged_obs": self.privileged_obs_buf},
        )

    def change_cmd(self):
        # update new cmd
        need_update = (self.ep_len == self.cmd_change_time) & (self.ep_len < self.ep_len_max - 200)
        len_update = need_update.long().sum().item()
        if len_update == 0:
            return

        low_vel_mask = torch.arange(self.num_envs, device=self.device) < self.low_vel_envs
        low_vel_update = need_update & low_vel_mask
        high_vel_update = need_update & ~low_vel_mask
        len_low_vel_update = low_vel_update.long().sum().item()
        len_high_vel_update = high_vel_update.long().sum().item()

        if len_low_vel_update > 0:
            self.vel_cmd[low_vel_update, 0] = (torch.rand(len_low_vel_update, device=self.device) * 2.0 - 1.0) * 0.4
            self.vel_cmd[low_vel_update, 1] = (torch.rand(len_low_vel_update, device=self.device) * 2.0 - 1.0) * 0.4
            self.vel_cmd[low_vel_update, 2] = (torch.rand(len_low_vel_update, device=self.device) * 2.0 - 1.0) * 0.5
            self.phase_rate[low_vel_update] = torch.rand(len_low_vel_update, device=self.device)

        if len_high_vel_update > 0:
            if self.curriculum:
                self.vel_cmd[high_vel_update, :], self.phase_rate[high_vel_update], self.env_grid_idx[high_vel_update] = self.get_grid_cmd(
                    len_high_vel_update
                )
            else:
                self.vel_cmd[high_vel_update, 0] = (torch.rand(len_high_vel_update, device=self.device) * 2.0 - 1.0) * self.forward_vel_max
                self.vel_cmd[high_vel_update, 1] = (torch.rand(len_high_vel_update, device=self.device) * 2.0 - 1.0) * self.lateral_vel_max
                self.vel_cmd[high_vel_update, 2] = (torch.rand(len_high_vel_update, device=self.device) * 2.0 - 1.0) * self.rot_vel_max
                self.phase_rate[high_vel_update] = torch.rand(len_high_vel_update, device=self.device)

        if self.num_envs > 10:
            set_zero_mask = (torch.rand(self.num_envs, device=self.device) < 0.3) & need_update
            set_zero_mask[: self.num_zero_vel_envs] = True
            self.vel_cmd[set_zero_mask, :] = 0.0
            self.env_grid_idx[set_zero_mask] = 41 * 20 + 20
            stop_mask = set_zero_mask & (torch.rand(self.num_envs, device=self.device) < 0.8)
            self.phase_rate[stop_mask] = -1

        if self.need_change_cmd:
            self.cmd_change_time[need_update] += 400 + torch.randint(200, (len_update,), device=self.device)
        else:
            self.cmd_change_time[need_update] += 200000
        
        # # 固定cmd
        # self.vel_cmd[:, 0] = 0.0
        # self.vel_cmd[:, 1] = 0.0
        # self.vel_cmd[:, 2] = 0.0
        # self.phase_rate[:] = -1

        self.root_states[need_update, 7:9] += torch.randn(len_update, 2, device=self.device) * 0.3
        self.root_states[need_update, 10:] += torch.randn(len_update, 3, device=self.device) * 0.1
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(need_update.nonzero(as_tuple=False).flatten().to(dtype=torch.int32)),
            len_update,
        )

    def refresh_feet_pos(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        feet_state = self.body_states[:, self.feet_indices, :]

        self.feet_global_pos[:] = self.env_offset.unsqueeze(1) + feet_state[:, :, 0:3]
        flattened_quat = feet_state[:, :, 3:7].flatten(end_dim=-2)
        feet_roll, _, feet_yaw = get_euler_xyz(flattened_quat)
        self.feet_yaw[:] = feet_yaw.reshape(self.num_envs, 2)
        self.feet_roll[:] = torch.fmod(feet_roll.reshape(self.num_envs, 2) + np.pi, 2 * np.pi) - np.pi

        self.last_feet_vel[:] = self.feet_global_vel
        self.feet_global_vel[:] = feet_state[:, :, 7:10]

        toe_relative_pos = torch.tensor([[0.1265, 0.0, -0.04]], device=self.device).expand(2 * self.num_envs, 3)
        toe_relative_pos = quat_rotate(flattened_quat, toe_relative_pos).reshape(self.num_envs, 2, 3)
        self.last_toe_global_pos[:] = self.toe_global_pos
        self.toe_global_pos[:] = self.feet_global_pos + toe_relative_pos
        self.toe_ground_height[:] = self.terrain.terrain_heights(self.toe_global_pos.view(-1, 3)[:, 0:2]).view(self.num_envs, 2)
        self.toe_contact[:] = self.toe_global_pos[:, :, 2] - self.toe_ground_height < 0.01

        heel_relative_pos = torch.tensor([[-0.0965, 0.0, -0.04]], device=self.device).expand(2 * self.num_envs, 3)
        heel_relative_pos = quat_rotate(flattened_quat, heel_relative_pos).reshape(self.num_envs, 2, 3)
        self.last_heel_global_pos[:] = self.heel_global_pos
        self.heel_global_pos[:] = self.feet_global_pos + heel_relative_pos
        self.heel_ground_height[:] = self.terrain.terrain_heights(self.heel_global_pos.view(-1, 3)[:, 0:2]).view(self.num_envs, 2)
        self.heel_contact[:] = self.heel_global_pos[:, :, 2] - self.heel_ground_height < 0.01

        new_feet_contact = self.toe_contact | self.heel_contact
        self.touch_down = new_feet_contact & ~self.feet_contact
        foot_raise = ~new_feet_contact & self.feet_contact
        self.feet_contact[:] = new_feet_contact

    def kick_robots(self):
        self.root_states[:, 7:10] += torch.randn(self.num_envs, 3, device=self.device) * 0.1
        self.root_states[:, 10:] += torch.randn(self.num_envs, 3, device=self.device) * 0.02
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def set_pushing_forces(self):
        self.pushing_forces[:, 0, :] = torch.randn(self.num_envs, 3, device=self.device) * 10.0
        self.pushing_torques[:, 0, :] = torch.randn(self.num_envs, 3, device=self.device) * 2.0

    def reset_pushing_forces(self):
        self.pushing_forces[:, 0, :].zero_()
        self.pushing_torques[:, 0, :].zero_()

    def teleport_robot(self):
        if not self.terrain.uneven:
            return

        needs_teleport = False
        horizontal_pos = self.root_states[:, :2] + self.env_offset[:, :2]
        half_length = self.terrain.center_pos[0, 0].item()
        half_width = self.terrain.center_pos[0, 1].item()

        is_out = horizontal_pos[:, 0] > (half_length - 0.25 * self.terrain.border_width)
        if is_out.any():
            needs_teleport = True
            self.root_states[is_out, 0] -= 2 * half_length - self.terrain.border_width

        is_out = horizontal_pos[:, 0] < (-half_length + 0.25 * self.terrain.border_width)
        if is_out.any():
            needs_teleport = True
            self.root_states[is_out, 0] += 2 * half_length - self.terrain.border_width

        is_out = horizontal_pos[:, 1] > (half_width - 0.25 * self.terrain.border_width)
        if is_out.any():
            needs_teleport = True
            self.root_states[is_out, 1] -= 2 * half_width - self.terrain.border_width

        is_out = horizontal_pos[:, 1] < (-half_width + 0.25 * self.terrain.border_width)
        if is_out.any():
            needs_teleport = True
            self.root_states[is_out, 1] += 2 * half_width - self.terrain.border_width

        if needs_teleport:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_rew_terms(self):
        grav_vec_raw = torch.tensor((0.0, 0.0, -1.0)).reshape(1, 3).expand(self.num_envs, 3).to(self.device)
        torso_grav_vec = quat_rotate_inverse(self.root_states[:, 3:7], grav_vec_raw)
        _, _, torso_yaw = get_euler_xyz(self.root_states[:, 3:7])
        torso_yaw_expected = self.feet_yaw.mean(dim=-1) + np.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > np.pi)
        torso_yaw_err = torch.fmod(torso_yaw_expected - torso_yaw + 3 * np.pi, 2 * np.pi) - np.pi
        cy = torch.cos(torso_yaw)
        sy = torch.sin(torso_yaw)
        feet_dist = self.feet_global_pos[:, 1, 0:2] - self.feet_global_pos[:, 0, 0:2]
        feet_lateral_dist = torch.abs(cy * feet_dist[:, 1] - sy * feet_dist[:, 0])
        feet_center = self.feet_global_pos[:, :, 0:2].mean(dim=1) - self.env_offset[:, 0:2] - self.root_states[:, 0:2]
        feet_center_forward = cy * feet_center[:, 0] + sy * feet_center[:, 1]
        feet_center_lateral = cy * feet_center[:, 1] - sy * feet_center[:, 0]
        feet_forward_ref = self.vel_cmd[:, 2] * self.vel_cmd[:, 1] * 0.85 * 0.66 / 9.81 + 0.03
        feet_lateral_ref = -self.vel_cmd[:, 2] * self.vel_cmd[:, 0] * 0.85 * 0.66 / 9.81

        action_dot = (self.action_history[:, 0] - self.action_history[:, 1]) / self.step_dt
        action_ddot = (self.action_history[:, 0] + self.action_history[:, 2] - 2 * self.action_history[:, 1]) / (self.step_dt**2)

        tiredness = torch.clip(self.torques / self.torque_limit, min=-1.0, max=1.0).square().sum(dim=-1)
        power = torch.clip(self.torques * self.dof_vel, min=0.0).sum(dim=-1)

        toe_vel = (self.toe_global_pos - self.last_toe_global_pos) / self.step_dt
        heel_vel = (self.heel_global_pos - self.last_heel_global_pos) / self.step_dt
        feet_vel_z = (toe_vel[:, :, 2] + heel_vel[:, :, 2]) / 2
        feet_acc = (self.feet_global_vel - self.last_feet_vel) / self.step_dt
        heel_rel_z = self.heel_global_pos[:, :, 2] - self.toe_global_pos[:, :, 2]

        relative_jpos = (self.dof_pos - self.dof_pos_limit[0]) / self.dof_pos_limit[2]
        jpos_at_limit = ((relative_jpos < 0.01) | (relative_jpos > 0.99)).float().sum(dim=-1)

        heel_raise_height = self.heel_global_pos[:, :, 2] - self.heel_ground_height
        toe_raise_height = self.toe_global_pos[:, :, 2] - self.toe_ground_height
        feet_raise_height = torch.max(heel_raise_height, toe_raise_height)
        left_foot_raised = (toe_raise_height[:, 0] > 0.01) & (heel_raise_height[:, 0] > 0.01)
        right_foot_raised = (toe_raise_height[:, 1] > 0.01) & (heel_raise_height[:, 1] > 0.01)
        feet_yaw_diff = torch.fmod(self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + 3 * np.pi, 2 * np.pi) - np.pi

        solid_contact = self.heel_contact & self.toe_contact

        stand_cmd = self.phase_rate < -0.5

        feet_should_raise = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        feet_should_raise[:, 0] = (self.phase > 0.15) & (self.phase < 0.35)
        feet_should_raise[:, 1] = (self.phase > 0.65) & (self.phase < 0.85)
        feet_should_step = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        feet_should_step[:, 0] = (self.phase > 0.65) & (self.phase < 0.85)
        feet_should_step[:, 1] = (self.phase > 0.15) & (self.phase < 0.35)
        feet_should_step |= stand_cmd.unsqueeze(-1)

        height_ref = 0.67

        terrain_height = self.terrain.terrain_heights(self.root_states[:, :2] + self.env_offset[:, :2])
        vel_norm = self.vel_cmd.square().sum(dim=-1)
        vel_cmd_weight = torch.clip(vel_norm, max=1.0) * 0.2 + 0.05
        weight_adjust = 0.1 / torch.clip(vel_norm, min=0.1, max=1.0)

        rew_vel_tracking = (
            torch.exp(-(self.filtered_vel[:, 0] - self.vel_cmd[:, 0]).square() / vel_cmd_weight) * 2.0
            + torch.exp(-(self.filtered_vel[:, 1] - self.vel_cmd[:, 1]).square() / vel_cmd_weight) * 2.0
            + torch.exp(-(self.filtered_vel[:, 3] - self.vel_cmd[:, 2]).square() / vel_cmd_weight) * 2.0
        )
        rew_height = torch.exp(-(self.root_states[:, 2] - height_ref - terrain_height).square() / 0.01) * 4e0
        rew_feet_center = (
            torch.exp(-(feet_center_forward - feet_forward_ref).square() / 0.01) * 0.0
            + torch.exp(-(feet_center_lateral - feet_lateral_ref).square() / 0.01) * 1.0
        )

        loss_angle = torso_grav_vec[:, 0].square() * 1e1 + torso_grav_vec[:, 1].square() * 1e1
        loss_tiredness = tiredness * 5e-2
        loss_power = power * 2e-4
        loss_torque = self.torques.square().sum(dim=-1) * 2e-5
        loss_vel = self.root_states[:, 10:12].square().sum(dim=-1) * 5e-1 + self.filtered_vel[:, 2].square() * 2e-1
        loss_acc = self.root_acc.square().sum(dim=-1) * weight_adjust * 1e-4
        loss_jvel = self.dof_vel.square().sum(dim=-1) * weight_adjust * 1e-3 + self.dof_vel[:, 0].abs() * weight_adjust * 1e-1
        loss_jacc = self.dof_acc.square().sum(dim=-1) * weight_adjust * 2e-6
        loss_jpos = (self.dof_pos - self.dof_pos_ref).square().sum(dim=-1) * 1e-2
        loss_action_rate = action_dot.square().sum(dim=-1) * 1e-3
        loss_action_jerk = action_ddot.square().sum(dim=-1) * 2e-7
        loss_limit = jpos_at_limit * 1.0
        loss_contact = (self.contact_forces[:, self.unallowed_contact_bodies].square().sum(dim=-1) > 5.0).sum(dim=-1) * 1.0
        loss_feet_slip = (toe_vel.square().sum(dim=-1) * self.toe_contact.float() + heel_vel.square().sum(dim=-1) * self.heel_contact.float()).sum(
            dim=-1
        ) * 1e-2
        loss_feet_z_vel = feet_vel_z.square().sum(dim=-1) * 4e0
        loss_feet_acc = feet_acc.square().sum(dim=(-1, -2)) * 1e-3
        loss_waist_pos = self.dof_pos[:, 0].square() * (1.0 + stand_cmd.float() * 2.0) * 2e0
        loss_facing_direction = torso_yaw_err.square() * 2e0
        loss_feet_roll = self.feet_roll.square().sum(dim=-1) * 1e1 + feet_yaw_diff.square() * 2e0
        loss_unexpected_step = (feet_should_raise & self.feet_contact).any(dim=-1).float() * 4.0
        loss_unexpected_swing = (feet_should_step & ~solid_contact).any(dim=-1).float() * 2.0
        loss_feet_too_close = torch.clip(0.21 - feet_lateral_dist, min=0.0, max=0.04) * 5e2

        moving_forward = (self.vel_cmd[:, 0] > 0.1).unsqueeze(-1)
        moving_backward = (self.vel_cmd[:, 0] < -0.1).unsqueeze(-1)
        moving_aside = ~(moving_forward | moving_backward)
        loss_heel_down = (
            (-torch.clip(heel_rel_z, max=0.0) * (~self.toe_contact & moving_forward).float()).sum(dim=-1) * 1e1
            + (torch.clip(heel_rel_z, min=0.0) * (~self.heel_contact & moving_backward).float()).sum(dim=-1) * 1e1
            + (heel_rel_z.abs() * (~self.feet_contact & moving_aside).float()).sum(dim=-1) * 1e0
        )

        self.rew_terms["alive"] = torch.full((self.num_envs,), fill_value=4.0 * self.step_dt, device=self.device)
        self.rew_terms["rew_vel_tracking"] = rew_vel_tracking * self.step_dt
        self.rew_terms["rew_height"] = rew_height * self.step_dt
        self.rew_terms["rew_feet_center"] = rew_feet_center * self.step_dt
        self.rew_terms["loss_angle"] = -loss_angle * self.step_dt
        self.rew_terms["loss_tiredness"] = -loss_tiredness * self.step_dt
        self.rew_terms["loss_power"] = -loss_power * self.step_dt
        self.rew_terms["loss_torque"] = -loss_torque * self.step_dt
        self.rew_terms["loss_vel"] = -loss_vel * self.step_dt
        self.rew_terms["loss_acc"] = -loss_acc * self.step_dt
        self.rew_terms["loss_jvel"] = -loss_jvel * self.step_dt
        self.rew_terms["loss_jacc"] = -loss_jacc * self.step_dt
        self.rew_terms["loss_jpos"] = -loss_jpos * self.step_dt
        self.rew_terms["loss_action_rate"] = -loss_action_rate * self.step_dt
        self.rew_terms["loss_action_jerk"] = -loss_action_jerk * self.step_dt
        self.rew_terms["loss_limit"] = -loss_limit * self.step_dt
        self.rew_terms["loss_contact"] = -loss_contact * self.step_dt
        self.rew_terms["loss_feet_slip"] = -loss_feet_slip * self.step_dt
        self.rew_terms["loss_feet_z_vel"] = -loss_feet_z_vel * self.step_dt
        self.rew_terms["loss_feet_acc"] = -loss_feet_acc * self.step_dt
        self.rew_terms["loss_waist_pos"] = -loss_waist_pos * self.step_dt
        self.rew_terms["loss_facing_direction"] = -loss_facing_direction * self.step_dt
        self.rew_terms["loss_feet_roll"] = -loss_feet_roll * self.step_dt
        self.rew_terms["loss_unexpected_step"] = -loss_unexpected_step * self.step_dt
        self.rew_terms["loss_unexpected_swing"] = -loss_unexpected_swing * self.step_dt
        self.rew_terms["loss_feet_too_close"] = -loss_feet_too_close * self.step_dt
        self.rew_terms["loss_heel_down"] = -loss_heel_down * self.step_dt

    def compute_rew_and_reset(self):
        self.compute_rew_terms()

        terrain_height = self.terrain.terrain_heights(self.root_states[:, :2] + self.env_offset[:, :2])
        abnormal_vel = self.root_states[:, 7:].square().sum(dim=-1) > 50
        terminate = (self.root_states[:, 2] < 0.45 + terrain_height) | abnormal_vel

        self.rew_buf[:] = torch.clip(sum(self.rew_terms.values()), min=0.0) + self.rew_terms["loss_jacc"]

        self.timeout_buf[:] = self.ep_len >= self.ep_len_max
        self.done_buf[:] = self.timeout_buf | terminate
        self.timeout_buf |= self.ep_len == self.cmd_change_time
        env_to_reset = self.done_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_to_reset)

    def compute_obs(self):
        self.obs_buf[:, :3] = quat_rotate_inverse(self.root_states[:, 3:7], self.grav_vec) + torch.randn(self.num_envs, 3, device=self.device) * 0.01
        self.obs_buf[:, 3:6] = (
            quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 10:13]) + torch.randn(self.num_envs, 3, device=self.device) * 0.05
        )
        self.obs_buf[:, 6:9] = self.vel_cmd
        self.obs_buf[:, 9] = 0  # left blank for other purpose
        self.obs_buf[:, 10] = torch.cos(2 * np.pi * self.phase) * (self.phase_rate > -0.5).float()
        self.obs_buf[:, 11] = torch.sin(2 * np.pi * self.phase) * (self.phase_rate > -0.5).float()
        self.obs_buf[:, 12] = self.phase_rate
        self.obs_buf[:, 13 : 13 + self.num_dof] = self.dof_pos - self.q0 + torch.randn(self.num_envs, self.num_dof, device=self.device) * 0.01
        self.obs_buf[:, 13 + self.num_dof : 13 + 2 * self.num_dof] = (
            self.dof_vel * 0.1 + torch.randn(self.num_envs, self.num_dof, device=self.device) * 0.01
        )
        self.obs_buf[:, 13 + 2 * self.num_dof : 13 + 2 * self.num_dof + 4 * self.num_act] = self.action_history[:, :4, :].flatten(start_dim=1)

        local_vel = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])
        terrain_height = self.terrain.terrain_heights(self.root_states[:, :2] + self.env_offset[:, :2])
        self.privileged_obs_buf[:, 4:7] = local_vel + torch.randn(self.num_envs, 3, device=self.device) * 0.05
        self.privileged_obs_buf[:, 7] = self.root_states[:, 2] - terrain_height + torch.randn(self.num_envs, device=self.device) * 0.02
        self.privileged_obs_buf[:, 8:11] = self.pushing_forces[:, 0, :] / 20.0
        self.privileged_obs_buf[:, 11:14] = self.pushing_torques[:, 0, :] / 4.0

    @staticmethod
    def mirror_obs(obs):
        mirrored_obs = torch.matmul(T1RunActHistoryEnv.mirror_obs_mat.to(obs.device), obs.unsqueeze(-1)).squeeze(-1)
        return mirrored_obs

    @staticmethod
    def mirror_priv(privileged):
        mirrored_priv = torch.matmul(T1RunActHistoryEnv.mirror_priv_mat.to(privileged.device), privileged.unsqueeze(-1)).squeeze(-1)
        return mirrored_priv

    @staticmethod
    def mirror_act(act):
        mirrored_act = torch.matmul(T1RunActHistoryEnv.mirror_act_mat.to(act.device), act.unsqueeze(-1)).squeeze(-1)
        return mirrored_act
