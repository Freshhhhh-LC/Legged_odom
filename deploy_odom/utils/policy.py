import numpy as np
import torch
import logging
from utils.model import DenoisingRMA

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

mirror_obs_mat = _build_t1_mirror_obs_mat()
mirror_priv_mat = _build_t1_mirror_privileged_mat()
mirror_act_mat = _build_t1_mirror_action_mat()

def mirror_obs(obs):
    mirrored_obs = torch.matmul(mirror_obs_mat.to(obs.device), obs.unsqueeze(-1)).squeeze(-1)
    return mirrored_obs

def mirror_priv(privileged):
    mirrored_priv = torch.matmul(mirror_priv_mat.to(privileged.device), privileged.unsqueeze(-1)).squeeze(-1)
    return mirrored_priv

def mirror_act(act):
    mirrored_act = torch.matmul(mirror_act_mat.to(act.device), act.unsqueeze(-1)).squeeze(-1)
    return mirrored_act

class Policy:
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.delta_time = self.cfg["policy"]["delta_time"]
        self.use_accel = self.cfg["policy"]["use_accel"]
        self.policy = DenoisingRMA(
            11, 
            3 * 2 + 2 + 1 + 1 + 2 + 1 + 2 * 13 + 4 * 11,
            50,
            14,
            64,
        )
        try:
            state_dict = torch.load(self.cfg["policy"]["policy_path"], weights_only=True) 
            self.policy.load_state_dict(state_dict["model"])
            self.odom_policy = torch.jit.load(self.cfg["policy"]["odom_policy_path"])
            self.odom_policy.eval()
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)
        self.gait_frequency = -1
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.stacked_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_observations"]), dtype=np.float32)
        self.stacked_odom_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_odom_obs"]), dtype=np.float32)
        self.stacked_yaw = np.zeros(self.cfg["policy"]["num_stack"] + 1, dtype=np.float32)
        self.stacked_pos = np.zeros((self.cfg["policy"]["num_stack"] + 1, 2), dtype=np.float32)
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.base_yaw = 0
        self.stacked_obs_init = False
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.actions_history = np.zeros((4, self.cfg["policy"]["num_actions"]), dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        # self.q_inc = np.zeros(1, 12)
        self.q_inc = np.zeros(12, dtype=np.float32)
        self.q_inc[[0, 6]] = -np.pi / 12  # hip Y
        self.q_inc[[3, 9]] = np.pi / 6  # knee
        self.dof_pos_ref = np.zeros(23, dtype=np.float32)

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, base_yaw, vx, vy, vyaw, base_acc):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.base_yaw = base_yaw
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = -1
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = (
            self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[7] = (
            self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[8] = (
            self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[9] = 0
        self.obs[10] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[12] = 1 / self.gait_frequency
        self.obs[13:26] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[26:39] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[39:39 + 11 *4] = self.actions_history.flatten()
        
        
        if not self.stacked_obs_init:
            self.stacked_odom_obs[:, 0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
            self.stacked_odom_obs[:, 3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
            self.stacked_odom_obs[:, 6:19] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
            self.stacked_odom_obs[:, 19:32] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
            acc = np.array(base_acc, dtype=np.float32)
            local_gravity = projected_gravity * 9.81
            acc += local_gravity
            self.stacked_odom_obs[:, 32:35] = acc * 0.1
        self.stacked_odom_obs[:-1, :] = self.stacked_odom_obs[1:, :]
        self.stacked_odom_obs[-1, 0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.stacked_odom_obs[-1, 3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.stacked_odom_obs[-1, 6:19] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.stacked_odom_obs[-1, 19:32] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
        acc = np.array(base_acc, dtype=np.float32)
        local_gravity = projected_gravity * 9.81
        acc += local_gravity
        self.stacked_odom_obs[-1, 32:35] = acc * 0.1
        self.stacked_yaw[:-1] = self.stacked_yaw[1:]
        self.stacked_yaw[-1] = base_yaw
        if self.use_accel:
            obs_input = torch.from_numpy(self.stacked_odom_obs).unsqueeze(0)
        else:
            obs_input = torch.from_numpy(self.stacked_odom_obs[:, :-3]).unsqueeze(0)
        local_odom = (
            self.odom_policy(
                obs_input,
                torch.from_numpy(self.stacked_yaw - self.stacked_yaw[0])[1:].unsqueeze(0),
                torch.from_numpy(
                    np.stack(
                        (
                            np.cos(self.stacked_yaw[0]) * (self.stacked_pos[:, 0] - self.stacked_pos[0, 0])
                            + np.sin(self.stacked_yaw[0]) * (self.stacked_pos[:, 1] - self.stacked_pos[0, 1]),
                            -np.sin(self.stacked_yaw[0]) * (self.stacked_pos[:, 0] - self.stacked_pos[0, 0])
                            + np.cos(self.stacked_yaw[0]) * (self.stacked_pos[:, 1] - self.stacked_pos[0, 1]),
                        ),
                        axis=-1,
                    )
                )[1:].unsqueeze(0),
            )
            .detach()
            .numpy()
        )
        # print(local_odom)
        
        local_odom = local_odom[0]
        if self.delta_time == 0.02:
            index = -1
        else:
            index = -51
        self.odom_pos[0] = np.cos(self.stacked_yaw[0]) * local_odom[0] - np.sin(self.stacked_yaw[0]) * local_odom[1] + self.stacked_pos[index, 0]
        
        self.odom_pos[1] = np.sin(self.stacked_yaw[0]) * local_odom[0] + np.cos(self.stacked_yaw[0]) * local_odom[1] + self.stacked_pos[index, 1]
        
        self.stacked_pos[:-1, :] = self.stacked_pos[1:, :]
        self.stacked_pos[-1, :] = self.odom_pos
        
        obs_input = torch.from_numpy(self.obs).unsqueeze(0)
        stacked_obs_input = torch.from_numpy(self.stacked_obs).unsqueeze(0)
        obs_m = mirror_obs(obs_input)
        stacked_obs_m = mirror_obs(stacked_obs_input)
        with torch.no_grad():
            dist, _ = self.policy.act(obs_input, stacked_obs_input)
            dist_m, _ = self.policy.act(obs_m, stacked_obs_m)
            act_mean = (dist.loc + mirror_act(dist_m.loc)) * 0.5
            act_std = dist.scale
            act = act_mean + act_std * torch.randn_like(act_std)
        self.actions[:] = act.detach().numpy()
        self.actions_history[1:, :] = self.actions_history[:-1, :]
        self.actions_history[0, :] = self.actions
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        stand_cmd = self.gait_frequency < -0.5
        rescaled_phase = self.gait_process / 0.4
        phase_offset = np.array([0.05 / 0.4, 0.55 / 0.4], dtype=np.float32).reshape(1, 2)
        feet_phases = np.clip(rescaled_phase - phase_offset, a_min=0.0, a_max=1.0)
        self.j_inc_scale = (1.0 - np.cos(2 * np.pi * feet_phases)) / 2 if not stand_cmd else np.zeros_like(feet_phases)
        self.dof_pos_ref[:] = self.default_dof_pos
        self.dof_pos_ref[11:17] += self.q_inc[0:6] * self.j_inc_scale[0, 0:1]
        self.dof_pos_ref[17:23] += self.q_inc[6:12] * self.j_inc_scale[0, 1:2]
        self.dof_targets[:] = self.dof_pos_ref
        self.dof_targets[10:16] += self.actions[0:6]
        self.dof_targets[17:22] += self.actions[6:11]
        
        self.logger.debug(f"Odom: {self.odom_pos}")
        self.logger.debug(f"Yaw: {base_yaw}")

        return self.dof_targets, self.odom_pos

class OdomPolicy:
    def __init__(self, cfg, delta_time, use_accel):
        device = torch.device("cpu")
        self.logger = logging.getLogger(__name__)
        self.delta_time = delta_time
        self.use_accel = use_accel
        self.cfg = cfg
        try:
            print(self.cfg["policy"]["odom_policy_path"])
            self.odom_policy = torch.jit.load(self.cfg["policy"]["odom_policy_path"]).to(device)
            self.odom_policy.eval()
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.stacked_odom_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_odom_obs"]), dtype=np.float32)
        self.stacked_yaw = np.zeros(self.cfg["policy"]["num_stack"] + 1, dtype=np.float32)
        self.stacked_pos = np.zeros((self.cfg["policy"]["num_stack"] + 1, 2), dtype=np.float32)
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.base_yaw = 0
        self.stacked_obs_init = False
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)

    def inference(self, dof_pos, dof_vel, base_ang_vel, projected_gravity, base_yaw, base_acc):
        self.base_yaw = base_yaw
        self.stacked_odom_obs[:-1, :] = self.stacked_odom_obs[1:, :]
        self.stacked_odom_obs[-1, 0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.stacked_odom_obs[-1, 3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.stacked_odom_obs[-1, 6:19] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.stacked_odom_obs[-1, 19:32] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
        acc = np.array(base_acc, dtype=np.float32)
        local_gravity = projected_gravity * 9.81
        acc += local_gravity
        self.stacked_odom_obs[-1, 32:35] = acc * 0.1
        self.stacked_yaw[:-1] = self.stacked_yaw[1:]
        self.stacked_yaw[-1] = base_yaw
        if self.use_accel:
            obs_input = torch.from_numpy(self.stacked_odom_obs).unsqueeze(0)
        else:
            obs_input = torch.from_numpy(self.stacked_odom_obs[:, :-3]).unsqueeze(0)
        local_odom = (
            self.odom_policy(
                obs_input,
                torch.from_numpy(self.stacked_yaw - self.stacked_yaw[0])[1:].unsqueeze(0),
                torch.from_numpy(
                    np.stack(
                        (
                            np.cos(self.stacked_yaw[0]) * (self.stacked_pos[:, 0] - self.stacked_pos[0, 0])
                            + np.sin(self.stacked_yaw[0]) * (self.stacked_pos[:, 1] - self.stacked_pos[0, 1]),
                            -np.sin(self.stacked_yaw[0]) * (self.stacked_pos[:, 0] - self.stacked_pos[0, 0])
                            + np.cos(self.stacked_yaw[0]) * (self.stacked_pos[:, 1] - self.stacked_pos[0, 1]),
                        ),
                        axis=-1,
                    )
                )[1:].unsqueeze(0),
            )
            .detach()
            .numpy()
        )
        # print(local_odom)
        
        local_odom = local_odom[0]
        if self.delta_time == 0.02:
            index = -1
        else:
            index = -51
        self.odom_pos[0] = np.cos(self.stacked_yaw[0]) * local_odom[0] - np.sin(self.stacked_yaw[0]) * local_odom[1] + self.stacked_pos[index, 0]
        
        self.odom_pos[1] = np.sin(self.stacked_yaw[0]) * local_odom[0] + np.cos(self.stacked_yaw[0]) * local_odom[1] + self.stacked_pos[index, 1]
        
        self.stacked_pos[:-1, :] = self.stacked_pos[1:, :]
        self.stacked_pos[-1, :] = self.odom_pos
        self.logger.debug(f"Odom: {self.odom_pos}")
        self.logger.debug(f"Yaw: {base_yaw}")

        return self.odom_pos
