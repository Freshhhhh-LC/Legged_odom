import numpy as np
import torch
import logging


class Policy:
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        try:
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
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
        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.stacked_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_observations"]), dtype=np.float32)
        self.stacked_odom_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_odom_obs"]), dtype=np.float32)
        self.stacked_yaw = np.zeros(self.cfg["policy"]["num_stack"], dtype=np.float32)
        self.stacked_pos = np.zeros((self.cfg["policy"]["num_stack"], 2), dtype=np.float32)
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.base_yaw = 0
        self.stacked_obs_init = False
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, base_yaw, ball_pos):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.base_yaw = base_yaw

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = np.cos(-base_yaw) * (ball_pos[0] - self.odom_pos[0]) - np.sin(-base_yaw) * (ball_pos[1] - self.odom_pos[1])
        self.obs[7] = np.sin(-base_yaw) * (ball_pos[0] - self.odom_pos[0]) + np.cos(-base_yaw) * (ball_pos[1] - self.odom_pos[1])
        self.obs[8] = np.cos(-base_yaw) * (7 - self.odom_pos[0]) - np.sin(-base_yaw) * (-self.odom_pos[1])
        self.obs[9] = np.sin(-base_yaw) * (7 - self.odom_pos[0]) + np.cos(-base_yaw) * (-self.odom_pos[1])
        self.obs[10] = np.cos(base_yaw)
        self.obs[11] = np.sin(base_yaw)
        self.obs[12] = np.cos(2 * np.pi * self.gait_process)
        self.obs[13] = np.sin(2 * np.pi * self.gait_process)
        self.obs[14:16] = (dof_pos - self.default_dof_pos)[:2] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[16:29] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[29:31] = dof_vel[:2] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[31:44] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[44:57] = self.actions
        if not self.stacked_obs_init:
            self.stacked_obs[:] = self.obs[np.newaxis, :]
            self.stacked_odom_obs[:, 0:6] = self.obs[np.newaxis, 0:6]
            self.stacked_odom_obs[:, 6:19] = self.obs[np.newaxis, 16:29]
            self.stacked_odom_obs[:, 19:32] = self.obs[np.newaxis, 31:44]
            self.stacked_yaw[:] = base_yaw
            self.stacked_obs_init = True
        self.stacked_obs[1:, :] = self.stacked_obs[:-1, :]
        self.stacked_obs[0, :] = self.obs
        self.stacked_odom_obs[1:, :] = self.stacked_odom_obs[:-1, :]
        self.stacked_odom_obs[0, 0:6] = self.obs[0:6]
        self.stacked_odom_obs[0, 6:19] = self.obs[16:29]
        self.stacked_odom_obs[0, 19:32] = self.obs[31:44]
        self.stacked_yaw[1:] = self.stacked_yaw[:-1]
        self.stacked_yaw[0] = base_yaw

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0), torch.from_numpy(self.stacked_obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[0:2] += self.cfg["policy"]["control"]["action_scale"] * self.actions[0:2]
        self.dof_targets[10:16] += self.cfg["policy"]["control"]["action_scale"] * self.actions[2:8]
        self.dof_targets[17:22] += self.cfg["policy"]["control"]["action_scale"] * self.actions[8:13]
        local_odom = (
            self.odom_policy(
                torch.from_numpy(self.stacked_odom_obs).unsqueeze(0),
                torch.from_numpy(self.stacked_yaw - self.stacked_yaw[-1]).unsqueeze(0),
                torch.from_numpy(
                    np.stack(
                        (
                            np.cos(self.stacked_yaw[-1]) * (self.stacked_pos[:, 0] - self.stacked_pos[-1, 0])
                            + np.sin(self.stacked_yaw[-1]) * (self.stacked_pos[:, 1] - self.stacked_pos[-1, 1]),
                            -np.sin(self.stacked_yaw[-1]) * (self.stacked_pos[:, 0] - self.stacked_pos[-1, 0])
                            + np.cos(self.stacked_yaw[-1]) * (self.stacked_pos[:, 1] - self.stacked_pos[-1, 1]),
                        ),
                        axis=-1,
                    )
                ).unsqueeze(0),
            )
            .detach()
            .numpy()
        )[0]
        self.odom_pos[0] = np.cos(self.stacked_yaw[-1]) * local_odom[0] - np.sin(self.stacked_yaw[-1]) * local_odom[1] + self.stacked_pos[-1, 0]
        self.odom_pos[1] = np.sin(self.stacked_yaw[-1]) * local_odom[0] + np.cos(self.stacked_yaw[-1]) * local_odom[1] + self.stacked_pos[-1, 1]
        self.stacked_pos[1:, :] = self.stacked_pos[:-1, :]
        self.stacked_pos[0, :] = self.odom_pos
        self.logger.debug(f"Odom: {self.odom_pos}")
        self.logger.debug(f"Yaw: {base_yaw}")

        return self.dof_targets

class OdomPolicy:
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        try:
            self.odom_policy = torch.jit.load(self.cfg["policy"]["odom_policy_path"])
            self.odom_policy.eval()
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.stacked_odom_obs = np.zeros((self.cfg["policy"]["num_stack"], self.cfg["policy"]["num_odom_obs"]), dtype=np.float32)
        self.stacked_yaw = np.zeros(self.cfg["policy"]["num_stack"], dtype=np.float32)
        self.stacked_pos = np.zeros((self.cfg["policy"]["num_stack"], 2), dtype=np.float32)
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.base_yaw = 0
        self.stacked_obs_init = False
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, dof_pos, dof_vel, base_ang_vel, projected_gravity, base_yaw, base_acc):
        self.base_yaw = base_yaw
        self.stacked_odom_obs[1:, :] = self.stacked_odom_obs[:-1, :]
        self.stacked_odom_obs[0, 0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.stacked_odom_obs[0, 0:3] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.stacked_odom_obs[0, 6:19] = (dof_pos - self.default_dof_pos)[10:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.stacked_odom_obs[0, 19:32] = dof_vel[10:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.stacked_odom_obs[0, 32:35] = base_acc * self.cfg["policy"]["normalization"]["base_acc"]
        self.stacked_yaw[1:] = self.stacked_yaw[:-1]
        self.stacked_yaw[0] = base_yaw
        local_odom = (
            self.odom_policy(
                torch.from_numpy(self.stacked_odom_obs).unsqueeze(0),
                torch.from_numpy(self.stacked_yaw - self.stacked_yaw[-1]).unsqueeze(0),
                torch.from_numpy(
                    np.stack(
                        (
                            np.cos(self.stacked_yaw[-1]) * (self.stacked_pos[:, 0] - self.stacked_pos[-1, 0])
                            + np.sin(self.stacked_yaw[-1]) * (self.stacked_pos[:, 1] - self.stacked_pos[-1, 1]),
                            -np.sin(self.stacked_yaw[-1]) * (self.stacked_pos[:, 0] - self.stacked_pos[-1, 0])
                            + np.cos(self.stacked_yaw[-1]) * (self.stacked_pos[:, 1] - self.stacked_pos[-1, 1]),
                        ),
                        axis=-1,
                    )
                ).unsqueeze(0),
            )
            .detach()
            .numpy()
        )[0]
        self.odom_pos[0] = np.cos(self.stacked_yaw[-1]) * local_odom[0] - np.sin(self.stacked_yaw[-1]) * local_odom[1] + self.stacked_pos[0, 0]
        
        self.odom_pos[1] = np.sin(self.stacked_yaw[-1]) * local_odom[0] + np.cos(self.stacked_yaw[-1]) * local_odom[1] + self.stacked_pos[0, 1]
        
        self.stacked_pos[1:, :] = self.stacked_pos[:-1, :]
        self.stacked_pos[0, :] = self.odom_pos
        self.logger.debug(f"Odom: {self.odom_pos}")
        self.logger.debug(f"Yaw: {base_yaw}")

        return self.odom_pos
    