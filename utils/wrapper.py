from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import get_euler_xyz

import torch


def ObsStackingEnvWrapperForOdom(base_env, obs_stacking, *args, **kwargs):
    class ObsStackingEnvImpl(base_env):
        def __init__(self, obs_stacking, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.obs_stacking = obs_stacking
            self.obs_history = torch.zeros(self.num_envs, self.obs_stacking, self.num_obs, device=self.device)
            self.odom_obs_history = torch.zeros(self.num_envs, self.obs_stacking, 45, device=self.device)
            self.yaw_history = torch.zeros(self.num_envs, self.obs_stacking, device=self.device)
            self.start_mask = torch.zeros(self.num_envs, self.obs_stacking, device=self.device)
            self.origin_pos = torch.zeros(self.num_envs, 2, device=self.device)
            self.origin_yaw = torch.zeros(self.num_envs, device=self.device)
            

        def reset(self):
            obs, infos = super().reset()
            self.obs_history[:, :, :] = obs.unsqueeze(1)
            self.odom_obs_history[:, :, 0:6] = obs[:, 0:6].unsqueeze(1)
            self.odom_obs_history[:, :, 6:8] = obs[:, 10:12].unsqueeze(1)
            self.odom_obs_history[:, :, 8:45] = obs[:, 13:50].unsqueeze(1)
            _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.yaw_history[:, :] = yaw.unsqueeze(-1)
            self.start_mask[:] = 0.0
            self.origin_pos[:] = self.root_states[:, 0:2]
            self.origin_yaw[:] = yaw
            infos.update(
                {
                    "obs_history": self.obs_history,
                    "odom_obs_history": self.odom_obs_history,
                    "yaw_history": torch.zeros_like(self.yaw_history),
                    "start_mask": self.start_mask,
                    "odom": torch.zeros_like(self.origin_pos),
                }
            )
            return obs, infos

        def step(self, *args, **kwargs):
            obs, rew, done, infos = super().step(*args, **kwargs)
            self.obs_history = torch.roll(self.obs_history, -1, dims=1)
            self.obs_history[done, :, :] = obs[done].unsqueeze(1)
            self.obs_history[:, -1, :] = obs

            self.odom_obs_history[:] = torch.roll(self.odom_obs_history, -1, dims=1)
            self.odom_obs_history[done, :, 0:6] = obs[done, 0:6].unsqueeze(1)
            self.odom_obs_history[done, :, 6:8] = obs[done, 10:12].unsqueeze(1)
            self.odom_obs_history[done, :, 8:45] = obs[done, 13:50].unsqueeze(1)
            self.odom_obs_history[:, -1, 0:6] = obs[:, 0:6]
            self.odom_obs_history[:, -1, 6:8] = obs[:, 10:12]
            self.odom_obs_history[:, -1, 8:45] = obs[:, 13:50]

            self.yaw_history = torch.roll(self.yaw_history, -1, dims=1)
            _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.yaw_history[done, :] = yaw[done].unsqueeze(-1)
            self.yaw_history[:, -1] = yaw

            self.start_mask = torch.roll(self.start_mask, -1, dims=1)
            self.start_mask[done, :] = 0.0
            self.start_mask[:, -1] = 1.0

            infos.update(
                {
                    "obs_history": self.obs_history,
                    "odom_obs_history": self.odom_obs_history,
                    "yaw_history": (self.yaw_history - self.origin_yaw.unsqueeze(-1)) * (1 - done.float()).unsqueeze(-1),
                    "start_mask": self.start_mask.clone(),
                    "odom": torch.stack(
                        (
                            torch.cos(self.origin_yaw) * (self.root_states[:, 0] - self.origin_pos[:, 0])
                            + torch.sin(self.origin_yaw) * (self.root_states[:, 1] - self.origin_pos[:, 1]),
                            -torch.sin(self.origin_yaw) * (self.root_states[:, 0] - self.origin_pos[:, 0])
                            + torch.cos(self.origin_yaw) * (self.root_states[:, 1] - self.origin_pos[:, 1]),
                        ),
                        dim=-1,
                    )
                    * (1 - done.float()).unsqueeze(-1),
                }
            )

            phase_inc = (self.phase_rate * 1.0 + 1.0) * self.step_dt
            odom_reset = (
                (self.phase_rate < -0.5)
                | ((self.phase >= 0.5) & (self.phase - phase_inc < 0.5))
                | ((self.phase >= 0.0) & (self.phase - phase_inc < 0.0))
                | done
            )
            
            self.start_mask[odom_reset, :] = 0.0
            self.origin_pos[odom_reset, :] = self.root_states[odom_reset, 0:2]
            self.origin_yaw[odom_reset] = yaw[odom_reset]

            return obs, rew, done, infos

    return ObsStackingEnvImpl(obs_stacking, *args, **kwargs)
