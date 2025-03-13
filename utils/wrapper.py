from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import get_euler_xyz

import torch


def ObsStackingEnvWrapperForOdom(base_env, obs_stacking, *args, **kwargs):
    class ObsStackingEnvImpl(base_env):
        def __init__(self, obs_stacking, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.obs_stacking = obs_stacking
            self.obs_history = torch.zeros(self.num_envs, self.obs_stacking, self.num_obs, device=self.device)

            self.odom_obs_history_wys = torch.zeros(self.num_envs, self.obs_stacking, 32, device=self.device)

            self.odom_obs_history_Legolas = torch.zeros(self.num_envs, self.obs_stacking, 46, device=self.device)

            self.yaw_history = torch.zeros(self.num_envs, self.obs_stacking, device=self.device)
            self.pos_history = torch.zeros(self.num_envs, self.obs_stacking + 1, 2, device=self.device)

        def reset(self):
            obs, infos = super().reset()
            self.obs_history[:, :, :] = obs.unsqueeze(1)
            self.odom_obs_history_wys[:, :, 0:6] = obs[:, 0:6].unsqueeze(1)
            self.odom_obs_history_wys[:, :, 6:32] = obs[:, 13:39].unsqueeze(1)

            self.odom_obs_history_Legolas[:, :, 0:9] = obs[:, 0:9].unsqueeze(1)
            self.odom_obs_history_Legolas[:, :, 9:46] = obs[:, 13:50].unsqueeze(1)

            _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.yaw_history[:, :] = yaw.unsqueeze(-1)
            infos.update(
                {
                    "obs_history": self.obs_history,
                    "odom_obs_history_wys": self.odom_obs_history_wys,
                    "odom_obs_history_Legolas": self.odom_obs_history_Legolas,
                    "yaw_history": torch.zeros_like(self.yaw_history),
                    "pos_history": torch.zeros_like(self.pos_history),
                    "pos_groundtruth": self.root_states[:, 0:2],
                    "abs_yaw_history": self.yaw_history,
                }
            )
            return obs, infos

        def step(self, *args, **kwargs):
            obs, rew, done, infos = super().step(*args, **kwargs)
            self.obs_history = torch.roll(self.obs_history, -1, dims=1)
            self.obs_history[done, :, :] = obs[done].unsqueeze(1)
            self.obs_history[:, -1, :] = obs

            self.odom_obs_history_wys[:] = torch.roll(self.odom_obs_history_wys, -1, dims=1)
            self.odom_obs_history_wys[done, :, 0:6] = obs[done, 0:6].unsqueeze(1)
            self.odom_obs_history_wys[done, :, 6:32] = obs[done, 13:39].unsqueeze(1)
            self.odom_obs_history_wys[:, -1, 0:6] = obs[:, 0:6]
            self.odom_obs_history_wys[:, -1, 6:32] = obs[:, 13:39]

            self.odom_obs_history_Legolas[:] = torch.roll(self.odom_obs_history_Legolas, -1, dims=1)
            self.odom_obs_history_Legolas[done, :, 0:9] = obs[done, 0:9].unsqueeze(1)
            self.odom_obs_history_Legolas[done, :, 9:46] = obs[done, 13:50].unsqueeze(1)
            self.odom_obs_history_Legolas[:, -1, 0:9] = obs[:, 0:9]
            self.odom_obs_history_Legolas[:, -1, 9:46] = obs[:, 13:50]

            self.yaw_history = torch.roll(self.yaw_history, -1, dims=1)
            _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
            self.yaw_history[done, :] = yaw[done].unsqueeze(-1)
            self.yaw_history[:, -1] = yaw

            self.pos_history[:] = torch.roll(self.pos_history, -1, dims=1)
            self.pos_history[done, :, 0:2] = self.root_states[done, 0:2].unsqueeze(1)
            self.pos_history[:, -1, 0:2] = self.root_states[:, 0:2]

            infos.update(
                {
                    "obs_history": self.obs_history,
                    "odom_obs_history_wys": self.odom_obs_history_wys,
                    "odom_obs_history_Legolas": self.odom_obs_history_Legolas,
                    "yaw_history": (self.yaw_history - self.yaw_history[:, 0].unsqueeze(1)),
                    "pos_history": torch.stack(
                        (
                            torch.cos(self.yaw_history[:, 0].unsqueeze(1)) * (self.pos_history[:, :, 0] - self.pos_history[:, 0, 0].unsqueeze(1))
                            + torch.sin(self.yaw_history[:, 0].unsqueeze(1)) * (self.pos_history[:, :, 1] - self.pos_history[:, 0, 1].unsqueeze(1)),
                            -torch.sin(self.yaw_history[:, 0].unsqueeze(1)) * (self.pos_history[:, :, 0] - self.pos_history[:, 0, 0].unsqueeze(1))
                            + torch.cos(self.yaw_history[:, 0].unsqueeze(1)) * (self.pos_history[:, :, 1] - self.pos_history[:, 0, 1].unsqueeze(1)),
                        ),
                        dim=-1,
                    ),
                    "pos_groundtruth": self.root_states[:, 0:2],
                    "abs_yaw_history": self.yaw_history,
                }
            )

            return obs, rew, done, infos

    return ObsStackingEnvImpl(obs_stacking, *args, **kwargs)
