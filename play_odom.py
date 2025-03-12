import isaacgym

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper import ObsStackingEnvWrapperForOdom
from utils.model import DenoisingRMA, OdomEstimator
from utils.dataset import Dataset
from envs.T1_run_act_history import T1RunActHistoryEnv

if __name__ == "__main__":
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 1, "cuda:0", False, curriculum=False, change_cmd=False) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)
    odom_model = OdomEstimator(46 + 2, env.obs_stacking).to(env.device)
    optimizer = torch.optim.Adam(odom_model.parameters(), lr=3e-4)
    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])

    latest_model_path = "logs/Legolas/model_3500.pth"
    if latest_model_path:
        checkpoint = torch.load(latest_model_path)
        odom_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from {latest_model_path}")

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    odom_obs_history = infos["odom_obs_history"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    pos_groundtruth = infos["pos_groundtruth"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    stacked_odom_pos = torch.zeros(env.num_envs, env.obs_stacking, 2, device=env.device)

    odom_pred_pos_list = list() # x_i
    # 添加五十个零向量
    for i in range(50):
        odom_pred_pos_list.append(torch.zeros(env.num_envs, 2, device=env.device))
    odom_groundtruth_list = list()

    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化绘图
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    odom_pred_pos_array = np.zeros((1000, env.num_envs, 2))
    odom_groundtruth_array = np.zeros((1000, env.num_envs, 2))
    line1, = ax.plot([], [], label="odom_pred_pos")
    line2, = ax.plot([], [], label="odom_groundtruth")
    ax.legend()

    for i in range(1000):
        obs_m = env.mirror_obs(obs)
        obs_history_m = env.mirror_obs(obs_history)
        with torch.no_grad():
            dist, _ = model.act(obs, obs_history)
            dist_m, _ = model.act(obs_m, obs_history_m)
            act_mean = (dist.loc + env.mirror_act(dist_m.loc)) * 0.5
            act_std = dist.scale
            act = act_mean + act_std * torch.randn_like(act_std)

        #odom_pred = d_i = x_i - x_{i-50}
        # odom_pred = odom_model(buf["odom_obs_history"], buf["yaw_history"])
        odom_pred = odom_model(odom_obs_history, yaw_history)

        # odom_pred_pos = odom_pred[i, :, :2] + odom_pred_pos_list[-1][:, :2] # x_i = d_i + x_i-1 
        odom_pred_pos = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 0]) * odom_pred[:, 0] - torch.sin(abs_yaw_history[:, 0]) * odom_pred[:, 1] + odom_pred_pos_list[-1][:, 0],
                torch.sin(abs_yaw_history[:, 0]) * odom_pred[:, 0] + torch.cos(abs_yaw_history[:, 0]) * odom_pred[:, 1] + odom_pred_pos_list[-1][:, 1]
            ),
            dim=-1
        ) # x_i = d_i + x_i-1 [num_envs, 2]

        groundtruth = pos_groundtruth[:, :2].clone()
        odom_pred_pos_list.append(odom_pred_pos) # add x_i，加到列表最后
        odom_groundtruth_list.append(groundtruth)

        # 更新绘图数据
        odom_pred_pos_array[i] = odom_pred_pos.detach().cpu().numpy()
        odom_groundtruth_array[i] = groundtruth.detach().cpu().numpy()
        line1.set_data(odom_pred_pos_array[:i+1, 0, 0], odom_pred_pos_array[:i+1, 0, 1])
        line2.set_data(odom_groundtruth_array[:i+1, 0, 0], odom_groundtruth_array[:i+1, 0, 1])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
        
        obs, rew, done, infos = env.step(act)
        obs_history = infos["obs_history"].to(env.device)
        odom_obs_history = infos["odom_obs_history"].to(env.device)
        yaw_history = infos["yaw_history"].to(env.device) # [num_envs, obs_stacking] yaw_{i-49} ~ yaw_i
        pos_history = infos["pos_history"].to(env.device)# [num_envs, obs_stacking + 1, 2] x_{i-50}-x_{i-50} ~ x_i - x_{i-50},此处没用上
        pos_groundtruth = infos["pos_groundtruth"].to(env.device) # [num_envs, 2] x_i
        abs_yaw_history = infos["abs_yaw_history"].to(env.device)

    # 关闭交互模式并显示最终图像
    plt.ioff()
    plt.show()



