import isaacgym

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper import ObsStackingEnvWrapperForOdom
from utils.model import DenoisingRMA, OdomEstimator_wys, OdomEstimator_Legolas, OdomEstimator_baseline
from utils.dataset import Dataset
from envs.T1_run_act_history import T1RunActHistoryEnv
import datetime

if __name__ == "__main__":
    dir = os.path.join("logs/traj", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 1, "cuda:0", False, curriculum=False, change_cmd=False) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)

    odom_model_wys = OdomEstimator_wys(35 + 4, env.obs_stacking).to(env.device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)

    odom_model_Legolas = OdomEstimator_Legolas(46 + 2, env.obs_stacking).to(env.device)
    optimizer_Legolas = torch.optim.Adam(odom_model_Legolas.parameters(), lr=3e-4)

    odom_model_baseline = OdomEstimator_baseline(45 + 3, env.obs_stacking).to(env.device)
    optimizer_baseline = torch.optim.Adam(odom_model_baseline.parameters(), lr=3e-4)

    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])
    recorder = SummaryWriter(dir)

    baseline_origin_pos = torch.zeros(env.num_envs, 2, device=env.device)
    baseline_origin_yaw = torch.zeros(env.num_envs, device=env.device)

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
    odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
    odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)

    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    pos_groundtruth = infos["pos_groundtruth"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    start_mask = infos["start_mask"].to(env.device)
    odom = infos["odom"].to(env.device)


    latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/models/model_wys.pt"
    if latest_wys_model_path:
        odom_model_wys = torch.jit.load(latest_wys_model_path).to(env.device)
        print(f"Loaded TorchScript model from {latest_wys_model_path}")
    
    latest_Legolas_model_path = "/home/lcs/RCL_Project/Legged_odom/models/model_Legolas.pt"
    if latest_Legolas_model_path:
        odom_model_Legolas = torch.jit.load(latest_Legolas_model_path).to(env.device)
        print(f"Loaded TorchScript model from {latest_Legolas_model_path}")
    
    latest_baseline_model_path = "/home/lcs/RCL_Project/Legged_odom/models/model_baseline.pt"
    if latest_baseline_model_path:
        odom_model_baseline = torch.jit.load(latest_baseline_model_path).to(env.device)
        print(f"Loaded TorchScript model from {latest_baseline_model_path}")

    odom_pred_wys_pos_list = list() # x_i
    for i in range(50):
        odom_pred_wys_pos_list.append(torch.zeros(env.num_envs, 2, device=env.device))
    
    odom_pred_Legolas_pos_list = list() # x_i
    odom_pred_Legolas_pos_list.append(torch.zeros(env.num_envs, 2, device=env.device))

    odom_groundtruth_list = list()
    odom_groundtruth_list.append(pos_groundtruth)

    odom_pred_baseline_pos_list = list() # x_i
    odom_pred_baseline_pos_list.append(torch.zeros(env.num_envs, 2, device=env.device))


    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化绘图
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    odom_pred_wys_pos_array = np.zeros((1000, env.num_envs, 2))
    odom_pred_Legolas_pos_array = np.zeros((1000, env.num_envs, 2))
    odom_pred_baseline_pos_array = np.zeros((1000, env.num_envs, 2))
    odom_groundtruth_array = np.zeros((1000, env.num_envs, 2))
    line1, = ax.plot([], [], label="odom_pred_pos_wys")
    line2, = ax.plot([], [], label="odom_groundtruth")
    line3, = ax.plot([], [], label="odom_pred_pos_Legolas")
    line4, = ax.plot([], [], label="odom_pred_pos_baseline")
    ax.legend()

    # 创建文件路径
    wys_file = os.path.join(dir, "odom_pred_wys_pos.txt")
    legolas_file = os.path.join(dir, "odom_pred_Legolas_pos.txt")
    groundtruth_file = os.path.join(dir, "odom_groundtruth.txt")
    baseline_file = os.path.join(dir, "odom_pred_baseline_pos.txt")

    # # 添加表头
    # with open(wys_file, "w") as f:
    #     f.write("timestamp x y z qx qy qz qw\n")
    # with open(legolas_file, "w") as f:
    #     f.write("timestamp x y z qx qy qz qw\n")
    # with open(groundtruth_file, "w") as f:
    #     f.write("timestamp x y z qx qy qz qw\n")
    # with open(baseline_file, "w") as f:
    #     f.write("timestamp x y z qx qy qz qw\n")

    for i in range(1000):
        obs_m = env.mirror_obs(obs)
        obs_history_m = env.mirror_obs(obs_history)
        with torch.no_grad():
            dist, _ = model.act(obs, obs_history)
            dist_m, _ = model.act(obs_m, obs_history_m)
            act_mean = (dist.loc + env.mirror_act(dist_m.loc)) * 0.5
            act_std = dist.scale
            act = act_mean + act_std * torch.randn_like(act_std)
        obs, rew, done, infos = env.step(act) # x_i
        obs_history = infos["obs_history"].to(env.device)
        odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
        odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
        yaw_history = infos["yaw_history"].to(env.device) # [num_envs, obs_stacking] yaw_{i-49} ~ yaw_i
        pos_history = infos["pos_history"].to(env.device)# [num_envs, obs_stacking + 1, 2] x_{i-50}-x_{i-50} ~ x_i - x_{i-50}
        pos_groundtruth = infos["pos_groundtruth"].to(env.device) # [num_envs, 2] x_i
        abs_yaw_history = infos["abs_yaw_history"].to(env.device)
        start_mask = infos["start_mask"].to(env.device)
        odom = infos["odom"].to(env.device)

        tmp_pos = torch.stack(odom_pred_wys_pos_list[-51:], dim=1) # 世界坐标系下的x_{i-50} ~ x_i（预测的）

        pos_input = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (tmp_pos[:, :, 0] - tmp_pos[:, 0, 0].unsqueeze(1)) + torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (tmp_pos[:, :, 1] - tmp_pos[:, 0, 1].unsqueeze(1)),
                -torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (tmp_pos[:, :, 0] - tmp_pos[:, 0, 0].unsqueeze(1)) + torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (tmp_pos[:, :, 1] - tmp_pos[:, 0, 1].unsqueeze(1))
            ),
            dim=-1
        ) # 输入的坐标要转为机器人初始坐标系下的坐标
        odom_pred_wys = odom_model_wys(odom_obs_history_wys[:, 1:], yaw_history[:, 2:], pos_input[:, 1:])
        
        odom_pred_wys_pos = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 1]) * odom_pred_wys[:, 0] - torch.sin(abs_yaw_history[:, 1]) * odom_pred_wys[:, 1] + odom_pred_wys_pos_list[-1][:, 0],
                torch.sin(abs_yaw_history[:, 1]) * odom_pred_wys[:, 0] + torch.cos(abs_yaw_history[:, 1]) * odom_pred_wys[:, 1] + odom_pred_wys_pos_list[-1][:, 1]
            ),
            dim=-1
        ) # x_i+1

        odom_pred_Legolas = odom_model_Legolas(odom_obs_history_Legolas, yaw_history)
        # odom_pred_Legolas = odom_model_Legolas(odom_obs_history_Legolas.unsqueeze(0), yaw_history.unsqueeze(0)).squeeze(0)

        # odom_pred_pos = odom_pred[i, :, :2] + odom_pred_pos_list[-1][:, :2] # x_i = d_i + x_i-1 
        odom_pred_Legolas_pos = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 1]) * odom_pred_Legolas[:, 0] - torch.sin(abs_yaw_history[:, 1]) * odom_pred_Legolas[:, 1] + odom_pred_Legolas_pos_list[-1][:, 0],
                torch.sin(abs_yaw_history[:, 1]) * odom_pred_Legolas[:, 0] + torch.cos(abs_yaw_history[:, 1]) * odom_pred_Legolas[:, 1] + odom_pred_Legolas_pos_list[-1][:, 1]
            ),
            dim=-1
        ) # x_i = d_i + x_i-1 [num_envs, 2]
        if abs(start_mask[0, -2] - 0.0) < 10e-2 and abs(start_mask[0, -1] - 1.0) < 10e-2:
            baseline_origin_pos[0] = odom_pred_baseline_pos_list[-1][0].clone()
            baseline_origin_yaw[0] = abs_yaw_history[0, -1].clone()

        odom_pred_baseline = odom_model_baseline(odom_obs_history_baseline, yaw_history, start_mask)
        odom_pred_baseline_pos = torch.stack(
            (
                torch.cos(baseline_origin_yaw[0])*odom_pred_baseline[:, 0] - torch.sin(baseline_origin_yaw[0])*odom_pred_baseline[:, 1] + baseline_origin_pos[0, 0],
                torch.sin(baseline_origin_yaw[0])*odom_pred_baseline[:, 0] + torch.cos(baseline_origin_yaw[0])*odom_pred_baseline[:, 1] + baseline_origin_pos[0, 1]
            ),
            dim=-1
        )
        odom_pred_baseline_pos_list.append(odom_pred_baseline_pos)

        groundtruth = pos_groundtruth[:, 0:2].clone() # [num_envs, 2]
        odom_pred_wys_pos_list.append(odom_pred_wys_pos) # add x_i
        odom_pred_Legolas_pos_list.append(odom_pred_Legolas_pos) # add x_i
        odom_pred_baseline_pos_list.append(odom_pred_baseline_pos)
        odom_groundtruth_list.append(groundtruth)

        # 获取当前 Unix 时间戳
        timestamp = time.time()

        # 写入 odom_pred_wys_pos_list
        with open(wys_file, "a") as f:
            for env_idx in range(env.num_envs):
                x, y = odom_pred_wys_pos[env_idx].detach().cpu().numpy()
                f.write(f"{timestamp} {x} {y} 0 0 0 0 1\n")  # TUM 格式

        # 写入 odom_pred_Legolas_pos_list
        with open(legolas_file, "a") as f:
            for env_idx in range(env.num_envs):
                x, y = odom_pred_Legolas_pos[env_idx].detach().cpu().numpy()
                f.write(f"{timestamp} {x} {y} 0.0 0.0 0.0 0.0 1.0\n")  # TUM 格式

        # 写入 odom_groundtruth_list
        with open(groundtruth_file, "a") as f:
            for env_idx in range(env.num_envs):
                x, y = groundtruth[env_idx].detach().cpu().numpy()
                f.write(f"{timestamp} {x} {y} 0.0 0.0 0.0 0.0 1.0\n")  # TUM 格式

        # 写入 odom_pred_baseline_pos_list
        with open(baseline_file, "a") as f:
            for env_idx in range(env.num_envs):
                x, y = odom_pred_baseline_pos[env_idx].detach().cpu().numpy()
                f.write(f"{timestamp} {x} {y} 0.0 0.0 0.0 0.0 1.0\n")  # TUM 格式

        # 更新绘图数据
        odom_pred_wys_pos_array[i] = odom_pred_wys_pos.detach().cpu().numpy()
        odom_groundtruth_array[i] = groundtruth.detach().cpu().numpy()
        odom_pred_Legolas_pos_array[i] = odom_pred_Legolas_pos.detach().cpu().numpy()
        odom_pred_baseline_pos_array[i] = odom_pred_baseline_pos.detach().cpu().numpy()
        line1.set_data(odom_pred_wys_pos_array[:i+1, 0, 0], odom_pred_wys_pos_array[:i+1, 0, 1])
        line2.set_data(odom_groundtruth_array[:i+1, 0, 0], odom_groundtruth_array[:i+1, 0, 1])
        line3.set_data(odom_pred_Legolas_pos_array[:i+1, 0, 0], odom_pred_Legolas_pos_array[:i+1, 0, 1])
        line4.set_data(odom_pred_baseline_pos_array[:i+1, 0, 0], odom_pred_baseline_pos_array[:i+1, 0, 1])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        


    # 关闭交互模式并显示最终图像
    plt.ioff()
    plt.show()



