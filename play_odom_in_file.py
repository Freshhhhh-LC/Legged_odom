import os
import time
import csv
import torch
import torch.nn.functional as F

from utils.model import OdomEstimator_wys, OdomEstimator_Legolas, OdomEstimator_baseline
from utils.wrapper_file import OdomStackingDataEnvFromFile

OBS_STACKING = 50
DATA_INDEX = "4"
DATA_NAME = "booster_seg_9.82s.csv"

if __name__ == "__main__":
    dir  = os.path.join("data", DATA_INDEX, "segments")
    data_file_path = os.path.join(dir, DATA_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    odom_model_wys = OdomEstimator_wys(32 + 4, OBS_STACKING).to(device)

    # odom_model_Legolas = OdomEstimator_Legolas(46 + 2, OBS_STACKING).to(device)

    # odom_model_baseline = OdomEstimator_baseline(45 + 3, OBS_STACKING).to(device)
    # baseline_origin_pos = torch.zeros(2, device=device)
    # baseline_origin_yaw = torch.zeros(device=device)
    
    env  = OdomStackingDataEnvFromFile(data_file_path, OBS_STACKING, device)

    infos = env.reset()
    odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
    # odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(device)
    # odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(device)
    yaw_history = infos["yaw_history"].to(device)
    pos_history = infos["pos_history"].to(device)
    pos_groundtruth = infos["pos_groundtruth"].to(device)
    abs_yaw_history = infos["abs_yaw_history"].to(device)
    # start_mask = infos["start_mask"].to(device)
    # odom = infos["odom"].to(device)


    # latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/retrain_model_wys_10.pth"
    latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-24-21-46-21/model_wys_900.pth"
    if latest_wys_model_path:
        checkpoint = torch.load(latest_wys_model_path)
        odom_model_wys.load_state_dict(checkpoint['model'])
        print(f"Loaded model from {latest_wys_model_path}")
    
    # latest_Legolas_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-18-10-43-25/model_Legolas_1000.pth"
    # if latest_Legolas_model_path:
    #     checkpoint = torch.load(latest_Legolas_model_path)
    #     odom_model_Legolas.load_state_dict(checkpoint['model'])
    #     print(f"Loaded model from {latest_Legolas_model_path}")
    
    # latest_baseline_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-18-10-43-25/model_baseline_1000.pth"
    # if latest_baseline_model_path:
    #     checkpoint = torch.load(latest_baseline_model_path)
    #     odom_model_baseline.load_state_dict(checkpoint['model'])
    #     print(f"Loaded model from {latest_baseline_model_path}")

    odom_pred_wys_pos_list = list() # x_i
    # 添加五十个零向量
    for i in range(50):
        odom_pred_wys_pos_list.append(torch.zeros(2, device=device))
    # odom_pred_Legolas_pos_list = list() # x_i
    # # 添加五十个零向量
    # for i in range(50):
    #     odom_pred_Legolas_pos_list.append(torch.zeros(2, device=device))
    odom_groundtruth_list = list()

    # odom_pred_baseline_pos_list = list() # x_i
    # for i in range(50):
    #     odom_pred_baseline_pos_list.append(torch.zeros(2, device=device))


    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化绘图
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    num_steps = env.num_rows
    odom_pred_wys_pos_array = np.zeros((num_steps, 1, 2))
    # odom_pred_Legolas_pos_array = np.zeros((num_steps, 1, 2))
    # odom_pred_baseline_pos_array = np.zeros((num_steps, 1, 2))
    odom_groundtruth_array = np.zeros((num_steps, 1, 2))
    line1, = ax.plot([], [], label="odom_pred_pos_wys")
    line2, = ax.plot([], [], label="odom_groundtruth")
    # line3, = ax.plot([], [], label="odom_pred_pos_Legolas")
    # line4, = ax.plot([], [], label="odom_pred_pos_baseline")
    ax.legend()

    for i in range(num_steps):
        infos, done = env.step()
        odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
        # odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(device)
        # odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(device)
        yaw_history = infos["yaw_history"].to(device) # [num_envs, obs_stacking] yaw_{i-49} ~ yaw_i
        pos_history = infos["pos_history"].to(device)# [num_envs, obs_stacking + 1, 2] x_{i-50}-x_{i-50} ~ x_i - x_{i-50}
        pos_groundtruth = infos["pos_groundtruth"].to(device) # [num_envs, 2] x_i
        abs_yaw_history = infos["abs_yaw_history"].to(device)
        # start_mask = infos["start_mask"].to(device)
        # odom = infos["odom"].to(device)

        tmp_pos = torch.stack(odom_pred_wys_pos_list[-50:], dim=0) # 世界坐标系下的x_{i-50} ~ x_i（预测的）

        pos_input = torch.stack(
            (
                torch.cos(abs_yaw_history[0].unsqueeze(0)) * (tmp_pos[:, 0] - tmp_pos[0, 0].unsqueeze(0)) + torch.sin(abs_yaw_history[0].unsqueeze(0)) * (tmp_pos[:, 1] - tmp_pos[0, 1].unsqueeze(0)),
                -torch.sin(abs_yaw_history[0].unsqueeze(0)) * (tmp_pos[:, 0] - tmp_pos[0, 0].unsqueeze(0)) + torch.cos(abs_yaw_history[0].unsqueeze(0)) * (tmp_pos[:, 1] - tmp_pos[0, 1].unsqueeze(0))
            ),
            dim=-1
        ) # 输入的坐标要转为机器人初始坐标系下的坐标
        odom_pred_wys = odom_model_wys(odom_obs_history_wys, yaw_history, pos_input)
        # print("shape", odom_pred_wys.shape)
        print("loss", F.mse_loss(odom_pred_wys, pos_history[-1] - pos_history[-2]))
        
        odom_pred_wys_pos = torch.stack(
            (
                torch.cos(abs_yaw_history[0]) * odom_pred_wys[0] - torch.sin(abs_yaw_history[0]) * odom_pred_wys[1] + odom_pred_wys_pos_list[-1][0],
                torch.sin(abs_yaw_history[0]) * odom_pred_wys[0] + torch.cos(abs_yaw_history[0]) * odom_pred_wys[1] + odom_pred_wys_pos_list[-1][1]
            ),
            dim=-1
        ) # x_i = d_i + x_i-1[num_envs, 2]

        # odom_pred_Legolas = odom_model_Legolas(odom_obs_history_Legolas, yaw_history)
        # # odom_pred_Legolas = odom_model_Legolas(odom_obs_history_Legolas.unsqueeze(0), yaw_history.unsqueeze(0)).squeeze(0)

        # # odom_pred_pos = odom_pred[i, :, :2] + odom_pred_pos_list[-1][:, :2] # x_i = d_i + x_i-1 
        # odom_pred_Legolas_pos = torch.stack(
        #     (
        #         torch.cos(abs_yaw_history[:, 0]) * odom_pred_Legolas[:, 0] - torch.sin(abs_yaw_history[:, 0]) * odom_pred_Legolas[:, 1] + odom_pred_Legolas_pos_list[-1][:, 0],
        #         torch.sin(abs_yaw_history[:, 0]) * odom_pred_Legolas[:, 0] + torch.cos(abs_yaw_history[:, 0]) * odom_pred_Legolas[:, 1] + odom_pred_Legolas_pos_list[-1][:, 1]
        #     ),
        #     dim=-1
        # ) # x_i = d_i + x_i-1 [num_envs, 2]
        # if abs(start_mask[0, -2] - 0.0) < 10e-2 and abs(start_mask[0, -1] - 1.0) < 10e-2:
        #     baseline_origin_pos[0] = odom_pred_baseline_pos_list[-1][0].clone()
        #     baseline_origin_yaw[0] = abs_yaw_history[0, -1].clone()

        # odom_pred_baseline = odom_model_baseline(odom_obs_history_baseline, yaw_history, start_mask)
        # odom_pred_baseline_pos = torch.stack(
        #     (
        #         torch.cos(baseline_origin_yaw[0])*odom_pred_baseline[:, 0] - torch.sin(baseline_origin_yaw[0])*odom_pred_baseline[:, 1] + baseline_origin_pos[0, 0],
        #         torch.sin(baseline_origin_yaw[0])*odom_pred_baseline[:, 0] + torch.cos(baseline_origin_yaw[0])*odom_pred_baseline[:, 1] + baseline_origin_pos[0, 1]
        #     ),
        #     dim=-1
        # )
        # odom_pred_baseline_pos_list.append(odom_pred_baseline_pos)

        groundtruth = pos_groundtruth[0:2].clone() # [num_envs, 2]
        odom_pred_wys_pos_list.append(odom_pred_wys_pos) # add x_i
        # odom_pred_Legolas_pos_list.append(odom_pred_Legolas_pos) # add x_i
        # odom_pred_baseline_pos_list.append(odom_pred_baseline_pos)
        # odom_groundtruth_list.append(groundtruth)

        # 更新绘图数据
        odom_pred_wys_pos_array[i] = odom_pred_wys_pos.detach().cpu().numpy()
        odom_groundtruth_array[i] = groundtruth.detach().cpu().numpy()
        # odom_pred_Legolas_pos_array[i] = odom_pred_Legolas_pos.detach().cpu().numpy()
        # odom_pred_baseline_pos_array[i] = odom_pred_baseline_pos.detach().cpu().numpy()
        line1.set_data(odom_pred_wys_pos_array[:i+1, 0, 0], odom_pred_wys_pos_array[:i+1, 0, 1])
        line2.set_data(odom_groundtruth_array[:i+1, 0, 0], odom_groundtruth_array[:i+1, 0, 1])
        # line3.set_data(odom_pred_Legolas_pos_array[:i+1, 0, 0], odom_pred_Legolas_pos_array[:i+1, 0, 1])
        # line4.set_data(odom_pred_baseline_pos_array[:i+1, 0, 0], odom_pred_baseline_pos_array[:i+1, 0, 1])
        ax.relim()
        ax.autoscale_view()
        # plt.draw()
        plt.pause(0.01)


    # 关闭交互模式并显示最终图像
    plt.ioff()
    plt.savefig("odom_pred.png")
    plt.show()


            
