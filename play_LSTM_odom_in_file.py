import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

from utils.wrapper_file import OdomStackingDataEnvFromFile
from utils.model import DenoisingRMA, OdomEstimator_wys_LSTM
from utils.dataset import Dataset

def umeyama_alignment(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    X = X.T
    Y = Y.T
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

if __name__ == "__main__":
    odom_path = "/home/luochangsheng/odom/logs/2025-05-26-19-36-57_file_0.02s_acc_actions/model_wys_64_file_0.02s_acc_actions.pt" # real data with acc, LSTM
    odom_path = "/home/luochangsheng/odom/logs/2025-05-26-20-57-40_file_LSTM_0.02s_actions/model_wys_1_file_LSTM_0.02s_actions.pt" # sim data without acc, LSTM
    odom_path = "/home/luochangsheng/odom/logs/2025-05-27-21-42-28_file_LSTM_0.02s_acc_actions/model_wys_1_file_LSTM_0.02s_acc_actions.pt" # after padding, LSTM
    
    data_dir = "/home/luochangsheng/odom/Enhanced_odom/data/test"
    DELTA_TIME = 0.02
    USE_ACC = True
    # USE_ACC = False
    USE_POS_SEQ = True
    USE_ACTIONS = True
    NAME = ""
    if DELTA_TIME == 0.02:
        NAME += "_0.02s"
    else:
        NAME += "_1.02s"
    if USE_ACC:
        NAME += "_acc"
    if not USE_POS_SEQ:
        NAME += "_no_pos_seq"
    if USE_ACTIONS:
        NAME += "_actions"
        
    num_obs_wys = 32
    if USE_ACC:
        num_obs_wys += 3
    if USE_ACTIONS:
        num_obs_wys += 11
    
    
    # odom_path = "/home/luochangsheng/odom/Legged_odom/logs/2025-04-02-12-09-59_0.02s_acc_no_pos_seq/model_wys_2000.pt"
    
    name_pred = "pred" + NAME
    name_gt = "gt" + NAME
    
    # data_dir = "/home/luochangsheng/odom/Legged_odom/data_mixed/segment_length=450"
    
    csv_file_paths = []
    num = 20
    # 随机抽取10个csv文件
    for data_file in os.listdir(data_dir):
        if data_file.endswith(".csv"):
            csv_file_paths.append(os.path.join(data_dir, data_file))
            num -= 1
            if num == 0:
                break
    csv_file_paths.sort()
    num_envs = len(csv_file_paths)
    
    env = OdomStackingDataEnvFromFile(csv_file_paths, obs_stacking=50, device="cuda:3")
    num_steps = env.num_rows[0]
    
    # odom_model_wys = OdomEstimator_wys(num_obs_wys + 4, env.obs_stacking).to(env.device)
    # optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)
    odom_model_wys = OdomEstimator_wys_LSTM(num_obs_wys, env.obs_stacking).to(env.device)
    odom_model_wys = torch.jit.load(odom_path).to(env.device)
    odom_model_wys.eval()
        
    
    
    # Load the model

    env = OdomStackingDataEnvFromFile(csv_file_paths, obs_stacking=50, device="cuda:3")
    infos = env.reset()
    if USE_ACC and USE_ACTIONS:
        # odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:], infos["odom_obs_history_wys"][..., -14:-11]), dim=-1).to(env.device)
        odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
    elif USE_ACC and not USE_ACTIONS:
        odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-11].to(env.device)
    elif not USE_ACC and USE_ACTIONS:
        odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:]), dim=-1).to(env.device)
    else:
        odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-14].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    pos_groundtruth = infos["pos_groundtruth"].to(env.device)
    # [batch_size, num_envs, num_stack, 2]
    pred_pos_history = torch.zeros(env.num_envs, env.obs_stacking + 3, 2, device=env.device)
    loss_list = list()
    
    output_dir = "/home/luochangsheng/odom/Legged_odom/output/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + NAME
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_envs):
        name = os.path.join(output_dir, name_gt + "_" + str(i) + ".txt")
        if not os.path.exists(name):
            with open(name, "w") as f:
                f.write("")
        else:
            os.remove(name)
            with open(name, "w") as f:
                f.write("")
        
        name = os.path.join(output_dir, name_pred + "_" + str(i) + ".txt")
        if not os.path.exists(name):
            with open(name, "w") as f:
                f.write("")
        else:
            os.remove(name)
            with open(name, "w") as f:
                f.write("")
    
    pred_pos = torch.zeros(env.num_envs, num_steps, 2, device=env.device)
    gt_pos = torch.zeros(env.num_envs, num_steps, 2, device=env.device)
        
    for i in range(num_steps):       
        infos, done = env.step()
        if done:
            break
        if USE_ACC and USE_ACTIONS:
            # odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:], infos["odom_obs_history_wys"][..., -14:-11]), dim=-1).to(env.device)
            odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
        elif USE_ACC and not USE_ACTIONS:
            odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-11].to(env.device)
        elif not USE_ACC and USE_ACTIONS:
            odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:]), dim=-1).to(env.device)
        else:
            odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-14].to(env.device)
        yaw_history = infos["yaw_history"].to(env.device)
        pos_history = infos["pos_history"].to(env.device)
        abs_yaw_history = infos["abs_yaw_history"].to(env.device) # env_nums, stack_nums(yaw_i)
        pos_groundtruth = infos["pos_groundtruth"].to(env.device) # env_nums, 2
        
        pos_input = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 2, 0].unsqueeze(1)) + torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 2, 1].unsqueeze(1)),
                -torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 2, 0].unsqueeze(1)) + torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 2, 1].unsqueeze(1))
            ),
            dim=-1
        )[:, 3:] # x_i - x_{i-1}
        yaw_input = abs_yaw_history[:, :] - abs_yaw_history[:, 1].unsqueeze(1)
        yaw_input = yaw_input[:, 2:]
        with torch.no_grad(): # 本次预测不需要梯度
            if not USE_POS_SEQ:
                pos_input = torch.zeros_like(pos_input)
            odom_pred_wys = odom_model_wys(
                odom_obs_history_wys[:, 1:],  # [envs, stack, obs]
                yaw_input,                    # [envs, stack]
                pos_input                     # [envs, stack, 2]
            )
        if DELTA_TIME == 0.02:
            index = -1
        else:
            index = -51
        odom_pred_wys_pos = torch.stack(
        (
            torch.cos(abs_yaw_history[:, 1]) * odom_pred_wys[:, 0] - torch.sin(abs_yaw_history[:, 1]) * odom_pred_wys[:, 1] + pred_pos_history[:, index, 0],
            torch.sin(abs_yaw_history[:, 1]) * odom_pred_wys[:, 0] + torch.cos(abs_yaw_history[:, 1]) * odom_pred_wys[:, 1] + pred_pos_history[:, index, 1]
        ),
        dim=-1
        ) # x_i+1
        pred_pos_history = torch.roll(pred_pos_history, -1, dims=1)
        # print("done.shape", done.shape)
        pred_pos_history[:, -1, :] = odom_pred_wys_pos
        
        x = pred_pos_history[:, -2, 0]
        y = pred_pos_history[:, -2, 1]
        timestamp_inter = env.timestamp_inter
        
        gt_x = pos_groundtruth[:, -1, 0]
        gt_y = pos_groundtruth[:, -1, 1]
        timestamp_mocap = env.timestamp_mocap
        
        pred_pos[:, i, 0] = x
        pred_pos[:, i, 1] = y
        gt_pos[:, i, 0] = gt_x
        gt_pos[:, i, 1] = gt_y
        # if i < 10 or i > num_steps-10:
        #     print(x[33].item(), y[33].item(), gt_x[33].item(), gt_y[33].item())

        for k in range(num_envs):
            name = os.path.join(output_dir, name_pred + "_" + str(k) + ".txt")
            # 将结果写入文件
            with open(name, "a") as f:
                f.write(f"{timestamp_inter[k]+0.02} {x[k].item()} {y[k].item()} 0 0 0 0 1\n")
            name = os.path.join(output_dir, name_gt + "_" + str(k) + ".txt")
            # 将结果写入文件
            with open(name, "a") as f:
                f.write(f"{timestamp_mocap[k]} {gt_x[k].item()} {gt_y[k].item()} 0 0 0 0 1\n")

    # Align trajectories
    pred_pos_np = pred_pos.cpu().numpy()
    gt_pos_np = gt_pos.cpu().numpy()
    aligned_pred_pos = np.zeros_like(pred_pos_np)

    for i in range(num_envs):
        scale, R_matrix, translation = umeyama_alignment(pred_pos_np[i], gt_pos_np[i])
        print(f"Environment {i}: Scale: {scale:.4f}" 
              f" Rotation: {R_matrix}\n"
              f" Translation: {translation}")
        aligned_pred_pos[i] = (
            scale * (R_matrix @ pred_pos_np[i].T).T + translation.T
        )

    aligned_pred_pos = torch.tensor(aligned_pred_pos, device=env.device)

    # Recalculate metrics
    ATE_u = torch.sqrt(torch.mean(torch.norm(aligned_pred_pos - gt_pos, dim=-1) ** 2, dim=-1))  # 计算ATE_o: 方均根
    ATE_o = torch.sqrt(torch.mean(torch.norm(pred_pos - gt_pos, dim=-1) ** 2, dim=-1))  # 计算ATE_u: 方均根
    rpe = torch.sqrt(torch.mean(torch.norm(
        aligned_pred_pos[:, 1:] - aligned_pred_pos[:, :-1] - (gt_pos[:, 1:] - gt_pos[:, :-1]), dim=-1) ** 2, dim=-1))  # 计算RPE
    pred_traj_length = torch.norm(aligned_pred_pos[:, 1:] - aligned_pred_pos[:, :-1], dim=-1).sum(dim=-1)  # 预测轨迹长度
    gt_traj_length = torch.norm(gt_pos[:, 1:] - gt_pos[:, :-1], dim=-1).sum(dim=-1)  # 真实轨迹长度
    duration = num_steps * DELTA_TIME  # 持续时间

    avg_ATE_o = ATE_o.mean().item()
    avg_ATE_u = ATE_u.mean().item()
    avg_rpe = rpe.mean().item()
    avg_pred_traj_length = pred_traj_length.mean().item()
    avg_gt_traj_length = gt_traj_length.mean().item()

    # for i in range(num_envs):
    #     print(f"Environment {i}:")
    #     print(f"  ATE_o: {ATE_o[i].item():.4f}")
    #     print(f"  ATE_u: {ATE_u[i].item():.4f}")
    #     print(f"  RPE: {rpe[i].item():.4f}")
    #     print(f"  Predicted Trajectory Length: {pred_traj_length[i].item():.4f}")
    #     print(f"  Ground Truth Trajectory Length: {gt_traj_length[i].item():.4f}")
    #     print(f"  Duration: {duration:.2f} seconds")

    # 保存每个环境的轨迹对比图
    for i in range(num_envs):
        plt.figure()
        plt.plot(pred_pos[i, :-1, 0].cpu(), pred_pos[i, :-1, 1].cpu(), label="Predicted Trajectory")
        plt.plot(aligned_pred_pos[i, :-1, 0].cpu(), aligned_pred_pos[i, :-1, 1].cpu(), label="Aligned Predicted Trajectory")
        plt.plot(gt_pos[i, :-1, 0].cpu(), gt_pos[i, :-1, 1].cpu(), label="Ground Truth Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Environment {i} Trajectory Comparison")
        plt.legend()
        plt.grid()

        # 添加指标信息
        metrics_text = (
            f"ATE_o: {ATE_o[i].item():.4f}\n"
            f"ATE_u: {ATE_u[i].item():.4f}\n"
            f"RPE: {rpe[i].item():.4f}\n"
            f"Predicted Length: {pred_traj_length[i].item():.4f}\n"
            f"Ground Truth Length: {gt_traj_length[i].item():.4f}"
        )
        plt.gcf().text(0.01, 0.83, metrics_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(os.path.join(output_dir, f"traj_{i}" + NAME + ".png"))
        plt.close()

    print("Average Metrics:")
    print(f"  Average ATE_o: {avg_ATE_o:.4f}")
    print(f"  Average ATE_u: {avg_ATE_u:.4f}")
    print(f"  Average RPE: {avg_rpe:.4f}")
    print(f"  Average Predicted Trajectory Length: {avg_pred_traj_length:.4f}")
    print(f"  Average Ground Truth Trajectory Length: {avg_gt_traj_length:.4f}")