import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

from utils.wrapper_file import OdomStackingDataEnvFromFile
from utils.model import DenoisingRMA, OdomEstimator_baseline
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
    odom_path = "/home/luochangsheng/odom/Legged_odom/logs/2025-04-10-16-58-36_0.02s_no_pos_seq_actions/model_baseline_2000.pt"
    data_dir = "/home/luochangsheng/odom/Legged_odom/data/segment_length=2000"
    NAME = "Baseline"
    
    name_pred = "pred" + NAME
    name_gt = "gt" + NAME
    
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
    
    odom_model_baseline = OdomEstimator_baseline(45 + 3, env.obs_stacking).to(env.device)
    optimizer_baseline = torch.optim.Adam(odom_model_baseline.parameters(), lr=3e-4)
    odom_model_baseline = torch.jit.load(odom_path).to(env.device)
    odom_model_baseline.eval()
    
    # Load the model

    env = OdomStackingDataEnvFromFile(csv_file_paths, obs_stacking=50, device="cuda:3")
    infos = env.reset()
    odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    pos_groundtruth = infos["pos_groundtruth"].to(env.device)
    start_mask = infos["start_mask"].to(env.device)
    odom = infos["odom"].to(env.device)
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
    baseline_origin_pos = torch.zeros(env.num_envs, 2, device=env.device)
    baseline_origin_yaw = torch.zeros(env.num_envs, device=env.device)
        
    for i in range(num_steps):       
        infos, done = env.step()
        if done:
            break
        odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
        yaw_history = infos["yaw_history"].to(env.device)
        pos_history = infos["pos_history"].to(env.device)
        abs_yaw_history = infos["abs_yaw_history"].to(env.device) # env_nums, stack_nums(yaw_i)
        pos_groundtruth = infos["pos_groundtruth"].to(env.device) # env_nums, 2
        start_mask = infos["start_mask"].to(env.device) # env_nums, stack_nums
        odom = infos["odom"].to(env.device) # env_nums, 2
        
        # if abs(start_mask[0, -2] - 0.0) < 10e-2 and abs(start_mask[0, -1] - 1.0) < 10e-2:
        update_mask = ((start_mask[:, -1] - 1.0).abs() < 10e-2) & ((start_mask[:, -2] - 0.0).abs() < 10e-2)
        baseline_origin_pos[update_mask] = pred_pos_history[update_mask, -1, 0:2]
        baseline_origin_yaw[update_mask] = yaw_history[update_mask, -1]
        
        # odom_pred_baseline = odom_model_baseline(odom_obs_history_baseline, yaw_history[:, 2:], start_mask)
        odom_pred_baseline = odom
        odom_pred_baseline_pos = torch.stack(
            (
                torch.cos(baseline_origin_yaw) * odom_pred_baseline[:, 0] - torch.sin(baseline_origin_yaw) * odom_pred_baseline[:, 1] + baseline_origin_pos[:, 0],
                torch.sin(baseline_origin_yaw) * odom_pred_baseline[:, 0] + torch.cos(baseline_origin_yaw) * odom_pred_baseline[:, 1] + baseline_origin_pos[:, 1],
            ),
            dim=-1,
        )
        
        pred_pos_history = torch.roll(pred_pos_history, -1, dims=1)
        # print("done.shape", done.shape)
        pred_pos_history[:, -1, :] = odom_pred_baseline_pos
        
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
    pred_pos_np = pred_pos.detach().cpu().numpy()
    gt_pos_np = gt_pos.detach().cpu().numpy()
    aligned_pred_pos = np.zeros_like(pred_pos_np)

    for i in range(num_envs):
        scale, R_matrix, translation = umeyama_alignment(pred_pos_np[i], gt_pos_np[i])
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
        # plt.plot(pred_pos[i, :-1, 0].cpu(), pred_pos[i, :-1, 1].cpu(), label="Predicted Trajectory")
        plt.plot(pred_pos[i, :-1, 0].detach().cpu(), pred_pos[i, :-1, 1].detach().cpu(), label="Predicted Trajectory")
        plt.plot(aligned_pred_pos[i, :-1, 0].detach().cpu(), aligned_pred_pos[i, :-1, 1].detach().cpu(), label="Aligned Predicted Trajectory")
        plt.plot(gt_pos[i, :-1, 0].detach().cpu(), gt_pos[i, :-1, 1].detach().cpu(), label="Ground Truth Trajectory")
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