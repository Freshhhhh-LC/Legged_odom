import isaacgym
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from isaacgym.torch_utils import get_euler_xyz
import csv
import numpy as np

from utils.wrapper import ObsStackingEnvWrapperForOdom
from utils.model import DenoisingRMA, OdomEstimator_wys, OdomEstimator_Legolas, OdomEstimator_baseline
from utils.dataset import Dataset
from envs.T1_run_act_history import T1RunActHistoryEnv
from isaacgym.torch_utils import *

if __name__ == "__main__":
    data_dir = os.path.join("data_sim", "segments")
    data_length = 48000
    num_envs = 256
    os.makedirs(data_dir, exist_ok=True)  # 创建 data 文件夹
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, num_envs, "cuda:3", True, curriculum=False, change_cmd=True) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)

    odom_model_wys = OdomEstimator_wys(35 + 4, env.obs_stacking).to(env.device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)

    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])
    recorder = SummaryWriter(dir)

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
    odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
    odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    start_mask = infos["start_mask"].to(env.device)
    odom = infos["odom"].to(env.device)
    # [batch_size, num_envs, num_stack, 2]
    pred_pos_history = torch.zeros(env.num_envs, env.obs_stacking + 3, 2, device=env.device)

    for i in range(data_length):
        if i % 100 == 0:
            print(f"Step: {i}")
        obs_m = env.mirror_obs(obs)
        obs_history_m = env.mirror_obs(obs_history)
        with torch.no_grad():
            dist, _ = model.act(obs, obs_history)
            dist_m, _ = model.act(obs_m, obs_history_m)
            act_mean = (dist.loc + env.mirror_act(dist_m.loc)) * 0.5
            act_std = dist.scale
            act = act_mean + act_std * torch.randn_like(act_std)
        obs, rew, done, infos = env.step(act)
        print("obs shape:", obs.shape)
        obs_history = infos["obs_history"].to(env.device)
        times = torch.zeros(env.num_envs, device=env.device).cpu().numpy()
        _, _, yaw = get_euler_xyz(env.root_states[:, 3:7])
        yaw = yaw.cpu().numpy()
        projected_gravity = (quat_rotate_inverse(env.root_states[:, 3:7], env.grav_vec) + torch.randn(env.num_envs, 3, device=env.device) * 0.01).cpu().numpy()
        ang_vel = (quat_rotate_inverse(env.root_states[:, 3:7], env.root_states[:, 10:13]) + torch.randn(env.num_envs, 3, device=env.device) * 0.05).cpu().numpy()
        acc = env.root_acc[:, 0:3]
        local_acc = (quat_rotate_inverse(env.root_states[:, 3:7], acc) + torch.randn_like(acc) * 0.01).cpu().numpy()
        lin_acc = local_acc
        q = (env.dof_pos + torch.randn(env.num_envs, env.num_dof, device=env.device) * 0.01).cpu().numpy()  # [256, 13]
        q = np.concatenate([np.zeros((env.num_envs, 10)), q], axis=1)  # [256, 23]
        dq = (env.dof_vel + torch.randn(env.num_envs, env.num_dof, device=env.device) * 0.01).cpu().numpy()
        dq = np.concatenate([np.zeros((env.num_envs, 10)), dq], axis=1)
        mocap_time = torch.zeros(env.num_envs, device=env.device).cpu().numpy()
        mocap_timestamp = torch.zeros(env.num_envs, device=env.device).cpu().numpy()
        robot_pos = env.root_states[:, 0:2].cpu().numpy()
        ball_pos = torch.zeros(env.num_envs, 2, device=env.device).cpu().numpy()
        actions = torch.zeros(env.num_envs, 11, device=env.device).cpu().numpy()
        actions = act.clone()

        for k in range(env.num_envs):
            
            # 为每个环境生成单独的文件路径
            env_file_path = os.path.join(data_dir, f"segment_{k}.csv")
            # 如果文件不存在，写入表头
            
            if not os.path.exists(env_file_path):
                with open(env_file_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([
                        "time", "yaw", "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z", "lin_acc_x", "lin_acc_y", "lin_acc_z",
                        "q_0", "q_1", "q_2", "q_3", "q_4", "q_5", "q_6", "q_7", "q_8", "q_9", "q_10", "q_11", "q_12",
                        "q_13", "q_14", "q_15", "q_16", "q_17", "q_18", "q_19", "q_20", "q_21", "q_22",
                        "dq_0", "dq_1", "dq_2", "dq_3", "dq_4", "dq_5", "dq_6", "dq_7", "dq_8", "dq_9", "dq_10", "dq_11",
                        "dq_12", "dq_13", "dq_14", "dq_15", "dq_16", "dq_17", "dq_18", "dq_19", "dq_20", "dq_21", "dq_22",
                        "mocap_time", "mocap_timestamp", "robot_x", "robot_y", "robot_yaw", "ball_x", "ball_y"
                    ] + [f"obs_{i}" for i in range(83)]
                    )

            # 写入数据到对应环境的文件
            with open(env_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    times[k], yaw[k].item(), projected_gravity[k, 0].item(), projected_gravity[k, 1].item(),
                    projected_gravity[k, 2].item(), ang_vel[k, 0].item(), ang_vel[k, 1].item(), ang_vel[k, 2].item(),
                    lin_acc[k, 0].item(), lin_acc[k, 1].item(), lin_acc[k, 2].item(),
                    q[k, 0].item(), q[k, 1].item(), q[k, 2].item(), q[k, 3].item(), q[k, 4].item(),
                    q[k, 5].item(), q[k, 6].item(), q[k, 7].item(), q[k, 8].item(), q[k, 9].item(),
                    q[k, 10].item(), q[k, 11].item(), q[k, 12].item(), q[k, 13].item(), q[k, 14].item(),
                    q[k, 15].item(), q[k, 16].item(), q[k, 17].item(), q[k, 18].item(), q[k, 19].item(),
                    q[k, 20].item(), q[k, 21].item(), q[k, 22].item(),
                    dq[k, 0].item(), dq[k, 1].item(), dq[k, 2].item(), dq[k, 3].item(), dq[k, 4].item(),
                    dq[k, 5].item(), dq[k, 6].item(), dq[k, 7].item(), dq[k, 8].item(), dq[k, 9].item(),
                    dq[k, 10].item(), dq[k, 11].item(), dq[k, 12].item(), dq[k, 13].item(), dq[k, 14].item(),
                    dq[k, 15].item(), dq[k, 16].item(), dq[k, 17].item(), dq[k, 18].item(), dq[k, 19].item(),
                    dq[k, 20].item(), dq[k, 21].item(), dq[k, 22].item(),
                    mocap_time[k].item(), mocap_timestamp[k].item(),
                    robot_pos[k, 0].item(), robot_pos[k, 1].item(), yaw[k].item(),
                    ball_pos[k, 0].item(), ball_pos[k, 1].item(),
                    obs[k, 0].item(), obs[k, 1].item(), obs[k, 2].item(), obs[k, 3].item(), obs[k, 4].item(),
                    obs[k, 5].item(), obs[k, 6].item(), obs[k, 7].item(), obs[k, 8].item(), obs[k, 9].item(),
                    obs[k, 10].item(), obs[k, 11].item(), obs[k, 12].item(), obs[k, 13].item(), obs[k, 14].item(),
                    obs[k, 15].item(), obs[k, 16].item(), obs[k, 17].item(), obs[k, 18].item(), obs[k, 19].item(),
                    obs[k, 20].item(), obs[k, 21].item(), obs[k, 22].item(), obs[k, 23].item(), obs[k, 24].item(),
                    obs[k, 25].item(), obs[k, 26].item(), obs[k, 27].item(), obs[k, 28].item(), obs[k, 29].item(),
                    obs[k, 30].item(), obs[k, 31].item(), obs[k, 32].item(), obs[k, 33].item(), obs[k, 34].item(),
                    obs[k, 35].item(), obs[k, 36].item(), obs[k, 37].item(), obs[k, 38].item(), obs[k, 39].item(),
                    obs[k, 40].item(), obs[k, 41].item(), obs[k, 42].item(), obs[k, 43].item(), obs[k, 44].item(),
                    obs[k, 45].item(), obs[k, 46].item(), obs[k, 47].item(), obs[k, 48].item(), obs[k, 49].item(),
                    obs[k, 50].item(), obs[k, 51].item(), obs[k, 52].item(), obs[k, 53].item(), obs[k, 54].item(),
                    obs[k, 55].item(), obs[k, 56].item(), obs[k, 57].item(), obs[k, 58].item(), obs[k, 59].item(),
                    obs[k, 60].item(), obs[k, 61].item(), obs[k, 62].item(), obs[k, 63].item(), obs[k, 64].item(),
                    obs[k, 65].item(), obs[k, 66].item(), obs[k, 67].item(), obs[k, 68].item(), obs[k, 69].item(),
                    obs[k, 70].item(), obs[k, 71].item(), obs[k, 72].item(), obs[k, 73].item(), obs[k, 74].item(),
                    obs[k, 75].item(), obs[k, 76].item(), obs[k, 77].item(), obs[k, 78].item(), obs[k, 79].item(),
                    obs[k, 80].item(), obs[k, 81].item(), obs[k, 82].item(),
                ])
