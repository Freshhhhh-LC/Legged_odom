import isaacgym

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper import ObsStackingEnvWrapperForOdom
from utils.model import DenoisingRMA, OdomEstimator_wys, OdomEstimator_Legolas
from utils.dataset import Dataset
from envs.T1_run_act_history import T1RunActHistoryEnv

if __name__ == "__main__":
    dir = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 1024, "cuda:0", True, curriculum=False, change_cmd=True) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)

    odom_model_wys = OdomEstimator_wys(32 + 4, env.obs_stacking).to(env.device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)

    odom_model_Legolas = OdomEstimator_Legolas(46 + 2, env.obs_stacking).to(env.device)
    optimizer_Legolas = torch.optim.Adam(odom_model_Legolas.parameters(), lr=3e-4)

    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])
    recorder = SummaryWriter(dir)

    buf = Dataset(24, env.num_envs)
    buf.AddBuffer("obs_history", (env.obs_stacking, env.num_obs), device=env.device)
    buf.AddBuffer("odom_obs_history_wys", (env.obs_stacking, 32), device=env.device)
    buf.AddBuffer("odom_obs_history_Legolas", (env.obs_stacking, 46), device=env.device)
    buf.AddBuffer("yaw_history", (env.obs_stacking,), device=env.device)
    buf.AddBuffer("pos_history", (env.obs_stacking + 1, 2), device=env.device)

    latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-13-10-18-11/model_wys_400.pth"
    if latest_wys_model_path:
        checkpoint = torch.load(latest_wys_model_path)
        odom_model_wys.load_state_dict(checkpoint['model'])
        optimizer_wys.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from {latest_wys_model_path}")
    
    latest_Legolas_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-13-10-18-11/model_Legolas_400.pth"
    if latest_Legolas_model_path:
        checkpoint = torch.load(latest_Legolas_model_path)
        odom_model_Legolas.load_state_dict(checkpoint['model'])
        optimizer_Legolas.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from {latest_Legolas_model_path}")

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
    odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)

    for i in range(60000):
        for j in range(24):
            buf.Record("obs_history", j, obs_history)
            buf.Record("odom_obs_history_wys", j, odom_obs_history_wys)
            buf.Record("odom_obs_history_Legolas", j, odom_obs_history_Legolas)
            buf.Record("yaw_history", j, yaw_history)
            buf.Record("pos_history", j, pos_history)
            obs_m = env.mirror_obs(obs)
            obs_history_m = env.mirror_obs(obs_history)
            with torch.no_grad():
                dist, _ = model.act(obs, obs_history)
                dist_m, _ = model.act(obs_m, obs_history_m)
                act_mean = (dist.loc + env.mirror_act(dist_m.loc)) * 0.5
                act_std = dist.scale
                act = act_mean + act_std * torch.randn_like(act_std)
            obs, rew, done, infos = env.step(act)
            obs_history = infos["obs_history"].to(env.device)
            odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
            odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
            yaw_history = infos["yaw_history"].to(env.device)
            pos_history = infos["pos_history"].to(env.device)

        odom_loss_list_wys = list()
        for j in range(20):
            odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"], buf["yaw_history"], buf["pos_history"][..., :-1, :])
            odom_loss_wys = F.mse_loss(odom_pred_wys, buf["pos_history"][..., -1, :])
            optimizer_wys.zero_grad()
            odom_loss_wys.backward()
            optimizer_wys.step()
            odom_loss_list_wys.append(odom_loss_wys.item())
        odom_loss_mean_wys = sum(odom_loss_list_wys) / len(odom_loss_list_wys)
        recorder.add_scalar("odom_loss_wys", odom_loss_mean_wys, i)

        if i % 10 == 9:
            print(f"iter: {i + 1}, \todom_loss_wys: {odom_loss_mean_wys}")
        if i % 100 == 99:
            torch.save({"model": odom_model_wys.state_dict(), "optimizer": optimizer_wys.state_dict()}, os.path.join(dir, f"model_wys_{i + 1}.pth"))
        
        odom_loss_list_Legolas = list()
        for j in range(20):
            odom_pred_Legolas = odom_model_Legolas(buf["odom_obs_history_Legolas"], buf["yaw_history"])
            pos_inc = buf["pos_history"][..., -1, :] - buf["pos_history"][..., -2 , :]
            odom_loss_Legolas = F.mse_loss(odom_pred_Legolas, pos_inc)
            optimizer_Legolas.zero_grad()
            odom_loss_Legolas.backward()
            optimizer_Legolas.step()
            odom_loss_list_Legolas.append(odom_loss_Legolas.item())
        odom_loss_mean_Legolas = sum(odom_loss_list_Legolas) / len(odom_loss_list_Legolas)
        recorder.add_scalar("odom_loss_Legolas", odom_loss_mean_Legolas, i)

        if i % 10 == 9:
            print(f"iter: {i + 1}, \todom_loss_Legolas: {odom_loss_mean_Legolas}")
        if i % 100 == 99:
            torch.save({"model": odom_model_Legolas.state_dict(), "optimizer": optimizer_Legolas.state_dict()}, os.path.join(dir, f"model_Legolas_{i + 1}.pth"))
