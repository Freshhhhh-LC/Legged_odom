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
    dir = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 1024, "cuda:0", True, curriculum=False, change_cmd=True) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)
    odom_model = OdomEstimator(32 + 4, env.obs_stacking).to(env.device)
    optimizer = torch.optim.Adam(odom_model.parameters(), lr=3e-4)
    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])
    recorder = SummaryWriter(dir)
    buf = Dataset(24, env.num_envs)
    buf.AddBuffer("obs_history", (env.obs_stacking, env.num_obs), device=env.device)
    buf.AddBuffer("odom_obs_history", (env.obs_stacking, 32), device=env.device)
    buf.AddBuffer("yaw_history", (env.obs_stacking,), device=env.device)
    buf.AddBuffer("pos_history", (env.obs_stacking + 1, 2), device=env.device)

    # latest_model_path = 
    # if latest_model_path:
    #     odom_model.load_state_dict(torch.load(latest_model_path))
    #     print(f"Loaded model from {latest_model_path}")

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    odom_obs_history = infos["odom_obs_history"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)

    for i in range(60000):
        for j in range(24):
            buf.Record("obs_history", j, obs_history)
            buf.Record("odom_obs_history", j, odom_obs_history)
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
            odom_obs_history = infos["odom_obs_history"].to(env.device)
            yaw_history = infos["yaw_history"].to(env.device)
            pos_history = infos["pos_history"].to(env.device)

        odom_loss_list = list()
        for j in range(20):
            odom_pred = odom_model(buf["odom_obs_history"], buf["yaw_history"], buf["pos_history"][..., :-1, :])
            odom_loss = F.mse_loss(odom_pred, buf["pos_history"][..., -1, :])
            optimizer.zero_grad()
            odom_loss.backward()
            optimizer.step()
            odom_loss_list.append(odom_loss.item())
        odom_loss_mean = sum(odom_loss_list) / len(odom_loss_list)
        recorder.add_scalar("odom_loss", odom_loss_mean, i)

        if i % 10 == 9:
            print(f"iter: {i + 1}, \todom_loss: {odom_loss_mean}")
        if i % 100 == 99:
            torch.save({"model": odom_model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(dir, f"model_{i + 1}.pth"))
