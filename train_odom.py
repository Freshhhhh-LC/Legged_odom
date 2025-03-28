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

if __name__ == "__main__":
    dir = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 256, "cuda:0", True, curriculum=False, change_cmd=True) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
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

    buf = Dataset(24, env.num_envs)
    buf.AddBuffer("obs_history", (env.obs_stacking, env.num_obs), device=env.device)
    buf.AddBuffer("odom_obs_history_wys", (env.obs_stacking, 35), device=env.device)
    buf.AddBuffer("odom_obs_history_Legolas", (env.obs_stacking, 46), device=env.device)
    buf.AddBuffer("odom_obs_history_baseline", (env.obs_stacking, 45), device=env.device)
    buf.AddBuffer("yaw_history", (env.obs_stacking,), device=env.device)
    buf.AddBuffer("pos_history", (env.obs_stacking + 1, 2), device=env.device)
    buf.AddBuffer("pred_pos_history", (env.obs_stacking + 1, 2), device=env.device)
    buf.AddBuffer("abs_yaw_history", (env.obs_stacking,), device=env.device)
    buf.AddBuffer("start_mask", (env.obs_stacking,), device=env.device)
    buf.AddBuffer("odom", (2,), device=env.device)

    # latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-17-15-19-07/model_wys_500.pth"
    # if latest_wys_model_path:
    #     checkpoint = torch.load(latest_wys_model_path)
    #     odom_model_wys.load_state_dict(checkpoint['model'])
    #     optimizer_wys.load_state_dict(checkpoint['optimizer'])
    #     print(f"Loaded model from {latest_wys_model_path}")
    
    # latest_Legolas_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-13-10-18-11/model_Legolas_400.pth"
    # if latest_Legolas_model_path:
    #     checkpoint = torch.load(latest_Legolas_model_path)
    #     odom_model_Legolas.load_state_dict(checkpoint['model'])
    #     optimizer_Legolas.load_state_dict(checkpoint['optimizer'])
    #     print(f"Loaded model from {latest_Legolas_model_path}")

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
    pred_pos_history = torch.zeros(env.num_envs, env.obs_stacking + 1, 2, device=env.device)

    for i in range(2000):
        for j in range(24):
            buf.Record("obs_history", j, obs_history)
            buf.Record("odom_obs_history_wys", j, odom_obs_history_wys)
            buf.Record("odom_obs_history_Legolas", j, odom_obs_history_Legolas)
            buf.Record("odom_obs_history_baseline", j, odom_obs_history_baseline)
            buf.Record("yaw_history", j, yaw_history)
            buf.Record("pos_history", j, pos_history)
            buf.Record("abs_yaw_history", j, abs_yaw_history)
            buf.Record("start_mask", j, start_mask)
            buf.Record("odom", j, odom)
            buf.Record("pred_pos_history", j, pred_pos_history) #用前面的所有buf来预测出来的pos，记前面的为yaw_i，则这里为x_i-48~x_i+1 (50)
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
            odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
            yaw_history = infos["yaw_history"].to(env.device)
            pos_history = infos["pos_history"].to(env.device)
            abs_yaw_history = infos["abs_yaw_history"].to(env.device) # env_nums, stack_nums(yaw_i)
            start_mask = infos["start_mask"].to(env.device)
            odom = infos["odom"].to(env.device)
            pos_input = torch.stack(
                (
                    torch.cos(abs_yaw_history[:, 0].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 1, 0].unsqueeze(1)) + torch.sin(abs_yaw_history[:, 0].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 1, 1].unsqueeze(1)),
                    -torch.sin(abs_yaw_history[:, 0].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 1, 0].unsqueeze(1)) + torch.cos(abs_yaw_history[:, 0].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 1, 1].unsqueeze(1))
                ),
                dim=-1
            )
            with torch.no_grad(): # 本次预测不需要梯度
                odom_pred_wys = odom_model_wys(odom_obs_history_wys, yaw_history, pos_input[:, 1:]) # 预测的x_i+1 - x_i
            odom_pred_wys_pos = torch.stack(
            (
                torch.cos(abs_yaw_history[:, 0]) * odom_pred_wys[:, 0] - torch.sin(abs_yaw_history[:, 0]) * odom_pred_wys[:, 1] + pred_pos_history[:, -1, 0],
                torch.sin(abs_yaw_history[:, 0]) * odom_pred_wys[:, 0] + torch.cos(abs_yaw_history[:, 0]) * odom_pred_wys[:, 1] + pred_pos_history[:, -1, 1]
            ),
            dim=-1
            ) # x_i+1
            pred_pos_history = torch.roll(pred_pos_history, -1, dims=1)
            # print("done.shape", done.shape)
            pred_pos_history[done,:,0:2] = odom_pred_wys_pos[done,0:2].unsqueeze(1)
            pred_pos_history[:, -1, :] = odom_pred_wys_pos
            
        use_pred_pos = True
        odom_loss_list_wys = list()
        for j in range(20):
            if use_pred_pos==False:
                odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"], buf["yaw_history"], buf["pos_history"][..., :-1, :])
            else:
                pos_input = torch.stack(
                    (
                        torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1)),
                        -torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1))
                    ),
                    dim=-1
                )
                odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"], buf["yaw_history"], pos_input[:, :, :-1, :])
            odom_loss_wys = F.mse_loss(odom_pred_wys, buf["pos_history"][..., -1, :] - buf["pos_history"][..., -2, :])
            optimizer_wys.zero_grad()
            odom_loss_wys.backward(retain_graph=True)
            optimizer_wys.step()
            odom_loss_list_wys.append(odom_loss_wys.item())
        odom_loss_mean_wys = sum(odom_loss_list_wys) / len(odom_loss_list_wys)
        recorder.add_scalar("odom_loss_wys", odom_loss_mean_wys, i)

        if i % 10 == 9:
            print(f"iter: {i + 1}, \todom_loss_wys: {odom_loss_mean_wys}")
        if i % 10 == 9:
            # 保存为 TorchScript 格式
            odom_model_wys.eval()
            odom_model_wys.cpu()
            scripted_model = torch.jit.script(odom_model_wys)
            scripted_model.save(os.path.join(dir, f"model_wys_{i + 1}.pt"))
            odom_model_wys.to(env.device)
            odom_model_wys.train()
        
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
            # 保存为 TorchScript 格式
            scripted_model = torch.jit.script(odom_model_Legolas)
            scripted_model.save(os.path.join(dir, f"model_Legolas_{i + 1}.pt"))
        
        odom_loss_list_baseline = list()
        for j in range(20):
            odom_pred_baseline = odom_model_baseline(buf["odom_obs_history_baseline"], buf["yaw_history"], buf["start_mask"])

            odom_loss_baseline = F.mse_loss(odom_pred_baseline, buf["odom"])
            optimizer_baseline.zero_grad()
            odom_loss_baseline.backward()
            optimizer_baseline.step()
            odom_loss_list_baseline.append(odom_loss_baseline.item())
        odom_loss_mean_baseline = sum(odom_loss_list_baseline) / len(odom_loss_list_baseline)
        recorder.add_scalar("odom_loss_baseline", odom_loss_mean_baseline, i)

        if i % 10 == 9:
            print(f"iter: {i + 1}, \todom_loss_baseline: {odom_loss_mean_baseline}")
        if i % 100 == 99:
            # 保存为 TorchScript 格式
            scripted_model = torch.jit.script(odom_model_baseline)
            scripted_model.save(os.path.join(dir, f"model_baseline_{i + 1}.pt"))