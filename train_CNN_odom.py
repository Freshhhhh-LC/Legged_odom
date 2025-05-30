import isaacgym

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper import ObsStackingEnvWrapperForOdom
from utils.model import DenoisingRMA, OdomEstimator_wys_CNN
from utils.dataset import Dataset
from envs.T1_run_act_history import T1RunActHistoryEnv

if __name__ == "__main__":
    DELTA_TIME = 0.02
    USE_ACC = False
    USE_POS_SEQ = True
    USE_ACTIONS = True
    NAME = "_CNN"
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
    
    dir = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + NAME)
    os.makedirs(dir, exist_ok=True)
    env = ObsStackingEnvWrapperForOdom(T1RunActHistoryEnv, 50, 256, "cuda:1", True, curriculum=False, change_cmd=True) # T1RunActHistoryEnv, 50, 4096, "cuda:0", True, curriculum=False, change_cmd=True
    model = DenoisingRMA(env.num_act, env.num_obs, env.obs_stacking, env.num_privileged_obs, 64).to(env.device)

    
    odom_model_wys = OdomEstimator_wys_CNN(num_obs_wys, env.obs_stacking).to(env.device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)

    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    model.load_state_dict(state_dict["model"])
    recorder = SummaryWriter(dir)

    buf = Dataset(24, env.num_envs)
    buf.AddBuffer("obs_history", (env.obs_stacking, env.num_obs), device=env.device)
    buf.AddBuffer("odom_obs_history_wys", (env.obs_stacking + 1, num_obs_wys), device=env.device)
    buf.AddBuffer("odom_obs_history_Legolas", (env.obs_stacking, 46), device=env.device)
    buf.AddBuffer("odom_obs_history_baseline", (env.obs_stacking, 45), device=env.device)
    buf.AddBuffer("yaw_history", (env.obs_stacking + 2,), device=env.device)
    buf.AddBuffer("pos_history", (env.obs_stacking + 2, 2), device=env.device)
    buf.AddBuffer("pred_pos_history", (env.obs_stacking + 3, 2), device=env.device)
    buf.AddBuffer("abs_yaw_history", (env.obs_stacking + 2,), device=env.device)
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
    
    # latest_LSTM_model_path = "/home/luochangsheng/odom/Legged_odom/logs/2025-05-27-00-03-03_LSTM_0.02s_actions/model_wys_2000.pt"
    # if latest_LSTM_model_path:
    #     odom_model_wys = torch.jit.load(latest_LSTM_model_path).to(env.device)
    #     odom_model_wys.train()
    #     optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)
    #     print(f"Loaded model from {latest_LSTM_model_path}")

    obs, infos = env.reset()
    obs_history = infos["obs_history"].to(env.device)
    if USE_ACC and USE_ACTIONS:
        odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
    elif USE_ACC and not USE_ACTIONS:
        odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-11].to(env.device)
    elif not USE_ACC and USE_ACTIONS:
        odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:]), dim=-1).to(env.device)
    else:
        odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-14].to(env.device)
    odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
    odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
    yaw_history = infos["yaw_history"].to(env.device)
    pos_history = infos["pos_history"].to(env.device)
    abs_yaw_history = infos["abs_yaw_history"].to(env.device)
    start_mask = infos["start_mask"].to(env.device)
    odom = infos["odom"].to(env.device)
    # [batch_size, num_envs, num_stack, 2]
    pred_pos_history = torch.zeros(env.num_envs, env.obs_stacking + 3, 2, device=env.device)

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
            buf.Record("pred_pos_history", j, pred_pos_history)
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
            if USE_ACC and USE_ACTIONS:
                odom_obs_history_wys = infos["odom_obs_history_wys"].to(env.device)
            elif USE_ACC and not USE_ACTIONS:
                odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-11].to(env.device)
            elif not USE_ACC and USE_ACTIONS:
                odom_obs_history_wys = torch.cat((infos["odom_obs_history_wys"][..., :-14], infos["odom_obs_history_wys"][..., -11:]), dim=-1).to(env.device)
            else:
                odom_obs_history_wys = infos["odom_obs_history_wys"][..., :-14].to(env.device)
            odom_obs_history_Legolas = infos["odom_obs_history_Legolas"].to(env.device)
            odom_obs_history_baseline = infos["odom_obs_history_baseline"].to(env.device)
            yaw_history = infos["yaw_history"].to(env.device)
            pos_history = infos["pos_history"].to(env.device)
            abs_yaw_history = infos["abs_yaw_history"].to(env.device) # env_nums, stack_nums(yaw_i)
            start_mask = infos["start_mask"].to(env.device)
            odom = infos["odom"].to(env.device)
            
            pos_input = torch.stack(
                (
                    torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 2, 0].unsqueeze(1)) + torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 2, 1].unsqueeze(1)),
                    -torch.sin(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 0] - pred_pos_history[:, 2, 0].unsqueeze(1)) + torch.cos(abs_yaw_history[:, 1].unsqueeze(1)) * (pred_pos_history[:, :, 1] - pred_pos_history[:, 2, 1].unsqueeze(1))
                ),
                dim=-1
            )[:, 3:] # x_i - x_{i-index}
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
            pred_pos_history[done,:,0:2] = odom_pred_wys_pos[done,0:2].unsqueeze(1)
            pred_pos_history[:, -1, :] = odom_pred_wys_pos
            
        use_pred_pos = True
        odom_loss_list_wys = list()
        for j in range(20):
            if use_pred_pos==False:
                odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"], buf["yaw_history"][:, :, 2:], buf["pos_history"][..., :-1, :])
            else:
                pos_input = torch.stack(
                    (
                        torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1)),
                        -torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1))
                    ),
                    dim=-1
                )[:, :, 1:-2]
                yaw_input = buf["abs_yaw_history"][:, :, :] - buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)
                yaw_input = yaw_input[:, :, 1:-1]
                if not USE_POS_SEQ:
                    pos_input = torch.zeros_like(pos_input)
                odom_pred_wys = odom_model_wys(
                    buf["odom_obs_history_wys"][:, :, :-1], 
                    yaw_input, 
                    pos_input
                )
            if DELTA_TIME == 0.02:
                index = -2
            else:
                index = -52
            odom_loss_wys = F.mse_loss(odom_pred_wys, buf["pos_history"][..., -1 - 3, :] - buf["pos_history"][..., index - 3, :]) #  在x_i-51坐标系下的坐标
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
            name = "model_wys_" + str(i + 1) + NAME + ".pt"
            scripted_model.save(os.path.join(dir, f"model_wys_{i + 1}.pt"))
            odom_model_wys.to(env.device)
            odom_model_wys.train()