import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper_file import OdomStackingDataEnvFromFile
from utils.model import OdomEstimator_wys
from utils.dataset import Dataset

if __name__ == "__main__":
    DELTA_TIME = 0.02
    USE_ACC = True
    USE_POS_SEQ = True
    USE_ACTIONS = True
    NAME = "_file"
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
    
    # data_dir = "/home/luochangsheng/odom/Legged_odom/data_mixed/segment_length=450"
    data_dir = "/home/luochangsheng/odom/Legged_odom/data_sim/segments"
    csv_file_paths = []
    for data_file in os.listdir(data_dir):
        if data_file.endswith(".csv"):
            csv_file_paths.append(os.path.join(data_dir, data_file))
    csv_file_paths.sort()
    num_envs = len(csv_file_paths)
    
    env = OdomStackingDataEnvFromFile(csv_file_paths, obs_stacking=50, device="cuda:3")
    num_steps = env.num_rows[0]
    
    odom_model_wys = OdomEstimator_wys(num_obs_wys + 4, env.obs_stacking).to(env.device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=3e-4)

    state_dict = torch.load("models/T1_run.pth", weights_only=True)
    recorder = SummaryWriter(dir)

    buf = Dataset(24, env.num_envs)
    buf.AddBuffer("odom_obs_history_wys", (env.obs_stacking + 1, num_obs_wys), device=env.device)
    buf.AddBuffer("yaw_history", (env.obs_stacking + 2,), device=env.device)
    buf.AddBuffer("pos_history", (env.obs_stacking + 2, 2), device=env.device)
    buf.AddBuffer("pred_pos_history", (env.obs_stacking + 3, 2), device=env.device)
    buf.AddBuffer("abs_yaw_history", (env.obs_stacking + 2,), device=env.device)

    for epoch in range(100):
        env = OdomStackingDataEnvFromFile(csv_file_paths, obs_stacking=50, device="cuda:3")
        infos = env.reset()
        if USE_ACC and USE_ACTIONS:
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
        # [batch_size, num_envs, num_stack, 2]
        pred_pos_history = torch.zeros(env.num_envs, env.obs_stacking + 3, 2, device=env.device)
        loss_list = list()
        
        for i in range(num_steps // 24):
            for j in range(24):
                if i * 24 + j >= num_steps:
                    break
                buf.Record("odom_obs_history_wys", j, odom_obs_history_wys)
                buf.Record("yaw_history", j, yaw_history)
                buf.Record("pos_history", j, pos_history)
                buf.Record("abs_yaw_history", j, abs_yaw_history)
                buf.Record("pred_pos_history", j, pred_pos_history)
                    
                infos, done = env.step()
                if done:
                    break
                if USE_ACC and USE_ACTIONS:
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
                    odom_pred_wys = odom_model_wys(odom_obs_history_wys[:, 1:], yaw_input, pos_input)
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
                    odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"][:, :, :-1], yaw_input, pos_input)
                if DELTA_TIME == 0.02:
                    index = -2
                else:
                    index = -52
                odom_loss_wys = F.mse_loss(odom_pred_wys, buf["pos_history"][..., -1, :] - buf["pos_history"][..., index, :]) #  在x_i-51坐标系下的坐标
                odom_loss_wys = odom_loss_wys.clip(min=-10.0, max=10.0)
                optimizer_wys.zero_grad()
                odom_loss_wys.backward(retain_graph=True)
                optimizer_wys.step()
                odom_loss_list_wys.append(odom_loss_wys.item())
            odom_loss_mean_wys = sum(odom_loss_list_wys) / len(odom_loss_list_wys)
            recorder.add_scalar("odom_loss_wys", odom_loss_mean_wys, i)
            loss_list.append(odom_loss_mean_wys) 
            print(f"epoch: {epoch}, iter: {i + 1}, \todom_loss_wys: {odom_loss_mean_wys}")  
        
        loss_mean = sum(loss_list) / len(loss_list)
        odom_model_wys.eval()
        odom_model_wys.cpu()
        scripted_model = torch.jit.script(odom_model_wys)
        name = "model_wys_" + str(epoch) + NAME + ".pt"
        scripted_model.save(os.path.join(dir, name))
        odom_model_wys.to(env.device)
        odom_model_wys.train()
        print(f"epoch: {epoch}, iter: {i + 1}, \todom_loss_wys: {loss_mean}")