import os
import time
import csv
import torch
import torch.nn.functional as F

from utils.model import OdomEstimator_wys, OdomEstimator_Legolas, OdomEstimator_baseline
from utils.dataset import Dataset
from utils.wrapper_file import OdomStackingDataEnvFromFile

OBS_STACKING = 50
DATA_INDEXES = ["1", "2", "3", "4"]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    odom_model_wys = OdomEstimator_wys(32 + 4, OBS_STACKING).to(device)
    optimizer_wys = torch.optim.Adam(odom_model_wys.parameters(), lr=1e-6)
    latest_wys_model_path = "/home/lcs/RCL_Project/Legged_odom/logs/2025-03-24-10-47-15/model_wys_1700.pth"
    if latest_wys_model_path:
        checkpoint = torch.load(latest_wys_model_path)
        odom_model_wys.load_state_dict(checkpoint['model'])
        optimizer_wys.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from {latest_wys_model_path}")

    for data_index in DATA_INDEXES:
        dir = os.path.join("data", data_index, "segments")
        for data_file in os.listdir(dir):
            if data_file.startswith("booster_seg_"):
                data_file_path = os.path.join(dir, data_file)
                print(f"Processing file: {data_file_path}")

                env = OdomStackingDataEnvFromFile(data_file_path, OBS_STACKING, device)

                buf = Dataset(24, 1)
                buf.AddBuffer("odom_obs_history_wys", (OBS_STACKING, 32), device=device)
                buf.AddBuffer("yaw_history", (OBS_STACKING,), device=device)
                buf.AddBuffer("pos_history", (OBS_STACKING + 1, 2), device=device)
                buf.AddBuffer("pred_pos_history", (OBS_STACKING + 1, 2), device=device)
                buf.AddBuffer("abs_yaw_history", (OBS_STACKING,), device=device)
                
                infos = env.reset()
                odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
                yaw_history = infos["yaw_history"].to(device)
                pos_history = infos["pos_history"].to(device)
                pos_groundtruth = infos["pos_groundtruth"].to(device)
                abs_yaw_history = infos["abs_yaw_history"].to(device)
                pred_pos_history = torch.zeros(1, OBS_STACKING + 1, 2, device=device)

                num_steps = env.num_rows
                for i in range(num_steps // 24):
                    for j in range(24):
                        buf.Record("odom_obs_history_wys", j, odom_obs_history_wys.unsqueeze(0))
                        buf.Record("yaw_history", j, yaw_history.unsqueeze(0))
                        buf.Record("pos_history", j, pos_history.unsqueeze(0))
                        buf.Record("abs_yaw_history", j, abs_yaw_history.unsqueeze(0)) # yaw_i-1
                        infos, done = env.step()
                        if done:
                            break
                        odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
                        yaw_history = infos["yaw_history"].to(device) 
                        pos_history = infos["pos_history"].to(device) # x_i
                        abs_yaw_history = infos["abs_yaw_history"].to(device)

                        pos_input = torch.stack(
                            (
                                torch.cos(abs_yaw_history[0].unsqueeze(0)) * (pred_pos_history[0, :, 0] - pred_pos_history[0, 1, 0]) + torch.sin(abs_yaw_history[0].unsqueeze(0)) * (pred_pos_history[0, :, 1] - pred_pos_history[0, 1, 1]),
                                -torch.sin(abs_yaw_history[0].unsqueeze(0)) * (pred_pos_history[0, :, 0] - pred_pos_history[0, 1, 0]) + torch.cos(abs_yaw_history[0].unsqueeze(0)) * (pred_pos_history[0, :, 1] - pred_pos_history[0, 1, 1])
                            ),
                            dim=-1
                        )
                        with torch.no_grad():
                            # print("pos_input shape", pos_input.shape)
                            odom_pred_wys = odom_model_wys(odom_obs_history_wys, yaw_history, pos_input[:-1]) # x_i+1 - x_i

                        odom_pred_wys_pos = torch.stack(
                            (
                                torch.cos(abs_yaw_history[0]) * odom_pred_wys[0] - torch.sin(abs_yaw_history[0]) * odom_pred_wys[1] + pred_pos_history[0, -1, 0],
                                torch.sin(abs_yaw_history[0]) * odom_pred_wys[0] + torch.cos(abs_yaw_history[0]) * odom_pred_wys[1] + pred_pos_history[0, -1, 1]
                            ),
                            dim=-1
                        )
                        pred_pos_history = torch.roll(pred_pos_history, -1, dims=1)
                        pred_pos_history[:, -1, :] = odom_pred_wys_pos
                        buf.Record("pred_pos_history", j, pred_pos_history[0].unsqueeze(0)) # 用x_i-1预测的x_i
                    odom_loss_list_wys = list()
                    for k in range(20):
                        pos_input = torch.stack(
                            (
                                torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1)),
                                -torch.sin(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 0] - buf["pred_pos_history"][:, :, 0, 0].unsqueeze(-1)) + torch.cos(buf["abs_yaw_history"][:, :, 0].unsqueeze(-1)) * (buf["pred_pos_history"][:, :, :, 1] - buf["pred_pos_history"][:, :, 0, 1].unsqueeze(-1))
                            ),
                            dim=-1
                        )
                        # print("pos_input shape", pos_input.shape)
                        odom_pred_wys = odom_model_wys(buf["odom_obs_history_wys"], buf["yaw_history"], pos_input[:, :, :-1])
                        # print("buf shape", buf["pos_history"].shape)
                        odom_loss_wys = F.mse_loss(odom_pred_wys, (buf["pos_history"][:, :, -1] - buf["pos_history"][:, :, -2]))
                        optimizer_wys.zero_grad()
                        odom_loss_wys.backward()
                        optimizer_wys.step()
                        odom_loss_list_wys.append(odom_loss_wys.item())
                    odom_loss_mean_wys = sum(odom_loss_list_wys) / len(odom_loss_list_wys)

                    if i % 10 == 0:
                        print(f"Epoch {i}, WYS Loss: {odom_loss_mean_wys}")
                    if i % 10 == 0:
                        torch.save({
                            'model': odom_model_wys.state_dict(),
                            'optimizer': optimizer_wys.state_dict(),
                        }, f"logs/retrain_model_wys_{i}.pth")


