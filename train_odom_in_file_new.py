# 导入模块
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from utils.model import OdomEstimator_wys
from utils.wrapper_file import OdomStackingDataEnvFromFile

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 准备数据函数
def prepare_data(data_dir, seq_length, window_size, num_obs, device):
    num_sequences = sum(1 for data_file in os.listdir(data_dir) if data_file.endswith(".csv"))
    x_train = torch.zeros((num_sequences, seq_length - window_size, window_size, num_obs + 6), dtype=torch.float32).to(device)
    y_train = torch.zeros((num_sequences, seq_length - window_size, 2), dtype=torch.float32).to(device)
    
    for i, data_file in enumerate(os.listdir(data_dir)):
        if data_file.endswith(".csv"):
            file_path = os.path.join(data_dir, data_file)
            env = OdomStackingDataEnvFromFile(file_path, window_size, device=device)
            infos = env.reset()
            odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
            yaw_history = infos["yaw_history"].to(device)
            pos_history = infos["pos_history"].to(device)
            pos_groundtruth = infos["pos_groundtruth"].to(device)
            abs_yaw_history = infos["abs_yaw_history"].to(device)
            num_steps = env.num_rows
            done = False
            for j in range(window_size):
                infos, done = env.step()
                if done:
                    break
                odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
                yaw_history = infos["yaw_history"].to(device)
                pos_history = infos["pos_history"].to(device)
                abs_yaw_history = infos["abs_yaw_history"].to(device)
                pos_groundtruth = infos["pos_groundtruth"].to(device)
                x_train[i, j, :-1, 39:41] = x_train[i, j, 1:, 39:41]
                x_train[i, j, -1, 39] = pos_groundtruth[0]
                x_train[i, j, -1, 40] = pos_groundtruth[1]
                
            for j in range(num_steps - window_size):
                x_train[i, j, :, :35] = odom_obs_history_wys[1:]
                x_train[i, j, :, 35] = yaw_history[2:]
                x_train[i, j, :, 36] = abs_yaw_history[1:-1]
                x_train[i, j, :, 37:39] = pos_history[2:]
                x_train[i, j, :-1, 39:41] = x_train[i, j, 1:, 39:41]
                x_train[i, j, -1, 39] = pos_groundtruth[0]
                x_train[i, j, -1, 40] = pos_groundtruth[1]
                tmp_pos = pos_groundtruth.clone()
                infos, done = env.step()
                if done:
                    break
                odom_obs_history_wys = infos["odom_obs_history_wys"].to(device)
                yaw_history = infos["yaw_history"].to(device)
                pos_history = infos["pos_history"].to(device)
                abs_yaw_history = infos["abs_yaw_history"].to(device)
                pos_groundtruth = infos["pos_groundtruth"].to(device)
                y_train[i, j, 0] = torch.cos(abs_yaw_history[0]) * (pos_groundtruth[0] - tmp_pos[0]) + torch.sin(abs_yaw_history[0]) * (pos_groundtruth[1] - tmp_pos[1])
                y_train[i, j, 1] = -torch.sin(abs_yaw_history[0]) * (pos_groundtruth[0] - tmp_pos[0]) + torch.cos(abs_yaw_history[0]) * (pos_groundtruth[1] - tmp_pos[1])
    
    return x_train, y_train

# 训练函数
def train(model, dataloader, epochs=2000, initial_teacher_prob=1., min_teacher_prob=1.):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        p_teacher = max(min_teacher_prob, initial_teacher_prob * (1 - epoch / epochs))
        for batch in dataloader:
            x_train, y_train = batch # x_train: [batch_size, seq_length, window_size, feature_size], y_train: [batch_size, seq_length, 2]
            for i in range(x_train.size(1)):
                x = x_train[:, i]  # x: [batch_size, window_size, feature_size]
                predictions = []
                ground_truth = []
                pred_pos_hisitory = x_train[:, i, :, 39:41]  # [batch_size, window_size, 2]
                # print(pred_pos_hisitory.shape)
                # 第二维从50变成51，在前面加
                pred_pos_hisitory = torch.cat((torch.zeros(x.size(0), 1, 2).to(device), pred_pos_hisitory), dim=1)
                
                for t in range(mini_step):
                    if i + t + 1 >= y_train.size(1):
                        break
                    y = y_train[:, i + t]  # y: [batch_size, 2]
                    output = model(x[:, :, :35], x[:, :, 35], x[:, :, 37:39])
                    pred = output # pred: [batch_size, 2]
                    
                    odom_pos = torch.zeros(x.size(0), 2).to(device)
                    odom_pos[:, 0] = pred[:, 0] * torch.cos(x[:, 0, 36]) - pred[:, 1] * torch.sin(x[:, 0, 36]) + pred_pos_hisitory[:, -1, 0]
                    
                    odom_pos[:, 1] = pred[:, 0] * torch.sin(x[:, 0, 36]) + pred[:, 1] * torch.cos(x[:, 0, 36]) + pred_pos_hisitory[:, -1, 1]
                    
                    pred_pos_hisitory = torch.roll(pred_pos_hisitory, -1, dims=1)
                    pred_pos_hisitory[:, -1, :] = odom_pos
                    
                    # 使用 Scheduled Sampling
                    if random.random() < p_teacher:
                        next_input = x_train[:, i + t + 1]  # next_input: [batch_size, window_size, feature_size]
                    else:
                        next_input = x_train[:, i + t + 1].clone()
                        abs_yaw = x[:, 1, 36].unsqueeze(-1)  # yaw_i-50
                        pos_input_x = torch.cos(abs_yaw) * (pred_pos_hisitory[:, :, 0] - pred_pos_hisitory[:, 0, 0].unsqueeze(1)) + torch.sin(abs_yaw) * (pred_pos_hisitory[:, :, 1] - pred_pos_hisitory[:, 0, 1].unsqueeze(1))
                        pos_input_y = -torch.sin(abs_yaw) * (pred_pos_hisitory[:, :, 0] - pred_pos_hisitory[:, 0, 0].unsqueeze(1)) + torch.cos(abs_yaw) * (pred_pos_hisitory[:, :, 1] - pred_pos_hisitory[:, 0, 1].unsqueeze(1))
                        next_input[:, :, 37] = pos_input_x[:, 1:]
                        next_input[:, :, 38] = pos_input_y[:, 1:]
                    predictions.append(pred)
                    x = next_input.clone()
                    ground_truth.append(y)
                    
                predictions = torch.stack(predictions, dim=1)
                # print(predictions.shape)
                ground_truth = torch.stack(ground_truth, dim=1) # [batch_size, mini_step]
                loss = criterion(predictions, ground_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}, Teacher Prob: {p_teacher:.4f}")
            model.eval()
            model.cpu()
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(dir, f"model_epoch_{epoch}_file.pt"))
            model.to(device)
            model.train()

# 主逻辑
if __name__ == "__main__":
    dir = os.path.join("logs", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    
    seq_length = 225
    window_size = 50
    num_obs = 35
    mini_step = 10
    data_dir = f"/home/luochangsheng/odom/Legged_odom/data/segment_length={seq_length}"
    
    x_train, y_train = prepare_data(data_dir, seq_length, window_size, num_obs, device)
    dataset = TensorDataset(x_train, y_train)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = OdomEstimator_wys(35 + 4, 50).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train(model, dataloader, epochs=200)
