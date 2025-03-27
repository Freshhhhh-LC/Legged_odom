import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据生成与处理函数
def generate_sin_data(seq_length, num_sequences=1, step=0.1, variable_length=False):
    data = []
    for i in range(num_sequences):
        offset = random.uniform(0, 2 * math.pi)  # 每个序列添加一个随机偏移
        length = seq_length
        if variable_length:
            length = random.randint(seq_length - 10, seq_length)  # 随机生成序列长度
        sequence = [math.sin(offset + j * step) for j in range(length)]
        data.append(sequence)
    return data

def prepare_data(data, window_size):
    x, y = [], []
    for sequence in data:
        seq_x, seq_y = [], []
        for i in range(len(sequence) - window_size):
            seq_x.append(sequence[i:i + window_size])
            seq_y.append(sequence[i + window_size])
        seq_x = torch.tensor(seq_x, dtype=torch.float32).to(device)
        seq_y = torch.tensor(seq_y, dtype=torch.float32).to(device)
        x.append(seq_x)
        y.append(seq_y)
    # print(len(x), len(x[0]), len(x[1]), len(x[2]), len(x[3]))
    return x, y

# 模型定义
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # 添加特征维度
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output.squeeze(-1)

def train(model, dataloader, epochs=200, initial_teacher_prob=1.0, min_teacher_prob=0.1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        p_teacher = max(min_teacher_prob, initial_teacher_prob * (1 - epoch / epochs))
        for batch in dataloader:
            x_train, y_train = batch # x_train: [batch_size, seq_length, window_size], y_train: [batch_size, seq_length]
            for i in range(x_train.size(1)):
                x = x_train[:, i]  # x: [batch_size, window_size]
                predictions = []
                ground_truth = []
                
                for t in range(mini_step):
                    if i + t >= y_train.size(1):
                        break
                    y = y_train[:, i + t]  # y: [batch_size]
                    output = model(x) # output: [batch_size, window_size]
                    pred = output[:, -1] # pred: [batch_size]
                    # 使用 Scheduled Sampling
                    if random.random() < p_teacher:
                        next_input = y.unsqueeze(1)  # next_input: [batch_size, 1]
                    else:
                        next_input = pred.unsqueeze(1)  # next_input: [batch_size, 1]
                    predictions.append(pred)
                    x = torch.cat((x[:, 1:], next_input), dim=1) # roll forward
                    ground_truth.append(y)
                predictions = torch.stack(predictions, dim=1)
                ground_truth = torch.stack(ground_truth, dim=1) # [batch_size, mini_step]
                mask = torch.ones_like(ground_truth) # mask to filter out padding
                mask[ground_truth == 0] = 0.
                if mask.sum() == 0:
                    continue
                loss = (criterion(predictions, ground_truth) * mask).sum() / mask.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}, Teacher Prob: {p_teacher:.4f}")

def predict(model, initial_seq, steps=20):
    model.eval()
    x = torch.tensor(initial_seq[-window_size:], dtype=torch.float32).unsqueeze(0).to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            output = model(x)
            pred = output[:, -1].item()
            predictions.append(pred)
            x = torch.cat((x[:, 1:], torch.tensor([[pred]], device=device)), dim=1)
    return predictions

# 主逻辑
if __name__ == "__main__":
    # 生成多个序列数据
    seq_length = 100
    num_sequences = 1000
    data = generate_sin_data(seq_length, num_sequences, variable_length = False)

    # 准备批量数据
    window_size = 10
    mini_step = 10
    x_train, y_train = prepare_data(data, window_size)
    # 过滤掉长度为 0 的张量
    x_train_filtered = [x for x in x_train if x.shape[0] > 0]
    y_train_filtered = [y for y in y_train if y.shape[0] > 0]
    x_train = x_train_filtered
    y_train = y_train_filtered
    x_train = pad_sequence(x_train, batch_first=True)
    y_train = pad_sequence(y_train, batch_first=True)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 初始化模型
    model = SimpleLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    train(model, dataloader, epochs=200)

    # 生成测试集
    seq = generate_sin_data(seq_length, 1)[0]
    test_seq = seq[:window_size]
    predictions = predict(model, test_seq, steps=len(seq) - window_size)

    # 画图
    plt.plot(seq, label="Ground Truth")
    plt.plot(range(window_size, window_size + len(predictions)), predictions, label="Predictions")
    plt.legend()
    plt.savefig("sin.png")
