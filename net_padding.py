import torch

# 加载TorchScript模型
# model_path = '/home/luochangsheng/odom/Enhanced_odom/models/best_odom_model.pt'
model_path = '/home/luochangsheng/odom/Legged_odom/logs/2025-05-27-00-03-03_LSTM_0.02s_actions/model_wys_2000.pt'
model = torch.jit.load(model_path, map_location='cpu')

# 先打印模型结构（每一层参数shape）
print("模型结构参数shape:")
for name, param in model.named_parameters():
    print(f"{name}: {tuple(param.shape)}")
    

# 修改 net.0.weight 的形状并保存为 .pth 文件
with torch.no_grad():
    state_dict = model.state_dict()
    # 获取第一层的名字
    first_weight_name = next((k for k in state_dict.keys() if k.endswith('weight')), None)
    if first_weight_name is None:
        raise RuntimeError("未找到任何权重参数")
    else:
        print(f"找到的第一个权重参数: {first_weight_name}")
    w = state_dict[first_weight_name]
    if w.shape[1] < 2500:
        print(f"{first_weight_name} 当前形状: {tuple(w.shape)}")
        print(f"{first_weight_name} 的类型: {w.dtype}")
        # 在w的47*k+31后插入3列0
        cols = w.shape[1]
        insert_points = [47 * k + 31 for k in range(50) if 47 * k + 31 < cols]
        w_splits = []
        last = 0
        for idx in insert_points:
            w_splits.append(w[:, last:idx+1])
            zeros = torch.zeros((w.shape[0], 3), dtype=w.dtype, device=w.device)
            w_splits.append(zeros)
            last = idx+1
        if last < cols:
            w_splits.append(w[:, last:])
        w_new = torch.cat(w_splits, dim=1)
        state_dict[first_weight_name] = w_new
        print(f"{first_weight_name} 已扩展到: {tuple(w_new.shape)}")
        # 打印为0的部分的索引
        zero_indices = torch.where(w_new == 0)
        print(f"扩展后的 {first_weight_name} 中为0的部分的索引:", zero_indices)
        # 保存扩展后的 state_dict
        # new_state_dict_path = '/home/luochangsheng/odom/Enhanced_odom/models/best_odom_model_padded.pth'
        new_state_dict_path = '/home/luochangsheng/odom/Legged_odom/models/best_LSTM_odom_model_padded.pth'
        torch.save(state_dict, new_state_dict_path)
        print(f"已保存扩展后的 state_dict 到: {new_state_dict_path}")
        # 打印新模型参数结构
        print("扩展后模型参数结构:")
        for k, v in state_dict.items():
            print(f"{k}: {tuple(v.shape)}")
    else:
        print(f"{first_weight_name} 已经是目标形状或更大，无需扩展。")

print("原始模型参数结构:")
for name, param in model.named_parameters():
    print(f"{name}: {tuple(param.shape)}")