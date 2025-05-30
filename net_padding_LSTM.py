import torch

# 加载TorchScript模型
# model_path = '/home/luochangsheng/odom/Enhanced_odom/models/best_odom_model.pt'
model_path = '/home/luochangsheng/odom/Legged_odom/logs/2025-05-27-17-19-49_LSTM_0.02s_actions/model_wys_2000.pt'
model = torch.jit.load(model_path, map_location='cpu')

# 先打印模型结构（每一层参数shape）
print("模型结构参数shape:")
for name, param in model.named_parameters():
    print(f"{name}: {tuple(param.shape)}")

# 修改 lstm.weight_ih_l0 的形状并保存为 .pth 文件
with torch.no_grad():
    state_dict = model.state_dict()
    weight_name = 'lstm.weight_ih_l0'
    if weight_name not in state_dict:
        raise RuntimeError(f"未找到参数 {weight_name}")
    w = state_dict[weight_name]
    print(f"{weight_name} 当前形状: {tuple(w.shape)}")
    if w.shape == (512, 47):
        # 在每一行的第32列后插入3个0，扩展为(512, 50)
        left = w[:, :32]  # 不包含第32列
        right = w[:, 32:]
        zeros = torch.zeros((w.shape[0], 3), dtype=w.dtype, device=w.device)
        w_new = torch.cat([left, zeros, right], dim=1)
        assert w_new.shape == (512, 50), f"扩展后形状为{w_new.shape}，不是(512, 50)"
        state_dict[weight_name] = w_new
        print(f"{weight_name} 已扩展到: {tuple(w_new.shape)}")
        # 保存扩展后的 state_dict
        new_state_dict_path = '/home/luochangsheng/odom/Legged_odom/models/best_LSTM_odom_model_padded.pth'
        torch.save(state_dict, new_state_dict_path)
        print(f"已保存扩展后的 state_dict 到: {new_state_dict_path}")
    else:
        print(f"{weight_name} 已经是目标形状或更大，无需扩展。")