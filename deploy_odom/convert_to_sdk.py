import torch

# 加载模型路径
model_path = "/home/luochangsheng/odom/Legged_odom/logs/2025-03-25-22-31-37/model_wys_800.pt"  # 替换为你的模型路径
output_path = "/home/luochangsheng/odom/Legged_odom/models/model_wys_cpu_withoutacc.pt"  # 转换后的模型保存路径

def convert_model_to_cpu(model_path, output_path):
    # 加载模型
    model = torch.jit.load(model_path, map_location="cpu")
    
    # 设置模型为 eval 模式
    model.eval()
    
    # 保存到 CPU 上的模型
    torch.jit.save(model, output_path)
    print(f"模型已成功转换为 eval 模式并保存到 CPU 上: {output_path}")

if __name__ == "__main__":
    convert_model_to_cpu(model_path, output_path)