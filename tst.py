import os
import shutil

model_moves = [
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-04-08-00-23-53_0.02s_actions/model_wys_2000.pt", "sim_no_acc.pt"),
    ("/home/luochangsheng/odom/logs/2025-05-26-19-24-02_file_0.02s_acc_actions/model_wys_81_file_0.02s_acc_actions.pt", "real_with_acc.pt"),
    ("/home/luochangsheng/odom/logs/2025-05-24-20-46-03_file_0.02s_actions/model_wys_1_file_0.02s_actions.pt", "real_no_acc.pt"),
    ("/home/luochangsheng/odom/logs/2025-05-23-19-12-51_file_0.02s_actions/model_wys_0_file_0.02s_actions.pt", "sim_no_acc_enhanced_by_no_acc_real.pt"),
    ("/home/luochangsheng/odom/logs/2025-05-23-21-00-55_file_0.02s_acc_actions/model_wys_0_file_0.02s_acc_actions.pt", "sim_with_acc_enhanced_id0.pt"),
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-05-23-22-02-29_file_0.02s_actions/model_wys_4_file_0.02s_actions.pt", "mixed_no_acc_1to1.pt"),
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-05-23-22-02-19_file_0.02s_acc_actions/model_wys_6_file_0.02s_acc_actions.pt", "mixed_with_acc_1to1.pt"),
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-05-23-22-14-56_file_0.02s_actions/model_wys_4_file_0.02s_actions.pt", "mixed_no_acc_10to1.pt"),
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-05-23-22-15-04_file_0.02s_acc_actions/model_wys_32_file_0.02s_acc_actions.pt", "mixed_with_acc_10to1.pt"),
    ("/home/luochangsheng/odom/logs/2025-05-24-21-42-50_file_0.02s_acc_actions/model_wys_2_file_0.02s_acc_actions.pt", "sim_with_acc_enhanced_pre3.pt"),
    ("/home/luochangsheng/odom/Legged_odom/logs/2025-05-24-19-52-41_0.02s_actions/model_wys_2000.pt", "ordinary_pre3.pt"),
]

target_dir = "/home/luochangsheng/odom/Legged_odom/collected_models"
os.makedirs(target_dir, exist_ok=True)

for src_path, new_name in model_moves:
    dst_path = os.path.join(target_dir, new_name)
    print(f"移动 {src_path} -> {dst_path}")
    shutil.copy2(src_path, dst_path)