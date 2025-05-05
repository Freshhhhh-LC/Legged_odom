import pandas as pd
import os

def remove_same_rows(input_file_path, output_dir):
    # 读取 CSV 文件
    df = pd.read_csv(input_file_path)
    
    # 检查前 5 列是否连续相同，保留第一行
    df_filtered = df.loc[(df.iloc[:, :5] != df.iloc[:, :5].shift()).any(axis=1)]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构造输出文件路径
    output_file_path = os.path.join(output_dir, os.path.basename(input_file_path))
    
    # 将结果保存到新目录
    df_filtered.to_csv(output_file_path, index=False)

def merge_files(booster_file_path, mocap_file_path, output_file_path):
    # 读取 booster 和 mocap 数据
    booster_df = pd.read_csv(booster_file_path)
    mocap_df = pd.read_csv(mocap_file_path)
    
    # 初始化合并结果
    merged_rows = []
    last_matched_row = None

    # 遍历 booster 数据，寻找与 mocap 数据时间最接近的行
    for _, booster_row in booster_df.iterrows():
        booster_time = booster_row[0]  # 假设时间在第一列
        matched_row = None
        min_diff = float('inf')
        
        for _, mocap_row in mocap_df.iterrows():
            mocap_time = mocap_row[-1]  # 假设时间在第一列
            time_diff = abs(mocap_time - booster_time)
            if time_diff < min_diff:
                min_diff = time_diff
                matched_row = mocap_row
        
        # 如果没有匹配的行，重复上一个匹配的行
        if matched_row is None:
            matched_row = last_matched_row
        else:
            last_matched_row = matched_row
        
        # 添加匹配的 mocap 数据到 booster 行
        if matched_row is not None:
            merged_rows.append(list(booster_row) + list(matched_row))
    
    # 保存合并结果
    merged_df = pd.DataFrame(merged_rows, columns=list(booster_df.columns) + list(mocap_df.columns))
    merged_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    input_file_path = "/home/luochangsheng/odom/Legged_odom/data/15/1744534640034/rl_locomotion_run_edited.csv"
    output_dir = "/home/luochangsheng/odom/Legged_odom/data/15"
    remove_same_rows(input_file_path, output_dir)
    
    booster_file_path = "/home/luochangsheng/odom/Legged_odom/data/15/segments/booster_seg_30.35s.csv"
    actions_file_path = "/home/luochangsheng/odom/Legged_odom/data/15/rl_locomotion_run_edited.csv"
    output_file_path = "/home/luochangsheng/odom/Legged_odom/data/15/merged_output.csv"
    
    merge_files(booster_file_path, actions_file_path, output_file_path)
