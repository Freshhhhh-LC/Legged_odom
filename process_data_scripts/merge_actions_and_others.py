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
    last_matched_index = None  # 记录上一次匹配的 mocap 行索引

    search_window = 10  # 在±10行范围内查找

    line = 0
    for _, booster_row in booster_df.iterrows():
        booster_time = booster_row[0]  # 假设时间在第一列
        matched_row = None
        min_diff = float('inf')
        print(f"Searching for line {line} in booster data")
        line += 1

        # 只在last_matched_index附近查找
        if last_matched_index is not None:
            start = max(0, last_matched_index - search_window)
            end = min(len(mocap_df), last_matched_index + search_window + 1)
            search_iter = mocap_df.iloc[start:end].iterrows()
        else:
            search_iter = mocap_df.iterrows()

        for idx, mocap_row in search_iter:
            mocap_time = mocap_row[-1]
            time_diff = abs(mocap_time - booster_time)
            if time_diff < min_diff:
                min_diff = time_diff
                matched_row = mocap_row
                matched_index = idx

        # 如果没有匹配的行，重复上一个匹配的行
        if matched_row is None:
            matched_row = last_matched_row
        else:
            last_matched_row = matched_row
            last_matched_index = matched_index  # 更新索引

        # 添加匹配的 mocap 数据到 booster 行
        if matched_row is not None:
            merged_rows.append(list(booster_row) + list(matched_row))
    
    # 保存合并结果
    merged_df = pd.DataFrame(merged_rows, columns=list(booster_df.columns) + list(mocap_df.columns))
    merged_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # input_file_path = "/home/luochangsheng/odom/Legged_odom/data/16/motion_record_1747051061227/rl_locomotion_edited.csv"
    # output_dir = "/home/luochangsheng/odom/Legged_odom/data/16"
    # remove_same_rows(input_file_path, output_dir)
    
    booster_file_path = "/home/luochangsheng/odom/Legged_odom/data/16/segments/booster_seg_1291.92s.csv"
    actions_file_path = "/home/luochangsheng/odom/Legged_odom/data/16/rl_locomotion_edited.csv"
    output_file_path = "/home/luochangsheng/odom/Legged_odom/data/16/merged_output.csv"
    
    merge_files(booster_file_path, actions_file_path, output_file_path)
