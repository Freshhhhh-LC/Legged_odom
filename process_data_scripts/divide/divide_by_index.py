import os
import pandas as pd

def divide_csv_by_intervals(file_path, intervals):
    # 确保intervals文件夹存在
    parent_dir = "/home/luochangsheng/odom/Legged_odom/data_mixed"
    segments_dir = os.path.join(parent_dir, "interval_segments")
    os.makedirs(segments_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 获取已有的segment_文件数量，确定索引起始值
    existing_files = [f for f in os.listdir(segments_dir) if f.startswith("segment_") and f.endswith(".csv")]
    start_index = len(existing_files)

    # 根据区间分割CSV文件
    for start, end in intervals:
        if start >= len(df):
            break
        segment = df.iloc[start:end]
        name = f"segment_{start_index}.csv"
        start_index += 1
        # 保存分割后的CSV文件
        segment_file = os.path.join(segments_dir, name)
        segment.to_csv(segment_file, index=False)


if __name__ == "__main__":
    BASE_DIR = "/home/luochangsheng/odom/Legged_odom"
    DATA_INDEX = ["data_sim"]
    intervals = [(0, 3000), (3000, 5700), (5700, 6000), (6000, 7500), 
                 (7500, 9800), (9800, 12934), (12934, 15138), 
                 (15138, 16749), (16749, 20822)]
    for index in DATA_INDEX:
        dir = os.path.join(BASE_DIR, index, "segments")
        for data_file in os.listdir(dir):
            if data_file.endswith(".csv"):
                data_file_path = os.path.join(dir, data_file)
                print(f"Processing file: {data_file_path}")
                divide_csv_by_intervals(data_file_path, intervals)
