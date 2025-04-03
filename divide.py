import os
import pandas as pd

def divide_csv(file_path, segment_length=450):
    # 确保segment_length=segment_length文件夹存在
    # base_dir = os.path.dirname(file_path)
    # parent_dir = os.path.dirname(base_dir)
    # parent_dir = os.path.dirname(parent_dir)
    parent_dir = "/home/luochangsheng/odom/Legged_odom/data_mixed"
    segments_dir = os.path.join(parent_dir, f"segment_length={segment_length}")
    os.makedirs(segments_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(file_path)
    total_rows = len(df)

    # 如果最后剩下的长度不足segment_length，舍弃
    total_rows = total_rows - (total_rows % segment_length)

    # 获取已有的segment_文件数量，确定索引起始值
    existing_files = [f for f in os.listdir(segments_dir) if f.startswith("segment_") and f.endswith(".csv")]
    start_index = len(existing_files)

    # 分割CSV文件
    for i in range(0, total_rows, segment_length):
        segment = df.iloc[i:i + segment_length]
        name = f"segment_{start_index}.csv"
        start_index += 1
        # 保存分割后的CSV文件
        segment_file = os.path.join(segments_dir, name)
        segment.to_csv(segment_file, index=False)


if __name__ == "__main__":
    # BASE_DIR = "/home/luochangsheng/odom/Legged_odom/data"
    # DATA_INDEX = ["9", "10", "11", "12"]
    BASE_DIR = "/home/luochangsheng/odom/Legged_odom"
    DATA_INDEX = ["data_sim"]
    for index in DATA_INDEX:
        dir = os.path.join(BASE_DIR, index, "segments")
        for data_file in os.listdir(dir):
            if data_file.endswith(".csv"):
                data_file_path = os.path.join(dir, data_file)
                print(f"Processing file: {data_file_path}")
                divide_csv(data_file_path, segment_length=1800)
