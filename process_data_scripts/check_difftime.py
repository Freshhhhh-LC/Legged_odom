import csv
import os
import os
import csv
import numpy as np

def calculate_difference_average(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行
        values = [float(row[0]) for row in reader if row]  # 提取第一列数据并转换为浮点数
    
    if len(values) < 2:
        return 0  # 如果数据不足两行，返回0

    differences = [values[i+1] - values[i] for i in range(len(values) - 1)]  # 计算相邻行差值
    return sum(differences) / len(differences) if differences else 0

if __name__ == "__main__":
    file_path = "/home/luochangsheng/odom/Legged_odom/data/8/mocap.csv"
    if os.path.isfile(file_path) and file_path.endswith(".csv"):
        difference_average = calculate_difference_average(file_path)
        print(f"文件 {file_path} 的第一列数据的相邻行差值平均值是: {difference_average}")

    TIME_THRESHOLD = 2.

    
    booster_reader = csv.reader(open("/home/luochangsheng/odom/Legged_odom/data/8/mocap.csv", "r"))
    next(booster_reader)
    booster_row = next(booster_reader)
    next_booster_row = next(booster_reader)

    time = float(booster_row[0])
    start_time = time
    next_time = float(next(booster_reader)[0])
    i = 1
    discontinuities = 0
    total_time = 0
    discontinuity_durations = []
    continuous_segments = []
    segment_start_time = time

    # 检查时间是否连续
    while True:
        try:
            if next_time - time > TIME_THRESHOLD:
                print("Time is not continuous at line ", i, "with time ", time, "and next time ", next_time)
                discontinuities += 1
                discontinuity_durations.append(next_time - time)
                # 记录当前连续段的时间长度
                continuous_segments.append(time - segment_start_time)
                segment_start_time = next_time
            # 计算总时间
            total_time = next_time - start_time
            time = next_time
            next_time = float(next(booster_reader)[0])
            i += 1
        except StopIteration:
            # 记录最后一段连续数据的时间长度
            continuous_segments.append(time - segment_start_time)
            break

    # 输出统计信息
    print("Total time length:", total_time)
    print("Number of discontinuities:", discontinuities)
    # print("Discontinuity durations:", discontinuity_durations)
    # 输出两位小数
    discontinuity_durations = [round(d, 2) for d in discontinuity_durations]
    print("Discontinuity durations:", discontinuity_durations)
    continuous_segments = [round(s, 2) for s in continuous_segments]
    print("Continuous segment durations:", continuous_segments)