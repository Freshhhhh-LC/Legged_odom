import csv
import os

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
    base_path = "/home/luochangsheng/odom/Legged_odom/data"
    for i in range(1, 8):  # 遍历 data/1 到 data/7
        segments_path = os.path.join(base_path, str(i), "segments")
        if not os.path.exists(segments_path):
            continue
        for file_name in os.listdir(segments_path):
            file_path = os.path.join(segments_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".csv"):
                difference_average = calculate_difference_average(file_path)
                print(f"文件 {file_path} 的第一列数据的相邻行差值平均值是: {difference_average}")
