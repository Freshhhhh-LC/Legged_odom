import csv

def extract_and_save_tum_format(input_csv, output_txt):
    with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as txt_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            timestamp = row['time']
            x = row['robot_x']
            y = row['robot_y']
            txt_file.write(f"{timestamp} {x} {y} 0 0 0 0 1\n")  # TUM 格式

# 输入文件路径和输出文件路径
input_csv_path = "/home/luochangsheng/odom/Legged_odom/data/segment_length=1800/segment_10.csv"
output_txt_path = "/home/luochangsheng/odom/Legged_odom/gt.txt"

# 调用函数
extract_and_save_tum_format(input_csv_path, output_txt_path)
