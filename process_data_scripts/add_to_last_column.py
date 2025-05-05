import csv
import os

def add_to_last_column(input_file, output_file, number_to_add):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader) 
        rows = [row for row in reader]

    for row in rows:
        if row:  # 确保行不为空
            row[-1] = str(float(row[-1]) + number_to_add)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # 写入表头
        writer.writerows(rows)

if __name__ == "__main__":
    input_csv = "/home/luochangsheng/odom/Legged_odom/data/15/1744534640034/rl_locomotion_run.csv"  # 输入文件路径
    output_csv = "/home/luochangsheng/odom/Legged_odom/data/15/1744534640034/rl_locomotion_run_edited.csv"  # 输出文件路径

    # 获取输入文件的父文件夹名称并计算要加的数
    parent_folder_name = os.path.basename(os.path.dirname(input_csv))
    number = float(parent_folder_name) / 1000

    add_to_last_column(input_csv, output_csv, number)
