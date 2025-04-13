import csv

def check_non_float_elements(file_path):
    non_float_elements = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 跳过表头
        rows = []
        for row_index, row in enumerate(reader, start=1):
            new_row = []
            for col_index, element in enumerate(row):
                if element.strip() == "":  # 检查空缺元素
                    element = "0.0"  # 用 0.0 替代
                try:
                    float(element)
                    new_row.append(element)
                except ValueError:
                    non_float_elements.append((row_index, col_index, element))
                    new_row.append(element)
            rows.append(new_row)
    return non_float_elements, header, rows

def write_fixed_file(file_path, header, rows):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 写入表头
        writer.writerows(rows)  # 写入修复后的数据

if __name__ == "__main__":
    file_path = "/home/luochangsheng/odom/Legged_odom/data/14/segments/booster_seg_447.41s_edited.csv"
    result, header, rows = check_non_float_elements(file_path)
    write_fixed_file(file_path, header, rows)
    if result:
        print("以下元素不能转换为float：")
        for row_index, col_index, element in result:
            print(f"行 {row_index + 1}, 列 {col_index + 1}: {element}")
    else:
        print("所有元素都可以转换为float，且空缺元素已替换为0.0。")
