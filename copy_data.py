import os
import shutil

def copy_files():
    source_folder = "/home/luochangsheng/odom/Legged_odom/data_mixed/segment_length=450"
    files = [f for f in os.listdir(source_folder) if f.startswith("segment_") and f.endswith(".csv")]

    for i, file in enumerate(files):
        file_path = os.path.join(source_folder, file)
        file_name, file_ext = os.path.splitext(file)
        # file_index为segment_后面的数字
        file_index = file_name.split("_")[1]
        file_index = int(file_index)  # Convert to integer for comparison
        if file_index >= 372:
            continue
        
        for i in range(1, 12):  # Create 11 copies
            new_file_name = f"{file_name}_copy_{i}{file_ext}"
            new_file_path = os.path.join(source_folder, new_file_name)
            shutil.copy(file_path, new_file_path)
    # 输出source_folder里的文件总数
    total_files = len(os.listdir(source_folder))
    print(f"Total files in {source_folder}: {total_files}")

if __name__ == "__main__":
    copy_files()
