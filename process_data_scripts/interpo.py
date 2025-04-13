import pandas as pd
import glob

def interpolate_robot_coordinates(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 检查并插值填充 robot_x 和 robot_y 列
    for col in ['robot_x', 'robot_y']:
        # 统计重复值数量
        duplicate_count = df[col].duplicated(keep='first').sum()
        print(f"文件 {file_path} 列 {col} 中的重复值数量: {duplicate_count}")
        
        # 将重复值（保留第一个）替换为 NaN
        df[col] = df[col].mask(df[col].duplicated(keep='first'))
        # 使用线性插值填充 NaN
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # 保存修改后的数据
    df.to_csv(file_path, index=False)

# 对所有以 /home/luochangsheng/odom/Legged_odom/data/segment_length= 开头的文件执行操作
file_pattern = '/home/luochangsheng/odom/Legged_odom/data/segment_length=2000/segment_*.csv'
for file_path in glob.glob(file_pattern):
    interpolate_robot_coordinates(file_path)
