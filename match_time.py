for i in range(13, 14):
    import os
    import csv
    import numpy as np

    DATA_NAME = str(i)
    TIME_THRESHOLD = 0.05
    TIME_OF_SEGMENT = 4.5

    dir = os.path.join("data", DATA_NAME)
    os.makedirs(dir, exist_ok=True) # 如果目录不存在则创建目录
    # time,timestamp,robot_x,robot_y,robot_yaw,ball_x,ball_y
    mocap_file = open(os.path.join("data", DATA_NAME, "mocap.csv"), mode="r", newline="", encoding="utf-8")
    # time(1),yaw(1),projected_gravity(3),ang_vel(3),lin_acc(3),q(23),dq(23)
    booster_file = open(os.path.join("data", DATA_NAME, "booster.csv"), mode="r", newline="", encoding="utf-8")
    mocap_reader = csv.reader(mocap_file)
    booster_reader = csv.reader(booster_file)

    next(mocap_reader) # next() to skip the header
    mocap_row = next(mocap_reader) # next() to get the first row
    next_mocap_row = next(mocap_reader) # next() to get the second row
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

    # 保存超过9秒的连续数据段
    output_dir = os.path.join(dir, "segments")
    os.makedirs(output_dir, exist_ok=True)

    # 重置文件读取器
    booster_file.seek(0)
    booster_reader = csv.reader(booster_file)
    next(booster_reader)  # 跳过表头

    # 读取 mocap 数据到内存
    mocap_file.seek(0)
    mocap_reader = csv.reader(mocap_file)
    next(mocap_reader)  # 跳过表头
    mocap_data = list(mocap_reader)

    current_segment = []
    segment_index = 0
    time = None

    # 修改 booster 段文件的写入逻辑
    for i, row in enumerate(booster_reader):
        if i % 10 != 0:  # 每隔10行读取一次
            continue
        current_time = float(row[0])
        # 检查时间间隔是否超过 TIME_THRESHOLD, 如果超过则保存当前段
        if time is not None and current_time - time > TIME_THRESHOLD:
            # 检查当前段是否超过3秒
            if current_segment and float(current_segment[-1][0]) - float(current_segment[0][0]) > TIME_OF_SEGMENT:
                segment_duration = round(float(current_segment[-1][0]) - float(current_segment[0][0]), 2)
                segment_file = os.path.join(output_dir, f"booster_seg_{segment_duration}s.csv")
                with open(segment_file, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 更新表头
                    writer.writerow([
                        "time", "yaw",
                        "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z",
                        "lin_acc_x", "lin_acc_y", "lin_acc_z",
                        *["q_" + str(i) for i in range(23)],
                        *["dq_" + str(i) for i in range(23)],
                        "mocap_time", "mocap_timestamp",
                        "robot_x", "robot_y", "robot_yaw", "ball_x", "ball_y"
                    ])
                    
                    last_matched_row = None
                    for booster_row in current_segment:
                        booster_time = float(booster_row[0])
                        matched_row = None
                        
                        min_diff = float('inf')  # 初始化最小差值为无穷大
                        # 在 mocap 数据中寻找时间差最短的行
                        for mocap_row in mocap_data:
                            mocap_time = float(mocap_row[0])
                            if abs(mocap_time - booster_time) < min_diff:
                                min_diff = abs(mocap_time - booster_time)
                                matched_row = mocap_row
                            
                            
                        
                        # 如果没有匹配的行，重复上一个匹配的行
                        if matched_row is None:
                            matched_row = last_matched_row
                        else:
                            last_matched_row = matched_row
                        
                        # 添加匹配的 mocap 数据到当前行
                        if matched_row:
                            writer.writerow(booster_row + matched_row)
                        else:
                            os.remove(segment_file)
                            print(f"Segment {segment_index} has no matching mocap data. Deleting file.")
                            break
                segment_index += 1
            current_segment = []
        current_segment.append(row)
        time = current_time

    # 检查最后一段
    if current_segment and float(current_segment[-1][0]) - float(current_segment[0][0]) > TIME_OF_SEGMENT:
        segment_duration = round(float(current_segment[-1][0]) - float(current_segment[0][0]), 2)
        segment_file = os.path.join(output_dir, f"booster_seg_{segment_duration}s.csv")
        with open(segment_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "yaw",
                "projected_gravity_x", "projected_gravity_y", "projected_gravity_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
                "lin_acc_x", "lin_acc_y", "lin_acc_z",
                *["q_" + str(i) for i in range(23)],
                *["dq_" + str(i) for i in range(23)],
                "mocap_time", "mocap_timestamp",
                "robot_x", "robot_y", "robot_yaw", "ball_x", "ball_y"
            ])
            
            last_matched_row = None
            for booster_row in current_segment:
                booster_time = float(booster_row[0])
                matched_row = None
                
                
                min_diff = float('inf')  # 初始化最小差值为无穷大
                # 在 mocap 数据中寻找时间差最短的行
                for mocap_row in mocap_data:
                    mocap_time = float(mocap_row[0])
                    if abs(mocap_time - booster_time) < min_diff:
                        min_diff = abs(mocap_time - booster_time)
                        matched_row = mocap_row
                    
                # 如果没有匹配的行，重复上一个匹配的行
                if matched_row is None:
                    matched_row = last_matched_row
                else:
                    last_matched_row = matched_row
                
                # 添加匹配的 mocap 数据到当前行
                if matched_row:
                    writer.writerow(booster_row + matched_row)
                else:
                    os.remove(segment_file)  # 如果没有匹配数据，删除文件
                    print(f"Segment {segment_index} has no matching mocap data. Deleting file.")
                    break

    # 检查 segments 文件夹中每个文件的时间连续性
    segment_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    for segment_file in segment_files:
        segment_path = os.path.join(output_dir, segment_file)
        print(f"Checking time continuity for {segment_file}...")
        
        with open(segment_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            
            time = None
            discontinuities = 0
            start_time = None
            end_time = None
            for row in reader:
                current_time = float(row[0])
                if start_time is None:
                    start_time = current_time
                end_time = current_time
                if time is not None and current_time - time > TIME_THRESHOLD:
                    print(f"Discontinuity found in {segment_file} at time {time} and {current_time}")
                    discontinuities += 1
                time = current_time
            
            if discontinuities == 0:
                print(f"No discontinuities found in {segment_file}.")
            else:
                print(f"Total discontinuities in {segment_file}: {discontinuities}")
            
            # 输出文件总时长
            if start_time is not None and end_time is not None:
                total_duration = round(end_time - start_time, 2)
                print(f"Total duration of {segment_file}: {total_duration} seconds")
