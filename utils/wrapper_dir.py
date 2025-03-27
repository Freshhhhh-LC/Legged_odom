import csv
import torch

class OdomStackingDataEnvFromFile:
    def __init__(self, csv_file_path, obs_stacking, device):
        self.obs_stacking = obs_stacking
        self.data_file = open(csv_file_path, mode="r", newline="", encoding="utf-8")
        self.data_reader = csv.reader(self.data_file)
        next(self.data_reader) # next() to skip the header        
        self.row = next(self.data_reader) # next() to get the first row # row:
        # 0: "time"
        # 1: "yaw"
        # 2:5 "projected_gravity_x", "projected_gravity_y", "projected_gravity_z"
        # 5:8 "ang_vel_x", "ang_vel_y", "ang_vel_z"
        # 8:11 "lin_acc_x", "lin_acc_y", "lin_acc_z"
        # 11:34 *["q_" + str(i) for i in range(23)]
        # 34:57 *["dq_" + str(i) for i in range(23)]
        # 57:59 "mocap_time", "mocap_timestamp"
        # 59:61 "robot_x", "robot_y"
        # 61:62 "robot_yaw"
        # 62:64 "ball_x", "ball_y"
        
        self.next_row = next(self.data_reader) # next() to get the second row
        self.odom_obs_history_wys = torch.zeros(self.obs_stacking, 32, device=device)
        # self.odom_obs_history_baseline = torch.zeros(self.obs_stacking, 45, device=device)
        self.yaw_history = torch.zeros(self.obs_stacking, device=device)
        self.pos_history = torch.zeros(self.obs_stacking + 1, 2, device=device)
        self.device = device
        
        self.start_yaw_of_segment = torch.tensor([float(self.row[1])], device=device)
        self.start_pos_of_segment = torch.tensor([float(self.row[59]), float(self.row[60])], device=device)
        self.num_rows = 0
        for row in self.data_reader:
            self.num_rows += 1
        
        self.q0 = torch.zeros(13, device=self.device)
        # ankle
        # self.q0[0, [1, 7]] = -0.2
        # self.q0[0, [4, 10]] = 0.4
        # self.q0[0, [5, 11]] = -0.25
        self.q0[[1, 7]] = -0.2
        self.q0[[4, 10]] = 0.4
        self.q0[[5, 11]] = -0.25


        
    def reset(self):
        self.data_file.seek(0)
        self.data_reader = csv.reader(self.data_file)
        next(self.data_reader) # next() to skip the header
        self.row = next(self.data_reader) # next() to get the first row
        self.start_yaw_of_segment = torch.tensor([float(self.row[1])], device=self.device)
        self.start_pos_of_segment = torch.tensor([float(self.row[59]), float(self.row[60])], device=self.device)
        
        project_gravity = torch.tensor([float(self.row[2]), float(self.row[3]), float(self.row[4])], device=self.device)
        ang_vel = torch.tensor([float(self.row[5]), float(self.row[6]), float(self.row[7])], device=self.device)
        q = torch.tensor([float(self.row[i]) for i in range(21, 34)], device=self.device)
        dq = torch.tensor([float(self.row[i]) for i in range(44, 57)], device=self.device)
        dq = dq * 0.1 
        self.odom_obs_history_wys[:] = torch.cat((project_gravity, ang_vel, q, dq)).unsqueeze(0)
        
        yaw = torch.tensor([float(self.row[1]) - self.start_yaw_of_segment[0]], device=self.device)
        self.yaw_history[:] = yaw.unsqueeze(0)
        
        pos_groundtruth = torch.tensor([float(self.row[59]) - self.start_pos_of_segment[0], float(self.row[60]) - self.start_pos_of_segment[1]], device=self.device)

        pos_groundtruth = torch.stack(
            (
                torch.cos(self.start_yaw_of_segment[0]) * (pos_groundtruth[0]) + torch.sin(self.start_yaw_of_segment[0]) * (pos_groundtruth[1]),
                -torch.sin(self.start_yaw_of_segment[0]) * (pos_groundtruth[0]) + torch.cos(self.start_yaw_of_segment[0]) * (pos_groundtruth[1]),
            ),
            dim=-1,
        )
        
        infos = {
            "odom_obs_history_wys": self.odom_obs_history_wys,
            "yaw_history": torch.zeros_like(self.yaw_history),
            "pos_history": torch.zeros_like(self.pos_history),
            "pos_groundtruth": pos_groundtruth,
            "abs_yaw_history": torch.zeros_like(self.yaw_history),
        }
        return infos
    
    def step(self):
        self.row = next(self.data_reader)
        if self.row is None:
            return None, True
        project_gravity = torch.tensor([float(self.row[2]), float(self.row[3]), float(self.row[4])], device=self.device)
        ang_vel = torch.tensor([float(self.row[5]), float(self.row[6]), float(self.row[7])], device=self.device)
        q = torch.tensor([float(self.row[i]) for i in range(21, 34)], device=self.device) - self.q0
        dq = torch.tensor([float(self.row[i]) for i in range(44, 57)], device=self.device)
        dq = dq * 0.1
        
        self.odom_obs_history_wys = torch.roll(self.odom_obs_history_wys, -1, dims=0)
        # print("cat", torch.cat((project_gravity, ang_vel, q, dq)).unsqueeze(0).shape)
        # print("-1", self.odom_obs_history_wys[-1].shape)
        self.odom_obs_history_wys[-1] = torch.cat((project_gravity, ang_vel, q, dq)).unsqueeze(0)
        
        yaw = torch.tensor([float(self.row[1]) - self.start_yaw_of_segment[0]], device=self.device)
        self.yaw_history = torch.roll(self.yaw_history, -1, dims=0)
        self.yaw_history[-1] = yaw
        
        pos_groundtruth = torch.tensor([float(self.row[59]) - self.start_pos_of_segment[0], float(self.row[60]) - self.start_pos_of_segment[1]], device=self.device)

        pos_groundtruth = torch.stack(
            (
                torch.cos(self.start_yaw_of_segment[0]) * (pos_groundtruth[0]) + torch.sin(self.start_yaw_of_segment[0]) * (pos_groundtruth[1]),
                -torch.sin(self.start_yaw_of_segment[0]) * (pos_groundtruth[0]) + torch.cos(self.start_yaw_of_segment[0]) * (pos_groundtruth[1]),
            ),
            dim=-1,
        ) # 转为机器人初始坐标系下的坐标

        self.pos_history = torch.roll(self.pos_history, -1, dims=0)
        self.pos_history[-1] = pos_groundtruth # 机器人初始坐标系下的坐标

        infos = {
            "odom_obs_history_wys": self.odom_obs_history_wys,
            "yaw_history": self.yaw_history - self.yaw_history[0].unsqueeze(0),
            "pos_history": torch.stack(
                (
                    torch.cos(self.yaw_history[0].unsqueeze(0)) * (self.pos_history[:, 0] - self.pos_history[0, 0].unsqueeze(0))
                    + torch.sin(self.yaw_history[0].unsqueeze(0)) * (self.pos_history[:, 1] - self.pos_history[0, 1].unsqueeze(0)),
                    -torch.sin(self.yaw_history[0].unsqueeze(0)) * (self.pos_history[:, 0] - self.pos_history[0, 0].unsqueeze(0))
                    + torch.cos(self.yaw_history[0].unsqueeze(0)) * (self.pos_history[:, 1] - self.pos_history[0, 1].unsqueeze(0)),
                ),
                dim=-1,
            ),
            "pos_groundtruth": pos_groundtruth,
            "abs_yaw_history": self.yaw_history,
        }
        
        return infos, False