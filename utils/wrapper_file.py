import csv
import torch

class OdomStackingDataEnvFromFile:
    def __init__(self, csv_file_paths, obs_stacking, device):
        self.num_envs = len(csv_file_paths)
        self.obs_stacking = obs_stacking
        self.data_files = [open(path, mode="r", newline="", encoding="utf-8") for path in csv_file_paths]
        self.data_readers = [csv.reader(file) for file in self.data_files]
        
        for reader in self.data_readers:
            next(reader)  # Skip headers
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
        # 64:75 ["actions_" + str(i) for i in range(11)]
        
        self.rows = [next(reader) for reader in self.data_readers]
        self.next_rows = [next(reader) for reader in self.data_readers]
        
        self.odom_obs_history_wys = torch.zeros(self.num_envs, self.obs_stacking + 1, 46, device=device)
        self.yaw_history = torch.zeros(self.num_envs, self.obs_stacking + 2, device=device)
        self.pos_history = torch.zeros(self.num_envs, self.obs_stacking + 2, 2, device=device)
        self.device = device
        
        self.start_yaw_of_segment = torch.tensor(
            [float(row[61]) for row in self.rows], device=device
        ).unsqueeze(1)
        
        self.yaw_0 = torch.tensor(
            [float(row[1]) for row in self.rows], device=device
        ).unsqueeze(1)
        
        self.start_pos_of_segment = torch.tensor(
            [[float(row[59]), float(row[60])] for row in self.rows], device=device
        )
        self.num_rows = [sum(1 for _ in reader) for reader in self.data_readers]
        self.current_rows = [0] * self.num_envs
        
        self.q0 = torch.zeros(self.num_envs, 13, device=self.device)
        self.q0[:, [1, 7]] = -0.2
        self.q0[:, [4, 10]] = 0.4
        self.q0[:, [5, 11]] = -0.25
        self.timestamp_inter = [0] * self.num_envs
        self.timestamp_mocap = [0] * self.num_envs

    def reset(self):
        for i, file in enumerate(self.data_files):
            file.seek(0)
            self.data_readers[i] = csv.reader(file)
            next(self.data_readers[i])  # Skip header
            self.rows[i] = next(self.data_readers[i])
        
        self.start_yaw_of_segment = torch.tensor(
            [float(row[61]) for row in self.rows], device=self.device
        ).unsqueeze(1)
        self.start_pos_of_segment = torch.tensor(
            [[float(row[59]), float(row[60])] for row in self.rows], device=self.device
        )
        
        project_gravity = torch.tensor(
            [[float(row[2]), float(row[3]), float(row[4])] for row in self.rows], device=self.device
        )
        ang_vel = torch.tensor(
            [[float(row[5]), float(row[6]), float(row[7])] for row in self.rows], device=self.device
        )
        q = torch.tensor(
            [[float(row[i]) for i in range(21, 34)] for row in self.rows], device=self.device
        ) - self.q0
        dq = torch.tensor(
            [[float(row[i]) for i in range(44, 57)] for row in self.rows], device=self.device
        ) * 0.1
        acc = torch.tensor(
            [[float(row[8]), float(row[9]), float(row[10])] for row in self.rows], device=self.device
        ) * 0.1
        actions = torch.zeros(self.num_envs, 11, device=self.device)
        actions = torch.tensor(
            [[float(row[i]) for i in range(64, 75)] for row in self.rows], device=self.device
        )
        
        self.odom_obs_history_wys[:] = torch.cat((project_gravity, ang_vel, q, dq, acc, actions), dim=-1).unsqueeze(1)
        
        yaw = torch.tensor(
            [[float(row[1]) - self.yaw_0[i, 0]] for i, row in enumerate(self.rows)], device=self.device
        )
        self.yaw_history[:] = yaw
        
        pos_groundtruth = torch.tensor(
            [
                [float(row[59]) - self.start_pos_of_segment[i, 0], float(row[60]) - self.start_pos_of_segment[i, 1]]
                for i, row in enumerate(self.rows)
            ],
            device=self.device,
        )
        pos_groundtruth = torch.stack(
            (
                torch.cos(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 0]
                + torch.sin(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 1],
                -torch.sin(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 0]
                + torch.cos(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 1],
            ),
            dim=-1,
        )
        
        self.timestamp_inter = [float(row[0]) for row in self.rows]
        self.timestamp_mocap = [float(row[57]) for row in self.rows]
        
        infos = {
            "odom_obs_history_wys": self.odom_obs_history_wys,
            "yaw_history": torch.zeros_like(self.yaw_history),
            "pos_history": torch.zeros_like(self.pos_history),
            "pos_groundtruth": pos_groundtruth,
            "abs_yaw_history": torch.zeros_like(self.yaw_history),
        }
        return infos

    def step(self):
        for i in range(self.num_envs):
            self.rows[i] = next(self.data_readers[i])
            self.current_rows[i] += 1
            if self.current_rows[i] >= self.num_rows[i]:
                self.data_files[i].close()
                return None, True
        
        project_gravity = torch.tensor(
            [[float(row[2]), float(row[3]), float(row[4])] for row in self.rows], device=self.device
        )
        ang_vel = torch.tensor(
            [[float(row[5]), float(row[6]), float(row[7])] for row in self.rows], device=self.device
        )
        q = torch.tensor(
            [[float(row[i]) for i in range(21, 34)] for row in self.rows], device=self.device
        ) - self.q0
        dq = torch.tensor(
            [[float(row[i]) for i in range(44, 57)] for row in self.rows], device=self.device
        ) * 0.1
        acc = torch.tensor(
            [[float(row[8]), float(row[9]), float(row[10])] for row in self.rows], device=self.device
        ) * 0.1
        actions = torch.zeros(self.num_envs, 11, device=self.device)
        actions = torch.tensor(
            [[float(row[i]) for i in range(64, 75)] for row in self.rows], device=self.device
        )
        
        self.odom_obs_history_wys = torch.roll(self.odom_obs_history_wys, -1, dims=1)
        self.odom_obs_history_wys[:, -1] = torch.cat((project_gravity, ang_vel, q, dq, acc, actions), dim=-1)
        
        yaw = torch.tensor(
            [[float(row[1]) - self.yaw_0[i, 0]] for i, row in enumerate(self.rows)], device=self.device
        )
        self.yaw_history = torch.roll(self.yaw_history, -1, dims=1)
        self.yaw_history[:, -1] = yaw.squeeze(1)
        
        pos_groundtruth = torch.tensor(
            [
                [float(row[59]) - self.start_pos_of_segment[i, 0], float(row[60]) - self.start_pos_of_segment[i, 1]]
                for i, row in enumerate(self.rows)
            ],
            device=self.device,
        )
        pos_groundtruth = torch.stack(
            (
                torch.cos(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 0]
                + torch.sin(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 1],
                -torch.sin(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 0]
                + torch.cos(self.start_yaw_of_segment[:, 0]) * pos_groundtruth[:, 1],
            ),
            dim=-1,
        )
        
        self.pos_history = torch.roll(self.pos_history, -1, dims=1)
        self.pos_history[:, -1] = pos_groundtruth
        
        self.timestamp_inter = [float(row[0]) for row in self.rows]
        self.timestamp_mocap = [float(row[57]) for row in self.rows]
        
        infos = {
            "odom_obs_history_wys": self.odom_obs_history_wys,
            "yaw_history": self.yaw_history - self.yaw_history[:, -51].unsqueeze(1),
            "pos_history": torch.stack(
                (
                    torch.cos(self.yaw_history[:, 1].unsqueeze(1)) * (self.pos_history[:, :, 0] - self.pos_history[:, -51, 0].unsqueeze(1))
                    + torch.sin(self.yaw_history[:, 1].unsqueeze(1)) * (self.pos_history[:, :, 1] - self.pos_history[:, -51, 1].unsqueeze(1)),
                    -torch.sin(self.yaw_history[:, 1].unsqueeze(1)) * (self.pos_history[:, :, 0] - self.pos_history[:, -51, 0].unsqueeze(1))
                    + torch.cos(self.yaw_history[:, 1].unsqueeze(1)) * (self.pos_history[:, :, 1] - self.pos_history[:, -51, 1].unsqueeze(1)),
                ),
                dim=-1,
            ),
            "pos_groundtruth": self.pos_history,
            "abs_yaw_history": self.yaw_history,
        }
        
        return infos, False