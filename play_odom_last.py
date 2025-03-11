import isaacgym
import os
import glob
import yaml
import argparse
import torch
import torch.nn.functional as F
from utils.wrapper import OdomWrapper
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.recorder import Recorder
from envs import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = args.task
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint
    cfg["viewer"]["record_video"] = False

    cfg["env"]["num_envs"] = 1
    cfg["basic"]["headless"] = False

    task_class = eval(args.task)
    env = task_class(cfg)
    env = OdomWrapper(env, cfg["runner"]["num_stack"])
    device = cfg["basic"]["rl_device"]
    learning_rate = cfg["algorithm"]["learning_rate"]
    model = DenoisingRMA(env.num_actions, env.num_obs, env.num_stack, env.num_privileged_obs, cfg["algorithm"]["num_embedding"]).to(device)

    odom_model = OdomEstimator(32 + 4, env.num_stack).to(device)
    odom_checkpoint = "/home/lcs/RCL_Project/train_odom/logs/2025-03-09-10-55-02/nn/model_100.pth"
    odom_model.load_state_dict(torch.load(odom_checkpoint, map_location=device, weights_only=True)["model"], strict=False)

    optimizer = torch.optim.Adam(odom_model.parameters(), lr=learning_rate)
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(model_dict["model"], strict=False)

    buffer = ExperienceBuffer(cfg["runner"]["horizon_length"], env.num_envs, device)
    buffer.add_buffer("stacked_obses", (env.num_stack, env.num_obs))
    buffer.add_buffer("stacked_odom_obses", (env.num_stack, 32))
    buffer.add_buffer("stacked_yaws", (env.num_stack,))
    buffer.add_buffer("stacked_poses", (env.num_stack + 1, 2))

    obs, infos = env.reset()
    obs = obs.to(device)
    stacked_obs = infos["stacked_obs"].to(device)
    stacked_odom_obs = infos["stacked_odom_obs"].to(device)
    stacked_yaw = infos["stacked_yaw"].to(device)
    stacked_pos = infos["stacked_pos"].to(device)

    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化图形
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='odom_pred')
    line2, = ax.plot([], [], label='true_poses')
    ax.legend()

    list_of_odom_preds = []
    list_of_true_poses = []
    for it in range(100):
        for n in range(cfg["runner"]["horizon_length"]):
            buffer.update_data("stacked_obses", n, stacked_obs)
            buffer.update_data("stacked_odom_obses", n, stacked_odom_obs)
            buffer.update_data("stacked_yaws", n, stacked_yaw)
            buffer.update_data("stacked_poses", n, stacked_pos)
            with torch.no_grad():
                dist, _ = model.act(obs, stacked_obs=stacked_obs)
                act = dist.sample()
            obs, rew, done, infos = env.step(act)
            stacked_obs = infos["stacked_obs"].to(device)
            stacked_odom_obs = infos["stacked_odom_obs"].to(device)
            stacked_yaw = infos["stacked_yaw"].to(device)
            stacked_pos = infos["stacked_pos"].to(device)

        statistics = {}
        odom_pred = odom_model(buffer["stacked_odom_obses"], buffer["stacked_yaws"], buffer["stacked_poses"][..., 1:, :])
        # print(f"Odom Prediction size: {odom_pred.size()}")
        # odom_loss = F.mse_loss(odom_pred, buffer["stacked_poses"][..., 0, :])
        # print(f"Odom Prediction size: {odom_pred.size()}")
        # print(f"True Position size: {buffer['stacked_poses'][..., 0, :].size()}")
        if it < 3:
            continue
        list_of_odom_preds.append(odom_pred[0].detach().cpu().numpy())
        list_of_true_poses.append(buffer["stacked_poses"][..., 0, :][0].detach().cpu().numpy())

        # 更新图形数据
        list_of_odom_preds_np = np.array(list_of_odom_preds)
        list_of_true_poses_np = np.array(list_of_true_poses)
        # if list_of_true_poses_np[:, 0, 0] < -5 or list_of_true_poses_np[:, 0, 0] > 5:
        #     continue
        line1.set_data(list_of_odom_preds_np[:, 0, 0], list_of_odom_preds_np[:, 0, 1])
        line2.set_data(list_of_true_poses_np[:, 0, 0], list_of_true_poses_np[:, 0, 1])
        ax.relim()  # 重新计算坐标轴范围
        ax.autoscale_view()  # 自动缩放视图
        plt.draw()  # 绘制当前图形
        plt.pause(0.01)  # 暂停以便更新图形
    
    # 关闭交互模式
    plt.ioff()
    plt.show()

    # # 画图
    # import matplotlib.pyplot as plt
    # import numpy as np
    # list_of_odom_preds = np.array(list_of_odom_preds)
    # list_of_true_poses = np.array(list_of_true_poses)
    # # print(list_of_odom_preds)
    # plt.figure()
    # print(list_of_odom_preds.shape)
    # print(list_of_true_poses.shape)
    # plt.plot(list_of_odom_preds[:, 0, 0], list_of_odom_preds[:, 0, 1], label='odom_pred')
    # plt.plot(list_of_true_poses[:, 0, 0], list_of_true_poses[:, 0, 1], label='true_poses')
    # plt.legend()
        
        

