import os
import csv
import time
import numpy as np
from typing import List
import threading
import logging
from nokov.nokovsdk import PySDKClient, POINTER, DataDescriptions, DataDescriptors
from booster_robotics_sdk_python import B1LowStateSubscriber, LowState, B1JointCnt, ChannelFactory


def rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R_z @ R_y @ R_x).T @ vector


class MotionCapture:

    def __init__(self, dir):
        SERVER_IP = "192.168.200.154"
        self.client = PySDKClient()
        logging.info("Begin to init the SDK Client")
        ret = self.client.Initialize(bytes(SERVER_IP, encoding="utf8"))
        if ret == 0:
            logging.info("Connect to the Nokov Succeed")
        else:
            logging.error(f"Connect Failed: {ret}")
            raise ConnectionError("Failed to connect to the Nokov server.")
        self.dataset = []
        self.filename = os.path.join(dir, "mocap.csv")
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "timestamp", "robot_x", "robot_y", "robot_yaw"])
        self.save_thread = threading.Thread(target=self.save_data)
        self.save_thread.daemon = True
        self.save_thread.start()
        self.lock = threading.Lock()
        self.preFrmNo = None
        self.client.PySetDataCallback(self.py_data_func, None)

    def py_data_func(self, pFrameOfMocapData, pUserData):
        timestamp = time.time()
        if pFrameOfMocapData is None:
            logging.warning("Not get the data frame.")
            return

        frameData = pFrameOfMocapData.contents
        if frameData.iFrame == self.preFrmNo:
            return
        self.preFrmNo = frameData.iFrame

        length = 128
        szTimeCode = bytes(length)
        self.client.PyTimecodeStringify(frameData.Timecode, frameData.TimecodeSubframe, szTimeCode, length)
        data = [timestamp, frameData.iTimeStamp]
        for i in range(frameData.nRigidBodies):
            body = frameData.RigidBodies[i]
            if abs(body.x) < 20000 and abs(body.y) < 20000:
                data.extend(
                    [
                        body.x * 1.0e-3,
                        body.y * 1.0e-3,
                        np.arctan2(
                            2.0 * (body.qw * body.qz + body.qx * body.qy),
                            body.qw * body.qw + body.qx * body.qx - body.qy * body.qy - body.qz * body.qz,
                        ),
                    ]
                )
                break
        if len(data) == 5:
            with self.lock:
                self.dataset.append(data)

    def save_data(self):
        while True:
            time.sleep(10)
            with self.lock:
                dataset = self.dataset.copy()
                self.dataset.clear()
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                for data in dataset:
                    writer.writerow(data)

class BoosterSDK:

    def __init__(self, dir):
        self.base_yaw_zero = None
        self.dataset = []
        self.filename = os.path.join(dir, "booster.csv")
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "yaw", "projected_gravity", "ang_vel", "lin_acc", "q", "dq"])
        self.save_thread = threading.Thread(target=self.save_data)
        self.save_thread.daemon = True
        self.save_thread.start()
        self.lock = threading.Lock()
        try:
            ChannelFactory.Instance().Init(0)
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_state_subscriber.InitChannel()
        except Exception as e:
            logging.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        data = [time.time()]
        if self.base_yaw_zero is None:
            self.base_yaw_zero = low_state_msg.imu_state.rpy[2]
        data.append(low_state_msg.imu_state.rpy[2] - self.base_yaw_zero)
        data.extend(
            rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
        )
        data.extend(low_state_msg.imu_state.gyro)
        data.extend(low_state_msg.imu_state.acc)
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            data.append(motor.q)
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            data.append(motor.dq)
        with self.lock:
            self.dataset.append(data)

    def save_data(self):
        while True:
            time.sleep(10)
            with self.lock:
                dataset = self.dataset.copy()
                self.dataset.clear()
            with open(self.filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                for data in dataset:
                    writer.writerow(data)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dir = os.path.join("data", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(dir, exist_ok=True)
    motion_capture = MotionCapture(dir)
    booster_sdk = BoosterSDK(dir)
    logging.info("Initialization complete.")
    while True:
        time.sleep(1)
