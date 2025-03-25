import numpy as np
import threading
import rclpy
from rclpy.node import Node
from vision_interface.msg import Detections


class RosSubscriber(Node):

    def __init__(self, policy):
        super().__init__("ros_subscriber")
        self.policy = policy
        self.base_pos = np.zeros(2)
        self.ball_pos = np.zeros(2)
        self.init = True
        self.subscription = self.create_subscription(Detections, "/booster_vision/detection", self.ros_callback, 10)

        self.spin_thread = threading.Thread(target=self.spin_node)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def ros_callback(self, msg):
        for obj in msg.detected_objects:
            if obj.label == "Ball" and obj.confidence > 0.3 and abs(obj.position_projection[0]) < 20 and abs(obj.position_projection[1]) < 20:
                if self.init:
                    weight = 1.0
                    self.init = False
                else:
                    weight = 0.3
                self.ball_pos[0] = (1.0 - weight) * self.ball_pos[0] + weight * (
                    self.policy.odom_pos[0]
                    + np.cos(self.policy.base_yaw) * obj.position_projection[0]
                    - np.sin(self.policy.base_yaw) * obj.position_projection[1]
                )
                self.ball_pos[1] = (1.0 - weight) * self.ball_pos[1] + weight * (
                    self.policy.odom_pos[1]
                    + np.sin(self.policy.base_yaw) * obj.position_projection[0]
                    + np.cos(self.policy.base_yaw) * obj.position_projection[1]
                )

    def spin_node(self):
        rclpy.spin(self)
