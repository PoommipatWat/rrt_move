import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool

from sensor_msgs.msg import LaserScan

import numpy as np

class obstacle_avoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.sub_ = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.sub_

        self.pub_ = self.create_publisher(Bool, 'obstacle', 10)

        self.deg = 40
        self.distances = 0.35

    def scan_callback(self, msg):
        used = []
        angle_range = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment) * 180 / np.pi
        for i, val in enumerate(angle_range):
            # if val <= self.deg or val >= 360 - self.deg:
            #     used.append(msg.ranges[i])
            used.append(msg.ranges[i])
        used = np.array(used)

        bools = Bool()
        bools.data = bool(np.any(used <= self.distances))
        self.pub_.publish(bools)



    


if __name__ == '__main__':
    rclpy.init()
    node = obstacle_avoidance()
    rclpy.spin(node)
    rclpy.shutdown()


    