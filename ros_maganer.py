import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_srvs.srv import SetBool
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from custom_interfaces.srv import SetGoal

import numpy as np
import time

class Manager(Node):
    def __init__(self):
        super().__init__('manager_node')
        self.state = 0
        self.timer_ = self.create_timer(0.1, self.timer_callback)
        self.plan_path_client = self.create_client(SetBool, 'plan_path')
        self.set_goal_client = self.create_client(SetGoal, 'set_goal')
        self.waiting_for_response = False
        self.path = None
        self.path_sub = self.create_subscription(Marker, '/path_marker', self.path_callback, 10)
        self.index_path = 1
        self.control_status_sub = self.create_subscription(Bool, 'control_status', self.control_status_callback, 10)
        self.control_status = False
        self.obs_sub_ = self.create_subscription(Bool, 'obstacle', self.obs_callback, 10)
        self.obs_status = False
        self.replanning = False
        self.get_logger().info('Manager node initialized')

    def obs_callback(self, msg):
        self.obs_status = msg.data

    def control_status_callback(self, msg):
        self.control_status = msg.data
        if not self.control_status and not self.replanning:
            # This indicates that an obstacle was encountered and we need to replan
            self.get_logger().info('Obstacle encountered after 3 goals, requesting new path')
            self.replanning = True
            self.request_path()
        elif self.control_status and self.state == 2:
            self.get_logger().info('Goal reached, sending next goal')
            self.send_next_goal()

    def timer_callback(self):
        if self.state == 0 and not self.waiting_for_response and not self.replanning:
            self.request_path()
        elif self.state == 1 and self.path is not None and not self.replanning:
            self.send_next_goal()

    def request_path(self):
        self.get_logger().info('Requesting path')
        if not self.plan_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path planning service not available, trying again')
            return
        request = SetBool.Request()
        request.data = True
        future = self.plan_path_client.call_async(request)
        future.add_done_callback(self.path_response_callback)
        self.waiting_for_response = True

    def path_response_callback(self, future):
        self.waiting_for_response = False
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Path planning successful')
                self.state = 1
                self.index_path = 1
                self.replanning = False
            else:
                self.get_logger().info('Path planning failed, retrying')
                self.request_path()
        except Exception as e:
            self.get_logger().error(f'Path planning service call failed: {str(e)}')
            self.request_path()

    def path_callback(self, msg):
        if len(msg.points) == 0:
            self.get_logger().warn('Received empty path')
            return
        self.path = [[point.x, point.y] for point in msg.points]
        self.get_logger().info(f'Received path with {len(self.path)} points')
        if self.state == 1 and not self.replanning:
            self.send_next_goal()

    def send_next_goal(self):
        if self.path and self.index_path < len(self.path):
            self.send_goal(self.path[self.index_path][0], self.path[self.index_path][1])
            self.index_path += 1
            self.state = 2
        else:
            self.get_logger().info('Path completed')
            self.state = 0
            self.path = None
            self.index_path = 1

    def send_goal(self, x, y):
        request = SetGoal.Request()
        request.x = float(x)
        request.y = float(y)
        self.get_logger().info(f'Sending goal: ({x}, {y}), Index: {self.index_path}')
        future = self.set_goal_client.call_async(request)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Goal set successfully')
            else:
                self.get_logger().info('Failed to set goal')
        except Exception as e:
            self.get_logger().error(f'Set goal service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    manager = Manager()
    rclpy.spin(manager)
    manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()