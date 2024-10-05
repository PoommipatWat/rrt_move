import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from custom_interfaces.srv import SetGoal
from std_msgs.msg import Bool

class Controller(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.odom_sub_ = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.robot_pose = None
        self.goal = None
        self.timer_ = self.create_timer(0.1, self.timer_callback)
        self.state = 0
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.angular_speed = 0.2
        self.linear_speed = 0.10
        self.status_pub = self.create_publisher(Bool, 'control_status', 10)
        self.error_deg = 5.0
        self.error_dis = 0.1
        self.srv = self.create_service(SetGoal, 'set_goal', self.set_goal_callback)
        self.obs_sub = self.create_subscription(Bool, 'obstacle', self.obstacle_callback, 10)
        self.obstacle_detected = False
        self.goal_count = 0
        self.waiting_for_new_path = False

    def set_goal_callback(self, request, response):
        self.goal = (request.x, request.y)
        self.get_logger().info(f"New goal set: {self.goal}")
        self.state = 0
        self.waiting_for_new_path = False
        response.success = True
        return response

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data
        if self.obstacle_detected and self.goal_count > 2 and not self.waiting_for_new_path:
            self.cmd_pub.publish(Twist())  # Stop the robot
            self.state = 0
            self.waiting_for_new_path = True
            self.status_pub.publish(Bool(data=False))
            self.get_logger().info('Obstacle detected, stopping robot and requesting new path')
            self.goal_count = 0

    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        x, y, z, w = orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        _, _, yaw = self.quaternion_to_euler(x, y, z, w)
        yaw_degrees = np.rad2deg(yaw)
        if yaw_degrees < 0:
            yaw_degrees += 360
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_degrees)

    def timer_callback(self):
        if self.goal is not None and not self.waiting_for_new_path:
            if self.goal_count <= 2 or not self.obstacle_detected:
                if self.state == 0:
                    self.rotate_to_goal()
                elif self.state == 1:
                    self.move_to_goal()
            else:
                self.cmd_pub.publish(Twist())  # Stop if obstacle detected after third goal

    def rotate_to_goal(self):
        if self.robot_pose is not None:
            x, y, yaw = self.robot_pose
            dx = self.goal[0] - x
            dy = self.goal[1] - y
            target_angle = np.rad2deg(np.arctan2(dy, dx))
            if target_angle < 0:
                target_angle += 360
            angle_diff = self.shortest_angle_diff(target_angle, yaw)
            msg = Twist()
            if abs(angle_diff) > self.error_deg:
                msg.angular.z = np.sign(angle_diff) * self.angular_speed
            else:
                msg.angular.z = 0.0
                self.state = 1
            self.cmd_pub.publish(msg)

    def move_to_goal(self):
        if self.robot_pose is not None:
            x, y, _ = self.robot_pose
            dx = self.goal[0] - x
            dy = self.goal[1] - y
            distance = np.hypot(dx, dy)
            msg = Twist()
            if distance > self.error_dis:
                msg.linear.x = self.linear_speed
            else:
                msg.linear.x = 0.0
                self.get_logger().info(f'Goal reached: {self.goal_count + 1}')
                self.goal = None
                self.state = 0
                self.goal_count += 1
                self.status_pub.publish(Bool(data=True))
            self.cmd_pub.publish(msg)

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def shortest_angle_diff(self, target_angle, current_angle):
        angle_diff = target_angle - current_angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        return angle_diff

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()