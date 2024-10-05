import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
from ai_samples.OccupancyGridMap import OccupancyGridMap

class TEST(Node):
    def __init__(self):
        super().__init__('test')
        self.subscription_scan = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.subscription_odom = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)

        self.timer_ = self.create_timer(0.01, self.timer_callback)
        
        # Initialize OccupancyGridMap
        self.ogm = OccupancyGridMap(-10, 10, -10, 10, cell_size=0.05)
        
        # Robot's position (will be updated by odom)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.msg_data = None
        self.update_count = 0
        self.log_interval = 100  # Log every 100 updates

    def odom_callback(self, msg):
        # Update robot's position from odometry data
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        # Convert quaternion to Euler angles
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.msg_data = msg

    def timer_callback(self):
        msg = self.msg_data

        if msg is not None:
            # Convert angle range to radians (LaserScan angles are typically in radians)
            angle_range = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
            ranges = np.array(msg.ranges)
            
            # Filter out inf and nan values
            valid_indices = np.isfinite(ranges)
            valid_angles = angle_range[valid_indices]
            valid_ranges = ranges[valid_indices]
            
            # Calculate end points of the laser beams in robot's local frame
            x_local = valid_ranges * np.cos(valid_angles)
            y_local = valid_ranges * np.sin(valid_angles)
            
            # Transform to global frame
            cos_yaw = np.cos(self.robot_yaw)
            sin_yaw = np.sin(self.robot_yaw)
            x_global = x_local * cos_yaw - y_local * sin_yaw + self.robot_x
            y_global = x_local * sin_yaw + y_local * cos_yaw + self.robot_y
            
            # Update the occupancy grid map
            for x, y in zip(x_global, y_global):
                self.ogm.update_line(self.robot_x, self.robot_y, x, y)
                self.update_count += 1
            
            # Publish the updated map
            self.publish_map()

    def publish_map(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"  # Adjust if needed
        
        grid_msg.info.resolution = self.ogm.cell_size
        grid_msg.info.width = self.ogm.grid_width
        grid_msg.info.height = self.ogm.grid_height
        
        grid_msg.info.origin.position.x = float(self.ogm.x_min)
        grid_msg.info.origin.position.y = float(self.ogm.y_min)
        
        # Convert probability values to occupancy values (0-100)
        occupancy_data = (self.ogm.get_map() * 100).astype(int)
        occupancy_data = np.clip(occupancy_data, 0, 100)
        
        grid_msg.data = occupancy_data.flatten().tolist()
        
        self.map_publisher.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    test = TEST()
    rclpy.spin(test)
    test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()