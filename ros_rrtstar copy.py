import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import numpy as np
from ai_samples.rrt_star import RRTStar
import matplotlib.pyplot as plt

class RRTStarPlanner(Node):
    def __init__(self):
        super().__init__('rrt_star_planner')
        # Subscribers
        self.occupancy_sub = self.create_subscription(
            OccupancyGrid,
            '/occupancy_grid',
            self.occupancy_callback,
            10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)
        
        # Publisher
        self.marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        
        # Initialize variables
        self.map_data = None
        self.map_origin = None
        self.map_resolution = None
        self.robot_position = None
        self.goal_position = None
        self.robot_radius = 0.2  # Adjusted for Turtlebot (in meters)
        
        # Timer for path planning
        self.create_timer(0.1, self.plan_path)  # Plan path every 0.1 seconds
        
        self.get_logger().info('RRT* Planner node initialized')

    def occupancy_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        self.map_data = np.array(msg.data).reshape((height, width))
        self.map_data = (self.map_data > 50).astype(int)  # Threshold occupancy
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.get_logger().info('Map received')

    def odom_callback(self, msg):
        if self.map_origin is None or self.map_resolution is None:
            self.get_logger().warn('Map info not available yet')
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_position = self.world_to_grid(x, y)
        self.goal_position = self.world_to_grid(1.5,1.5)
        self.get_logger().info(f'Robot position updated: {self.robot_position}')

    def goal_callback(self, msg):
        if self.map_origin is None or self.map_resolution is None:
            self.get_logger().warn('Map info not available yet')
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.goal_position = self.world_to_grid(x, y)
        self.get_logger().info(f'Goal position set: {self.goal_position}')

    def world_to_grid(self, x, y):
        if self.map_origin is None or self.map_resolution is None:
            self.get_logger().error('Map info not available for world_to_grid conversion')
            return None
        gx = int((x - self.map_origin[0]) / self.map_resolution)
        gy = int((y - self.map_origin[1]) / self.map_resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        if self.map_origin is None or self.map_resolution is None:
            self.get_logger().error('Map info not available for grid_to_world conversion')
            return None
        x = gx * self.map_resolution + self.map_origin[0]
        y = gy * self.map_resolution + self.map_origin[1]
        return (x, y)

    def plan_path(self):
        if self.map_data is None or self.robot_position is None or self.goal_position is None:
            self.get_logger().warn('Waiting for map, robot position, and goal...')
            return
        
        self.get_logger().info(f"Planning path from {self.robot_position} to {self.goal_position}")
        
        max_iterations = 5000
        max_travel_distance = 1.0  # in meters
        search_radius = 1.0  # in meters
        goal_radius = 0.2  # in meters

        rrt_star = RRTStar(
            self.map_data, 
            self.robot_position, 
            self.goal_position, 
            int(self.robot_radius / self.map_resolution),
            max_iterations=max_iterations,
            max_travel_distance=int(max_travel_distance / self.map_resolution),
            search_radius=int(search_radius / self.map_resolution),
            goal_radius=int(goal_radius / self.map_resolution)
        )
        path = rrt_star.build()

        if path:
            self.get_logger().info(f'Path found with {len(path)} waypoints')
            self.publish_path_marker(path)
        else:
            self.get_logger().warn('No path found')

    def publish_path_marker(self, path):
        # Publish path as LINE_STRIP
        path_marker = self.create_line_strip_marker(path, 'path', 0, [0.0, 1.0, 0.0, 1.0])  # Green path
        self.marker_pub.publish(path_marker)

        # Publish start point (robot position)
        if self.robot_position:
            start_marker = self.create_point_marker(self.robot_position, 'start', 1, [0.0, 0.0, 1.0, 1.0])  # Blue start point
            self.marker_pub.publish(start_marker)

        # Publish end point (goal position)
        if self.goal_position:
            end_marker = self.create_point_marker(self.goal_position, 'goal', 2, [1.0, 0.0, 0.0, 1.0])  # Red end point
            self.marker_pub.publish(end_marker)

        self.get_logger().info(f"Published path marker with {len(path)} points and start/end markers")

    def create_line_strip_marker(self, points, ns, id, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Line width
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])

        for point in points:
            world_point = self.grid_to_world(point[0], point[1])
            if world_point is not None:
                x, y = world_point
                marker.points.append(Point(x=x, y=y, z=0.0))

        return marker

    def create_point_marker(self, point, ns, id, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        world_point = self.grid_to_world(point[0], point[1])
        if world_point is not None:
            x, y = world_point
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # Point size
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        return marker

    def visualize_path(self, rrt_star, path):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the map
        ax.imshow(self.map_data, cmap='binary')
        
        # Plot start and goal
        ax.plot(self.robot_position[0], self.robot_position[1], 'go', markersize=8, label='Start')
        ax.plot(self.goal_position[0], self.goal_position[1], 'ro', markersize=8, label='Goal')
        
        # Plot RRT tree
        for node in rrt_star.nodes:
            if node.parent:
                ax.plot([node.position[0], node.parent.position[0]], 
                        [node.position[1], node.parent.position[1]], 'c-', alpha=0.3)
        
        # Plot the path
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
        
        ax.set_title("RRT* Path Planning")
        ax.legend()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    rrt_star_planner = RRTStarPlanner()
    rclpy.spin(rrt_star_planner)
    rrt_star_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()