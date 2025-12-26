import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher_node')
        
        # Must match the goal_topic parameter of mppi_node.
        self.topic_name = "/mppi/goal"
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)
        
        # Change goal every 5 seconds
        self.timer = self.create_timer(5.0, self.publish_goal)
        
        # Test goal positions (x, y, z) - considering Franka Panda workspace
        self.waypoints = [
            [0.4, 0.0, 0.8],   # front middle
            [0.6, 0.1, 0.8],   # lower left
            [0.6, -0.3, 0.6],  # upper right
            [0.4, 0.0, 0.6],   # inner low position
        ]
        self.current_idx = 0
        
        self.get_logger().info(f"Goal Publisher Started. Publishing to {self.topic_name}")

    def publish_goal(self):
        target = self.waypoints[self.current_idx]
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"  # reference frame (must match with mppi_node)

        # Set position
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = float(target[2])

        # Set orientation (for now keep upright: w=1.0)
        # If needed, you can change the quaternion (x,y,z,w) to give rotation commands as well
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        self.publisher_.publish(msg)
        
        self.get_logger().info(f"Published Goal [{self.current_idx}]: {target}")

        # Move to next index (cyclic)
        self.current_idx = (self.current_idx + 1) % len(self.waypoints)

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
