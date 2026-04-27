"""
Control ROS 2 Node
Wraps the vehicle controller to provide ROS 2 topic interface.

Subscribed Topics:
    /planning/path (nav_msgs/Path): Planned trajectory
    /planning/target_speed (std_msgs/Float64): Target speed
    /localization/pose (geometry_msgs/PoseStamped): Current pose
    /localization/twist (geometry_msgs/Twist): Current velocity

Published Topics:
    /control/cmd_vel (geometry_msgs/Twist): Velocity command
    /control/steering_angle (std_msgs/Float64): Steering angle
    /control/emergency_stop (std_msgs/Bool): Emergency stop flag

Parameters:
    pid_kp (float): Proportional gain
    pid_ki (float): Integral gain
    pid_kd (float): Derivative gain
    max_steering_angle (float): Maximum steering angle in degrees
"""
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import rclpy
    from rclpy.node import Node
    
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, Twist
    from std_msgs.msg import Float64, Bool
    
    ROS2_AVAILABLE = True
except ImportError:
    print("[WARNING] ROS 2 not available.")
    ROS2_AVAILABLE = False
    Node = object

try:
    from control.vehicle_controller import VehicleController
    from planning.apf_planner import PlannerOutput
    CONTROL_AVAILABLE = True
except ImportError:
    print("[WARNING] Control module not available.")
    CONTROL_AVAILABLE = False


class ControlNode(Node):
    """ROS 2 node for vehicle control."""
    
    def __init__(self):
        if not ROS2_AVAILABLE or not CONTROL_AVAILABLE:
            raise RuntimeError("Required dependencies not available.")
        
        super().__init__('control_node')
        
        # Parameters
        self.declare_parameter('pid_kp', 1.0)
        self.declare_parameter('pid_ki', 0.0)
        self.declare_parameter('pid_kd', 0.1)
        self.declare_parameter('max_steering_angle', 70.0)
        self.declare_parameter('control_rate', 20.0)
        
        # Initialize controller
        config = {
            'pid': {
                'steering': {
                    'kp': self.get_parameter('pid_kp').value,
                    'ki': self.get_parameter('pid_ki').value,
                    'kd': self.get_parameter('pid_kd').value,
                }
            },
            'max_steering_angle': self.get_parameter('max_steering_angle').value
        }
        self.controller = VehicleController(config)
        
        # State
        self.current_path = None
        self.target_speed = 0.0
        self.current_pose = None
        self.current_twist = None
        self.state_lock = threading.Lock()
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/planning/path', self._path_callback, 10
        )
        self.speed_sub = self.create_subscription(
            Float64, '/planning/target_speed', self._speed_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/localization/pose', self._pose_callback, 10
        )
        self.twist_sub = self.create_subscription(
            Twist, '/localization/twist', self._twist_callback, 10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/control/cmd_vel', 10)
        self.steering_pub = self.create_publisher(Float64, '/control/steering_angle', 10)
        self.emergency_pub = self.create_publisher(Bool, '/control/emergency_stop', 10)
        
        # Timer
        rate = self.get_parameter('control_rate').value
        self.timer = self.create_timer(1.0 / rate, self._control_loop)
        
        self.get_logger().info("ControlNode initialized")
    
    def _path_callback(self, msg: Path):
        """Update planned path."""
        with self.state_lock:
            self.current_path = msg
    
    def _speed_callback(self, msg: Float64):
        """Update target speed."""
        with self.state_lock:
            self.target_speed = msg.data
    
    def _pose_callback(self, msg: PoseStamped):
        """Update current pose."""
        with self.state_lock:
            self.current_pose = msg
    
    def _twist_callback(self, msg: Twist):
        """Update current velocity."""
        with self.state_lock:
            self.current_twist = msg
    
    def _control_loop(self):
        """Main control loop."""
        with self.state_lock:
            path = self.current_path
            target_speed = self.target_speed
            pose = self.current_pose
            twist = self.current_twist
        
        if pose is None:
            return
        
        try:
            # Extract current state
            current_x = pose.pose.position.x
            current_y = pose.pose.position.y
            current_speed = 0.0
            if twist is not None:
                current_speed = (twist.linear.x**2 + twist.linear.y**2)**0.5
            
            # Get next waypoint from path
            target_point = None
            if path and len(path.poses) > 0:
                # Find closest point ahead
                for pose_msg in path.poses:
                    px = pose_msg.pose.position.x
                    py = pose_msg.pose.position.y
                    dist = ((px - current_x)**2 + (py - current_y)**2)**0.5
                    if dist > 2.0:  # Look ahead at least 2m
                        target_point = (px, py)
                        break
                if target_point is None and path.poses:
                    target_point = (path.poses[-1].pose.position.x, 
                                   path.poses[-1].pose.position.y)
            
            # Run controller - build PlannerOutput from planning data
            if target_point:
                import math
                dx = target_point[0] - current_x
                dy = target_point[1] - current_y
                target_yaw = math.degrees(math.atan2(dy, dx))
                # Compute steering from heading error
                yaw_error = target_yaw - (self.current_pose.pose.orientation.z * 100)  # rough approx
                steering = max(-1.0, min(1.0, yaw_error / 45.0))

                # Build a PlannerOutput to pass to controller
                planner_out = PlannerOutput(
                    target_steering=steering,
                    target_speed=target_speed,
                    emergency_brake=False,
                    status="normal",
                    trajectory=[(p.pose.position.x, p.pose.position.y) for p in path.poses] if path else [],
                )

                control_output = self.controller.compute_control(
                    planner_out,
                    current_speed_kmh=current_speed * 3.6,
                )

                # Publish control commands
                cmd_vel = Twist()
                cmd_vel.linear.x = control_output.get('throttle', 0.0)
                cmd_vel.angular.z = control_output.get('steer', 0.0)
                self.cmd_vel_pub.publish(cmd_vel)

                steering_msg = Float64()
                steering_msg.data = control_output.get('steer', 0.0)
                self.steering_pub.publish(steering_msg)

                emergency = Bool()
                emergency.data = control_output.get('brake', 0.0) >= 0.8
                self.emergency_pub.publish(emergency)
                
        except Exception as e:
            self.get_logger().error(f"Control error: {e}")


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS 2 required but not available.")
        return 1
    
    rclpy.init(args=args)
    try:
        node = ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
