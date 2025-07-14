#!/usr/bin/env python3

import rospy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2 # For LiDAR, though not directly used in Pure Pursuit core
from geometry_msgs.msg import Point
import tf.transformations as tf_trans

# --- Configuration Parameters ---
TARGET_SPEED_KMPH = 10.0  # Target speed in km/h
TARGET_SPEED_MPS = TARGET_SPEED_KMPH / 3.6  # Convert to m/s

WHEELBASE_LENGTH = 1.6  # [m] - IMPORTANT: Adjust this value to your FSDS car's wheelbase!
LOOKAHEAD_DISTANCE = 3.0 # [m] - Tune this value for path tracking performance
                          # Larger -> smoother, less responsive
                          # Smaller -> more responsive, potentially unstable

# --- Global Variables ---
current_pose = None
path_waypoints = [] # List of geometry_msgs.Point objects representing the desired path
ackermann_pub = None

class PurePursuitController:
    def __init__(self):
        global ackermann_pub

        rospy.init_node('lidar_pure_pursuit_controller', anonymous=True)

        # Publishers
        ackermann_pub = rospy.Publisher('/fsds/ackermann_cmd', AckermannDriveStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)
        # We subscribe to LiDAR, but for *this* Pure Pursuit implementation,
        # it's mainly to confirm data flow. A more advanced system would use LiDAR
        # for obstacle avoidance, localization, or dynamic path planning.
        rospy.Subscriber('/fsds/lidar/front_lidar', PointCloud2, self.lidar_callback) # IMPORTANT: Replace 'front_lidar' with your actual LIDAR_NAME

        # Define a simple circular path (for demonstration)
        self.define_circular_path(radius=10.0, num_points=100)
        rospy.loginfo(f"Defined path with {len(path_waypoints)} waypoints.")

        rospy.spin()

    def define_circular_path(self, radius, num_points):
        global path_waypoints
        center_x, center_y = 0.0, -radius # Start at (0,0) heading +x, so center is behind
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoint = Point()
            waypoint.x = x
            waypoint.y = y
            waypoint.z = 0.0 # Assuming 2D path
            path_waypoints.append(waypoint)

    def odom_callback(self, msg):
        global current_pose
        current_pose = msg.pose.pose
        self.control_loop() # Call control loop whenever new odometry is received

    def lidar_callback(self, msg):
        # In a full autonomous system, LiDAR data would be used for:
        # 1. Obstacle detection and avoidance
        # 2. Localization (e.g., matching to a map)
        # 3. Dynamic path re-planning (e.g., avoiding dynamic obstacles)
        # For a basic Pure Pursuit, we don't *directly* use it for steering calculation.
        # However, you might want to add a simple check here for immediate obstacles.
        pass

    def control_loop(self):
        if current_pose is None or not path_waypoints:
            rospy.loginfo("Waiting for current pose or path waypoints...")
            return

        # 1. Get current vehicle position and orientation
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        
        # Convert quaternion to yaw (Z-axis rotation)
        quat = current_pose.orientation
        _, _, current_yaw = tf_trans.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        # 2. Find the closest point on the path to the vehicle
        distances = np.array([np.sqrt((wp.x - current_x)**2 + (wp.y - current_y)**2) for wp in path_waypoints])
        closest_point_idx = np.argmin(distances)
        
        # 3. Find the lookahead point
        lookahead_point = None
        for i in range(closest_point_idx, len(path_waypoints)):
            wp = path_waypoints[i]
            dist_from_current = np.sqrt((wp.x - current_x)**2 + (wp.y - current_y)**2)
            if dist_from_current > LOOKAHEAD_DISTANCE:
                lookahead_point = wp
                break
        
        # If no lookahead point found (e.g., at end of path), use the last point
        if lookahead_point is None:
            lookahead_point = path_waypoints[-1]
            # Optionally, if at the end of the path, stop the car
            # self.publish_ackermann_cmd(0.0, 0.0)
            # rospy.loginfo("Reached end of path. Stopping.")
            # return

        # 4. Calculate the angle alpha (angle between vehicle heading and lookahead vector)
        # Translate lookahead point to vehicle's coordinate frame (rear axle)
        # For simplicity, we'll use current_x, current_y as the reference point for the vector to lookahead
        # A more precise Pure Pursuit uses the rear axle's position.
        # Assuming current_pose is already 'base_link' (center of rear axle or vehicle center)
        
        # Vector from current_pose to lookahead_point
        vec_x = lookahead_point.x - current_x
        vec_y = lookahead_point.y - current_y

        # Angle of this vector relative to the global X-axis
        lookahead_angle = np.arctan2(vec_y, vec_x)
        
        # Alpha is the difference between the lookahead vector's angle and vehicle's yaw
        alpha = lookahead_angle - current_yaw
        
        # Normalize alpha to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        # 5. Calculate the steering angle using Pure Pursuit formula
        # tan(delta) = (2 * L * sin(alpha)) / Ld
        # delta = atan2(2 * L * sin(alpha), Ld) -> atan2 is safer for division by zero Ld or angle issues
        
        # Handle division by zero for Ld if it's too small, or alpha close to pi/2 or -pi/2
        # Use np.arctan2 for safer angle calculation
        
        # Steering angle calculation
        # Note: The formula is often derived for the rear axle. If your current_pose is center,
        # it might need slight adjustment, but for FSDS it's usually good enough.
        
        # Ensure Ld is not zero to prevent division by zero
        if LOOKAHEAD_DISTANCE == 0:
            steering_angle = 0.0
            rospy.logwarn("LOOKAHEAD_DISTANCE is zero. Steering angle set to 0.")
        else:
            steering_angle = np.arctan2(2 * WHEELBASE_LENGTH * np.sin(alpha), LOOKAHEAD_DISTANCE)

        # 6. Publish AckermannDriveStamped command
        self.publish_ackermann_cmd(TARGET_SPEED_MPS, steering_angle)

    def publish_ackermann_cmd(self, speed, steering_angle):
        cmd = AckermannDriveStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.drive.speed = speed
        cmd.drive.steering_angle = steering_angle
        ackermann_pub.publish(cmd)
        # rospy.loginfo(f"Published Speed: {speed:.2f} m/s, Steering: {np.degrees(steering_angle):.2f} deg")

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
    except rospy.ROSInterruptException:
        pass