#!/usr/bin/env python3
import rospy
import math
import numpy as np
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev

# ROS 메시지 타입 임포트
from fs_msgs.msg import ControlCommand, Track, Cone
from nav_msgs.msg import Odometry

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================

# 주행 관련
TARGET_SPEED = 4.5            # 목표 속도 (m/s)
TARGET_THROTTLE = 0.8         # 기본 스로틀 값 (0.0 ~ 1.0)
MAX_STEER = 0.5               # 최대 조향각 (1.0 = 45도)

# Stanley 제어기 관련
STANLEY_GAIN = 1.2            # 제어 게인 (k). 가장 중요한 튜닝 값.
SOFTENING_CONSTANT = 1e-6     # 저속에서 분모가 0이 되는 것을 방지하기 위한 작은 값
VEHICLE_LENGTH = 1.5          # 차량 축거 (m)

# --- 오차 경계 설정 ---
MAX_CROSS_TRACK_ERROR = 0.5   # 경로 오차의 최대값 (m)
MAX_HEADING_ERROR = math.radians(90.0) # 방향 오차의 최대값 (rad)

# 경로 생성 관련
INTERPOLATION_FACTOR = 10     # 웨이포인트 보간 배수

# ==============================================================================
# ==============================================================================


class StanleyControllerVisualizer:
    def __init__(self):
        rospy.init_node('stanley_controller_visualizer')

        # === 파라미터 클래스 변수로 저장 ===
        self.target_speed = TARGET_SPEED
        self.target_throttle = TARGET_THROTTLE  # NameError 수정을 위해 추가
        self.k = STANLEY_GAIN
        self.vehicle_length = VEHICLE_LENGTH
        self.softening_constant = SOFTENING_CONSTANT

        # === 변수 초기화 ===
        self.path = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.history_x = []
        self.history_y = []

        # === 시각화 설정 ===
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.path_plot, = self.ax.plot([], [], 'g--', label="Interpolated Path")
        self.raw_path_plot, = self.ax.plot([], [], 'kx', markersize=3, label="Raw Waypoints")
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=5, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'yo', markersize=5, label="Yellow Cones")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ro', markersize=8, label="Car")
        self.front_axle_plot, = self.ax.plot([], [], 'go', markersize=6, label="Front Axle")
        
        self.ax.set_title("FSDS Stanley Controller Visualization")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=100, blit=True)

        # === ROS 통신 ===
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.path_callback, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.loginfo("Stanley Controller Visualizer 노드 시작됨.")

    def init_plot(self):
        return self.history_plot, self.car_plot, self.front_axle_plot

    def path_callback(self, track_msg):
        if self.path is not None: return
        rospy.loginfo("트랙 데이터를 수신하여 경로를 생성합니다...")
        blue_cones = [c for c in track_msg.track if c.color == Cone.BLUE]
        yellow_cones = [c for c in track_msg.track if c.color == Cone.YELLOW]
        if len(blue_cones) < 2 or len(yellow_cones) < 2: return
        self.blue_cone_plot.set_data([c.location.x for c in blue_cones], [c.location.y for c in blue_cones])
        self.yellow_cone_plot.set_data([c.location.x for c in yellow_cones], [c.location.y for c in yellow_cones])
        path_points = []
        for i in range(min(len(blue_cones), len(yellow_cones))):
            mid_x = (blue_cones[i].location.x + yellow_cones[i].location.x) / 2.0
            mid_y = (blue_cones[i].location.y + yellow_cones[i].location.y) / 2.0
            path_points.append([mid_x, mid_y])
        raw_path = np.array(path_points)
        self.raw_path_plot.set_data(raw_path[:, 0], raw_path[:, 1])
        if len(raw_path) > 3 and INTERPOLATION_FACTOR > 1:
            tck, u = splprep([raw_path[:, 0], raw_path[:, 1]], s=0, per=True)
            u_new = np.linspace(u.min(), u.max(), len(raw_path) * INTERPOLATION_FACTOR)
            x_new, y_new = splev(u_new, tck, der=0)
            self.path = np.c_[x_new, y_new]
        else:
            self.path = raw_path
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        x_margin, y_margin = 10, 10
        self.ax.set_xlim(self.path[:, 0].min() - x_margin, self.path[:, 0].max() + x_margin)
        self.ax.set_ylim(self.path[:, 1].min() - y_margin, self.path[:, 1].max() + y_margin)
        rospy.loginfo(f"총 {len(self.path)}개의 웨이포인트 경로 생성 완료.")

    def odom_callback(self, odom_msg):
        if self.path is None:
            rospy.logwarn_throttle(1.0, "경로 데이터를 기다리는 중...")
            return

        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.history_x.append(self.current_x)
        self.history_y.append(self.current_y)
        
        self.current_speed = math.sqrt(odom_msg.twist.twist.linear.x**2 + odom_msg.twist.twist.linear.y**2)

        orientation_q = odom_msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        steering_angle = self.stanley_control()
        
        # 간단한 속도 제어 (P-제어)
        throttle = self.target_throttle + 0.5 * (self.target_speed - self.current_speed) # NameError 수정

        control_cmd = ControlCommand()
        control_cmd.header.stamp = rospy.Time.now()
        control_cmd.throttle = np.clip(throttle, 0.0, 1.0)
        control_cmd.steering = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        control_cmd.brake = 0.0
        
        self.control_pub.publish(control_cmd)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stanley_control(self):
        front_axle_x = self.current_x + self.vehicle_length * math.cos(self.current_yaw)
        front_axle_y = self.current_y + self.vehicle_length * math.sin(self.current_yaw)
        self.front_axle_plot.set_data([front_axle_x], [front_axle_y])

        dx = self.path[:, 0] - front_axle_x
        dy = self.path[:, 1] - front_axle_y
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)
        
        path_yaw_idx = max(closest_idx - 1, 0)
        path_yaw = math.atan2(self.path[closest_idx, 1] - self.path[path_yaw_idx, 1], 
                              self.path[closest_idx, 0] - self.path[path_yaw_idx, 0])
        
        path_dx = math.cos(path_yaw)
        path_dy = math.sin(path_yaw)
        error_vec_x = front_axle_x - self.path[closest_idx, 0]
        error_vec_y = front_axle_y - self.path[closest_idx, 1]
        
        cross_track_error = path_dx * error_vec_y - path_dy * error_vec_x

        heading_error = self.normalize_angle(path_yaw - self.current_yaw)
        
        limited_cross_track_error = np.clip(cross_track_error, -MAX_CROSS_TRACK_ERROR, MAX_CROSS_TRACK_ERROR)
        limited_heading_error = np.clip(heading_error, -MAX_HEADING_ERROR, MAX_HEADING_ERROR)
        
        crosstrack_steering = math.atan2(self.k * limited_cross_track_error, self.current_speed + self.softening_constant)
        
        delta = limited_heading_error + crosstrack_steering

        rospy.loginfo(f"CTE: {cross_track_error:.2f}({limited_cross_track_error:.2f}), HE: {math.degrees(heading_error):.1f}({math.degrees(limited_heading_error):.1f}), Steer: {math.degrees(delta):.1f}deg")

        return -delta

    def update_plot(self, frame):
        self.car_plot.set_data([self.current_x], [self.current_y])
        self.history_plot.set_data(self.history_x, self.history_y)
        return self.history_plot, self.car_plot, self.front_axle_plot

if __name__ == '__main__':
    try:
        controller = StanleyControllerVisualizer()
        plt.show()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass