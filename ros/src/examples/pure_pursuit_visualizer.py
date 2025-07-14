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
TARGET_SPEED = 7.5            # 목표 속도 (m/s)
TARGET_THROTTLE = 0.8         # 기본 스로틀 값 (0.0 ~ 1.0)
MAX_STEER = 0.5               # 최대 조향각 (1.0 = 45도)

# Pure Pursuit 관련
LOOKAHEAD_BASE = 4.0          # 기본 전방 주시 거리 (m)
LOOKAHEAD_SPEED_RATIO = 0.6   # 속도에 비례한 추가 주시 거리 (값이 클수록 고속에서 안정적)
VEHICLE_LENGTH = 1.5         # 차량 축거 (m)

# 경로 생성 관련
INTERPOLATION_FACTOR = 10     # 웨이포인트 보간 배수 (1 = 보간 안함, 10 = 10배로 늘림)

# ==============================================================================
# ==============================================================================


class PurePursuitVisualizer:
    def __init__(self):
        rospy.init_node('pure_pursuit_visualizer')

        # === 파라미터 설정 ===
        self.target_speed = TARGET_SPEED
        self.vehicle_length = VEHICLE_LENGTH
        self.lookahead_base = LOOKAHEAD_BASE
        self.lookahead_speed_ratio = LOOKAHEAD_SPEED_RATIO

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
        self.lookahead_plot, = self.ax.plot([], [], 'mo', markersize=8, label="Lookahead Point")
        
        self.ax.set_title("FSDS Pure Pursuit Visualization")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=100, blit=True)

        # === ROS 통신 ===
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.path_callback, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.loginfo("Pure Pursuit Visualizer 노드 시작됨.")

    def init_plot(self):
        return self.history_plot, self.car_plot, self.lookahead_plot

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

        # --- 웨이포인트 보간하여 부드러운 경로 생성 ---
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
        
        # 속도 값을 Odometry 메시지에서 직접 가져옴
        self.current_speed = math.sqrt(odom_msg.twist.twist.linear.x**2 + odom_msg.twist.twist.linear.y**2)

        orientation_q = odom_msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        steering_angle, target_idx = self.calculate_steering_angle()
        if target_idx is not None:
             self.lookahead_plot.set_data([self.path[target_idx, 0]], [self.path[target_idx, 1]])
        
        # 간단한 속도 제어 (P-제어)
        throttle = TARGET_THROTTLE + 0.5 * (self.target_speed - self.current_speed)

        control_cmd = ControlCommand()
        control_cmd.header.stamp = rospy.Time.now()
        control_cmd.throttle = np.clip(throttle, 0.0, 1.0) # 0~1 사이로 제한
        control_cmd.steering = np.clip(steering_angle, -MAX_STEER, MAX_STEER) # -MAX_STEER ~ MAX_STEER 사이로 제한
        control_cmd.brake = 0.0
        
        # --- 디버깅을 위한 로그 출력 추가 ---
        rospy.loginfo(f"Odom: X={self.current_x:.2f}, Y={self.current_y:.2f}, Yaw={math.degrees(self.current_yaw):.1f} deg, Speed={self.current_speed:.2f} m/s")
        lookahead_dist = self.lookahead_base + self.lookahead_speed_ratio * self.current_speed
        rospy.loginfo(f"Speed: {self.current_speed:.2f} m/s, Steer: {steering_angle:.2f}, Lookahead: {lookahead_dist:.2f} m")
        # ------------------------------------

        self.control_pub.publish(control_cmd)

    def calculate_steering_angle(self):
        lookahead_dist = self.lookahead_base + self.lookahead_speed_ratio * self.current_speed

        dx = self.path[:, 0] - self.current_x
        dy = self.path[:, 1] - self.current_y
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)

        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            if distances[i] >= lookahead_dist:
                target_idx = i
                break # 목표 지점 찾으면 루프 종료
        else: # 루프가 끝까지 돌았다면 경로의 마지막 점을 목표로 함
             target_idx = len(self.path) - 1

        target_x, target_y = self.path[target_idx]
        
        relative_y = (target_y - self.current_y) * math.cos(self.current_yaw) - (target_x - self.current_x) * math.sin(self.current_yaw)
        delta = math.atan2(2.0 * self.vehicle_length * relative_y, lookahead_dist**2)
        
        return -delta, target_idx

    def update_plot(self, frame):
        self.car_plot.set_data([self.current_x], [self.current_y])
        self.history_plot.set_data(self.history_x, self.history_y)
        return self.history_plot, self.car_plot, self.lookahead_plot

if __name__ == '__main__':
    try:
        controller = PurePursuitVisualizer()
        plt.show()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass