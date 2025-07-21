#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import rospy
import numpy as np
import math
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
from fs_msgs.msg import ControlCommand, Track, Cone
from nav_msgs.msg import Odometry
from scipy.interpolate import splprep, splev

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================
# --- 속도 제어 관련 (PID) ---
MIN_SPEED = 3.0              # 차량이 따라가야 할 최소 속도 (m/s)
MAX_SPEED = 6.0              # 차량이 따라가야 할 최대 속도 (m/s)
KP, KI, KD = 0.7, 0.2, 0.5   # 부드러운 제어를 위한 PID 게인

# --- Stanley 조향 제어 관련 ---
K_CTE = 1.2                 # 횡방향 오차(Crosstrack Error) 보정 강도 (핵심 튜닝값)
K_SOFT = 1.0                # 저속 주행 안정화 상수 (클수록 저속에서 부드러워짐)
STEERING_GAIN = 1.0         # 계산된 조향각에 대한 전체적인 민감도

# --- 기타 파라미터 ---
CURVATURE_GAIN = 5.0        # 경로 곡률에 따른 속도 감소 비율 (클수록 곡률이 큰 경로에서 속도를 더 줄임)
VEHICLE_LENGTH = 1.55       # 차량의 앞바퀴와 뒷바퀴 사이의 거리 (미터)
MAX_STEER_ANGLE = 0.45      # 최대 조향각 (라디안), 차량이 회전할 수 있는 최대 각도
LOCAL_PATH_WINDOW = 300     # 로컬 경로 윈도우 크기 (MPC가 참조할 경로의 길이)
INTERPOLATION_FACTOR = 20   # 경로 보간 시 점의 개수 (더 부드러운 곡선을 위해)
# ==============================================================================

class PIDController:
    """속도 제어를 위한 PID 제어기 클래스"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral, self.prev_error = 0.0, 0.0
    def calculate(self, target_speed, current_speed, dt):
        if dt <= 0: return 0.0
        error = target_speed - current_speed
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class StanleyPIDController:
    """Stanley 경로 추종과 PID 속도 제어를 수행하는 메인 클래스"""
    def __init__(self):
        rospy.init_node("stanley_pid_controller")
        
        self.pid_controller = PIDController(KP, KI, KD)
        self.path, self.path_curvatures = None, None
        self.current_pose, self.current_speed, self.is_path_generated = np.zeros(3), 0.0, False
        self.last_closest_idx = 0
        self.last_time = None
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        self.history_x, self.history_y, self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [], [], [], [], [], []
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.track_callback, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback, queue_size=1)
        
        rospy.loginfo("Stanley Controller with PID Speed Control 노드 시작됨.")

    def setup_plot(self):
        """시각화 그래프의 초기 설정을 담당하는 함수"""
        self.ax.set_title("Stanley Controller (PID Speed)")
        self.ax.set_xlabel("X (m)"), self.ax.set_ylabel("Y (m)")
        self.path_plot, = self.ax.plot([], [], 'g--', label="Interpolated Path")
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=4, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'o', color='gold', markersize=4, label="Yellow Cones")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ko', markersize=6, label="Car")
        self.closest_point_plot, = self.ax.plot([], [], 'mx', markersize=8, mew=2, label="Closest Path Point")
        self.ax.grid(True), self.ax.legend(), self.ax.set_aspect('equal', adjustable='box')

    def track_callback(self, track_msg: Track):
        """경로 생성 함수 (Pure Pursuit과 동일)"""
        if self.is_path_generated: return
        rospy.loginfo("트랙 정보를 수신하여 경로 생성 시작...")
        blue_cones, yellow_cones = [(c.location.x, c.location.y) for c in track_msg.track if c.color == Cone.BLUE], [(c.location.x, c.location.y) for c in track_msg.track if c.color == Cone.YELLOW]
        if not blue_cones or not yellow_cones: return
        self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [c[0] for c in blue_cones], [c[1] for c in blue_cones], [c[0] for c in yellow_cones], [c[1] for c in yellow_cones]
        raw_path, current_blue = [], min(blue_cones, key=lambda p: np.hypot(p[0], p[1]))
        while blue_cones:
            blue_cones.remove(current_blue)
            if not yellow_cones: break
            matching_yellow = min(yellow_cones, key=lambda p: np.hypot(p[0]-current_blue[0], p[1]-current_blue[1]))
            raw_path.append(((current_blue[0] + matching_yellow[0]) / 2.0, (current_blue[1] + matching_yellow[1]) / 2.0))
            if not blue_cones: break
            current_blue = min(blue_cones, key=lambda p: np.hypot(p[0]-current_blue[0], p[1]-current_blue[1]))
        raw_path = np.array(raw_path)
        signed_area = 0.0
        for i in range(len(raw_path)):
            x1, y1 = raw_path[i]; x2, y2 = raw_path[(i + 1) % len(raw_path)]
            signed_area += (x1 * y2 - x2 * y1)
        if signed_area < 0: raw_path = raw_path[::-1]
        if len(raw_path) > 3:
            tck, u = splprep([raw_path[:, 0], raw_path[:, 1]], s=1.0, k=3, per=True)
            u_new = np.linspace(u.min(), u.max(), len(raw_path) * INTERPOLATION_FACTOR)
            x_new, y_new = splev(u_new, tck, der=0)
            self.path = np.c_[x_new, y_new]
            self.calculate_curvatures()
            self.is_path_generated = True
            margin = 10
            self.ax.set_xlim(self.path[:, 0].min() - margin, self.path[:, 0].max() + margin)
            self.ax.set_ylim(self.path[:, 1].min() - margin, self.path[:, 1].max() + margin)
            rospy.loginfo("경로 생성 완료!")

    def calculate_curvatures(self):
        """경로 곡률 계산 함수 (Pure Pursuit과 동일)"""
        dx, dy = np.gradient(self.path[:, 0]), np.gradient(self.path[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        self.path_curvatures = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

    def normalize_angle(self, angle):
        """각도를 -pi ~ +pi 범위로 정규화하는 함수"""
        while angle > np.pi: angle -= 2.0 * np.pi
        while angle < -np.pi: angle += 2.0 * np.pi
        return angle

    def odom_callback(self, odom_msg: Odometry):
        """차량 상태 업데이트 함수 (Pure Pursuit과 동일)"""
        if not self.is_path_generated: return
        linear_vel = odom_msg.twist.twist.linear
        self.current_speed = math.sqrt(linear_vel.x**2 + linear_vel.y**2)
        q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
        self.history_x.append(self.current_pose[0]), self.history_y.append(self.current_pose[1])
        
        current_time = odom_msg.header.stamp.to_sec()
        dt = (current_time - self.last_time) if self.last_time else 0.02
        self.last_time = current_time
        
        self.control_loop(dt)

    def control_loop(self, dt):
        """차량의 조향과 속도를 계산하고 명령을 발행하는 메인 제어 루프"""
        # --- 조향 제어 (Stanley) ---
        
        # 1. 앞바퀴 축 위치 계산 및 가장 가까운 경로점 찾기 ('터널 시야' 적용)
        front_axle_pos = self.current_pose[:2] + VEHICLE_LENGTH * np.array([math.cos(self.current_pose[2]), math.sin(self.current_pose[2])])
        path_len = len(self.path)
        search_start_idx, search_end_idx = self.last_closest_idx, (self.last_closest_idx + LOCAL_PATH_WINDOW) % path_len
        if search_start_idx < search_end_idx:
            local_path_indices = np.arange(search_start_idx, search_end_idx)
        else:
            local_path_indices = np.concatenate([np.arange(search_start_idx, path_len), np.arange(0, search_end_idx)])
        local_path = self.path[local_path_indices]
        distances = np.linalg.norm(local_path - front_axle_pos, axis=1)
        local_closest_idx = np.argmin(distances)
        closest_idx = local_path_indices[local_closest_idx]
        self.last_closest_idx = closest_idx

        # 2. 헤딩 오차 (Heading Error) 계산
        #   경로의 진행 방향과 차량의 진행 방향 사이의 각도 차이
        path_segment_start = self.path[closest_idx - 1] if closest_idx > 0 else self.path[closest_idx]
        path_segment_end = self.path[closest_idx]
        path_yaw = math.atan2(path_segment_end[1] - path_segment_start[1], path_segment_end[0] - path_segment_start[0])
        heading_error = self.normalize_angle(path_yaw - self.current_pose[2])
        
        # 3. 횡방향 오차 (Crosstrack Error) 계산
        #   차량 앞바퀴와 경로 중앙선 사이의 거리
        crosstrack_error = np.min(distances)
        #   벡터 외적을 이용해 차량이 경로의 왼쪽/오른쪽 어느 쪽에 있는지 판단하여 오차의 부호 결정
        cross_prod = (path_segment_end[0] - path_segment_start[0]) * (front_axle_pos[1] - path_segment_start[1]) - \
                     (path_segment_end[1] - path_segment_start[1]) * (front_axle_pos[0] - path_segment_start[0])
        if cross_prod > 0:
            crosstrack_error = -crosstrack_error

        # 4. Stanley 조향각 공식 적용
        #   조향각 = 헤딩 오차 + 횡방향 오차
        crosstrack_steer = math.atan(K_CTE * crosstrack_error / (K_SOFT + self.current_speed))
        steering_angle = np.clip(STEERING_GAIN * (heading_error + crosstrack_steer), -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        # --- 속도 제어 (PID) ---
        
        # 1. 동적 목표 속도 계산 (경로 곡률 기반)
        max_future_curvature = np.max(self.path_curvatures[np.arange(closest_idx, closest_idx + 150) % path_len])
        target_speed = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * min(max_future_curvature * CURVATURE_GAIN, 1.0)
        
        # 2. PID 제어기로 스로틀/브레이크 값 계산
        pid_output = self.pid_controller.calculate(target_speed, self.current_speed, dt)
        
        if pid_output > 0:
            throttle, brake = np.clip(pid_output, 0.0, 0.8), 0.0
        else:
            throttle, brake = 0.0, np.clip(abs(pid_output), 0.0, 1.0)
        
        # --- 최종 명령 발행 ---
        cmd_msg = ControlCommand(throttle=throttle, steering=-steering_angle, brake=brake)
        self.control_pub.publish(cmd_msg)
        self.update_plot_data(self.path[closest_idx])

    def update_plot_data(self, closest_point):
        """시각화 그래프 업데이트 함수"""
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        self.blue_cone_plot.set_data(self.blue_cones_x, self.blue_cones_y)
        self.yellow_cone_plot.set_data(self.yellow_cones_x, self.yellow_cones_y)
        self.history_plot.set_data(self.history_x, self.history_y)
        self.car_plot.set_data([self.current_pose[0]], [self.current_pose[1]])
        self.closest_point_plot.set_data([closest_point[0]], [closest_point[1]])

if __name__ == '__main__':
    try:
        controller = StanleyPIDController()
        plt.show(block=False)
        while not rospy.is_shutdown():
            plt.pause(0.01)
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 종료되었습니다.")