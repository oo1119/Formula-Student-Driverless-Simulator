#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 필요한 라이브러리들을 가져옵니다.
import matplotlib
matplotlib.use('TkAgg') # ROS와 Matplotlib 충돌을 방지하기 위한 백엔드 설정
import rospy
import numpy as np
import math
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt

# Formula Student 시뮬레이터와 ROS 통신을 위한 메시지 타입을 가져옵니다.
from fs_msgs.msg import ControlCommand, Track, Cone
from nav_msgs.msg import Odometry
from scipy.interpolate import splprep, splev # 경로를 부드럽게 만들기 위한 보간 함수

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================
# 이 값들을 조절하여 차량의 주행 특성을 변경할 수 있습니다.

# --- 속도 제어 관련 ---
MIN_SPEED = 3.0                 # 코너링 시 최저 목표 속도 (m/s)
MAX_SPEED = 6.0                 # 직선 주로 최고 목표 속도 (m/s)
KP, KI, KD = 0.6, 0.1, 0.7      # PID 제어기 게인 (Kp: 반응성, Ki: 정상상태 오차, Kd: 안정성)

# --- 지능형 속도 제어 관련 ---
STEERING_SPEED_DAMPING = 0.8    # 조향각이 클수록 속도를 얼마나 줄일지에 대한 강도 (0.0 ~ 1.0)
CURVATURE_LOOKAHEAD = 150       # 미래 경로의 곡률을 몇 포인트 앞까지 볼 것인가
CURVATURE_GAIN = 6.0            # 곡률이 속도에 미치는 영향력 (클수록 코너에서 더 많이 감속)

# --- Pure Pursuit 조향 제어 관련 ---
LOOKAHEAD_DISTANCE = 3.0        # 전방 주시 거리 (m), 차량이 바라볼 앞의 목표점 거리
STEERING_GAIN = 0.9             # 계산된 조향각에 대한 전체적인 민감도
VEHICLE_LENGTH = 1.55           # 차량의 축거 (m), 바퀴 사이의 거리
MAX_STEER_ANGLE = 0.5          # 최대 조향각 (rad), 시뮬레이터 특성상 0.5보다 약간 작게 설정
LOCAL_PATH_WINDOW = 200         # 경로 탐색 시 '터널 시야'의 크기 (포인트 개수)
INTERPOLATION_FACTOR = 20       # 경로점 보간 배수 (클수록 경로가 부드러워짐)
# ==============================================================================

class PIDController:
    """속도 제어를 위한 PID 제어기 클래스"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral, self.prev_error = 0.0, 0.0

    def calculate(self, target_speed, current_speed, dt):
        # 목표 속도와 현재 속도의 차이(오차)를 기반으로 제어량 계산
        if dt <= 0: return 0.0
        error = target_speed - current_speed
        self.integral += error * dt # 오차의 합 (과거의 오차)
        self.integral = np.clip(self.integral, -2.0, 2.0) # 적분항이 너무 커지는 것을 방지
        derivative = (error - self.prev_error) / dt # 오차의 변화율 (미래의 오차 예측)
        self.prev_error = error
        
        # 비례(현재), 적분(과거), 미분(미래) 오차를 모두 더해 최종 제어량 반환
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class PerformanceController:
    """Pure Pursuit 경로 추종과 PID 속도 제어를 수행하는 메인 클래스"""
    def __init__(self):
        rospy.init_node("performance_controller")
        
        self.pid_controller = PIDController(KP, KI, KD) # PID 제어기 인스턴스 생성
        self.path, self.path_curvatures = None, None
        self.current_pose, self.current_speed, self.is_path_generated = np.zeros(3), 0.0, False
        self.last_closest_idx = 0
        self.last_time = None
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10)) # 시각화 그래프 설정 (10x10 크기)
        self.setup_plot()
        
        self.history_x, self.history_y, self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [], [], [], [], [], []
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1) # 제어 명령을 보내는 퍼블리셔
        rospy.Subscriber('/fsds/testing_only/track', Track, self.track_callback, queue_size=1) # 경로 정보를 받는 서브스크라이버
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback, queue_size=1) # 차량의 위치와 속도를 받는 서브스크라이버
        
        rospy.loginfo("Performance Controller (Pure Pursuit) 노드 시작됨.")

    def setup_plot(self):
        """시각화 그래프의 초기 설정을 담당하는 함수"""
        self.ax.set_title("Pure Pursuit Controller")
        self.ax.set_xlabel("X (m)"), self.ax.set_ylabel("Y (m)")
        self.path_plot, = self.ax.plot([], [], 'g--', label="Interpolated Path")
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=4, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'o', color='gold', markersize=4, label="Yellow Cones")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ko', markersize=6, label="Car")
        self.lookahead_plot, = self.ax.plot([], [], 'mx', markersize=8, mew=2, label="Lookahead Point")
        self.ax.grid(True), self.ax.legend(), self.ax.set_aspect('equal', adjustable='box')

    def track_callback(self, track_msg: Track):
        """'/fsds/testing_only/track' 토픽을 구독하여 경로를 생성하는 콜백 함수 (한 번만 실행)"""
        if self.is_path_generated: return
        rospy.loginfo("트랙 정보를 수신하여 경로 생성 시작...")
        
        # 1. 파란색, 노란색 콘 분리
        blue_cones = [(c.location.x, c.location.y) for c in track_msg.track if c.color == Cone.BLUE]
        yellow_cones = [(c.location.x, c.location.y) for c in track_msg.track if c.color == Cone.YELLOW]
        if not blue_cones or not yellow_cones: return
        
        self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [c[0] for c in blue_cones], [c[1] for c in blue_cones], [c[0] for c in yellow_cones], [c[1] for c in yellow_cones]
        
        # 2. 콘 쌍을 찾아 중앙 경로(waypoints) 생성
        raw_path, current_blue = [], min(blue_cones, key=lambda p: np.hypot(p[0], p[1]))
        while blue_cones:
            blue_cones.remove(current_blue)
            if not yellow_cones: break
            matching_yellow = min(yellow_cones, key=lambda p: np.hypot(p[0]-current_blue[0], p[1]-current_blue[1]))
            raw_path.append(((current_blue[0] + matching_yellow[0]) / 2.0, (current_blue[1] + matching_yellow[1]) / 2.0))
            if not blue_cones: break
            current_blue = min(blue_cones, key=lambda p: np.hypot(p[0]-current_blue[0], p[1]-current_blue[1]))
        raw_path = np.array(raw_path)

        # 3. 경로의 진행 방향을 반시계 방향으로 통일 (Shoelace formula 사용)
        signed_area = 0.0
        for i in range(len(raw_path)):
            x1, y1 = raw_path[i]; x2, y2 = raw_path[(i + 1) % len(raw_path)]
            signed_area += (x1 * y2 - x2 * y1)
        if signed_area < 0: raw_path = raw_path[::-1]
        
        if len(raw_path) > 3:
            # 4. 중앙 경로점들을 부드러운 곡선(spline)으로 보정
            tck, u = splprep([raw_path[:, 0], raw_path[:, 1]], s=1.0, k=3, per=True)
            u_new = np.linspace(u.min(), u.max(), len(raw_path) * INTERPOLATION_FACTOR)
            x_new, y_new = splev(u_new, tck, der=0)
            self.path = np.c_[x_new, y_new]
            self.calculate_curvatures() # 경로의 곡률 미리 계산
            self.is_path_generated = True
            
            # 5. 그래프의 x, y축 범위를 전체 트랙에 맞게 고정
            margin = 10
            self.ax.set_xlim(self.path[:, 0].min() - margin, self.path[:, 0].max() + margin)
            self.ax.set_ylim(self.path[:, 1].min() - margin, self.path[:, 1].max() + margin)
            rospy.loginfo("경로 생성 완료!")

    def calculate_curvatures(self):
        """생성된 경로의 모든 점에 대한 곡률을 미리 계산하는 함수"""
        dx, dy = np.gradient(self.path[:, 0]), np.gradient(self.path[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        self.path_curvatures = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

    def odom_callback(self, odom_msg: Odometry):
        """'/fsds/testing_only/odom' 토픽을 구독하여 차량의 현재 상태를 업데이트하는 콜백 함수"""
        if not self.is_path_generated: return
        
        # 1. 차량의 실제 속도(Speed) 계산 (전진/측면 속도 모두 고려)
        linear_vel = odom_msg.twist.twist.linear
        self.current_speed = math.sqrt(linear_vel.x**2 + linear_vel.y**2)
        
        # 2. 차량의 위치(x, y) 및 자세(yaw) 업데이트
        q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
        
        # 3. 주행 궤적 기록
        self.history_x.append(self.current_pose[0]), self.history_y.append(self.current_pose[1])
        
        # 4. 제어 루프 실행 시간(dt) 계산 및 제어 함수 호출
        current_time = odom_msg.header.stamp.to_sec()
        dt = (current_time - self.last_time) if self.last_time else 0.02
        self.last_time = current_time
        self.control_loop(dt)

    def control_loop(self, dt):
        """차량의 조향과 속도를 계산하고 명령을 발행하는 메인 제어 루프"""
        # --- 조향 제어 (Pure Pursuit) ---
        
        # 1. '운전자 시점'인 앞바퀴 축 위치 계산
        front_axle_pos = self.current_pose[:2] + VEHICLE_LENGTH * np.array([math.cos(self.current_pose[2]), math.sin(self.current_pose[2])])
        
        # 2. '터널 시야' 로직: 현재 위치 주변의 경로만 보도록 탐색 범위 제한
        path_len = len(self.path)
        search_start_idx = self.last_closest_idx
        search_end_idx = (self.last_closest_idx + LOCAL_PATH_WINDOW) % path_len
        if search_start_idx < search_end_idx:
            local_path_indices = np.arange(search_start_idx, search_end_idx)
        else:
            local_path_indices = np.concatenate([np.arange(search_start_idx, path_len), np.arange(0, search_end_idx)])
        local_path = self.path[local_path_indices]
        
        # 3. 제한된 범위 내에서 가장 가까운 경로점(closest_idx) 찾기
        distances = np.linalg.norm(local_path - front_axle_pos, axis=1)
        local_closest_idx = np.argmin(distances)
        closest_idx = local_path_indices[local_closest_idx]
        self.last_closest_idx = closest_idx # 다음 계산을 위해 현재 위치 기억

        # 4. 전방 주시점(Lookahead Point) 찾기
        target_idx = closest_idx
        for i in range(len(local_path_indices)):
            check_idx = local_path_indices[(local_closest_idx + i) % len(local_path_indices)]
            dist_from_car = np.linalg.norm(self.path[check_idx] - front_axle_pos)
            if dist_from_car >= LOOKAHEAD_DISTANCE:
                target_idx = check_idx
                break
        if target_idx == closest_idx: target_idx = (closest_idx + 10) % path_len
        target_point = self.path[target_idx]
        
        # 5. Pure Pursuit 공식으로 조향각(delta) 계산
        alpha = math.atan2(target_point[1] - front_axle_pos[1], target_point[0] - front_axle_pos[0]) - self.current_pose[2]
        delta = math.atan2(2.0 * VEHICLE_LENGTH * math.sin(alpha), LOOKAHEAD_DISTANCE)
        steering_angle = np.clip(STEERING_GAIN * delta, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        # --- 속도 제어 (PID) ---
        
        # 1. 지능형 목표 속도 계산 (경로 곡률 + 조향각 연동)
        future_indices = np.arange(closest_idx, closest_idx + CURVATURE_LOOKAHEAD) % len(self.path)
        max_future_curvature = np.max(self.path_curvatures[future_indices])
        target_speed_by_curve = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * min(max_future_curvature * CURVATURE_GAIN, 1.0)
        steering_damping_factor = 1.0 - STEERING_SPEED_DAMPING * abs(steering_angle / MAX_STEER_ANGLE)
        final_target_speed = target_speed_by_curve * steering_damping_factor
        
        # 2. PID 제어기로 스로틀/브레이크 값 계산
        pid_output = self.pid_controller.calculate(final_target_speed, self.current_speed, dt)
        
        if pid_output > 0:
            throttle, brake = np.clip(pid_output, 0.0, 0.4), 0.0 # 스로틀 상한을 0.4로 제한하여 안정성 확보 (최대 1.0)
        else:
            throttle, brake = 0.0, np.clip(abs(pid_output), 0.0, 1.0) # 브레이크 상한은 1.0으로 제한 (더 낮은 값으로 제한하여 부드러운 브레이킹 가능)
        
        # --- 최종 명령 발행 ---
        cmd_msg = ControlCommand(throttle=throttle, steering=-steering_angle, brake=brake)
        self.control_pub.publish(cmd_msg)
        self.update_plot_data(target_point)


    def update_plot_data(self, target_point):
        """시각화 그래프의 데이터를 실시간으로 업데이트하는 함수"""
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        self.blue_cone_plot.set_data(self.blue_cones_x, self.blue_cones_y)
        self.yellow_cone_plot.set_data(self.yellow_cones_x, self.yellow_cones_y)
        self.history_plot.set_data(self.history_x, self.history_y)
        self.car_plot.set_data([self.current_pose[0]], [self.current_pose[1]])
        self.lookahead_plot.set_data([target_point[0]], [target_point[1]])

if __name__ == '__main__':
    try:
        controller = PerformanceController()
        plt.show(block=False)
        # ROS가 종료되지 않는 한, 루프를 돌며 그래프를 계속 업데이트
        while not rospy.is_shutdown():
            plt.pause(0.01)
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 종료되었습니다.")