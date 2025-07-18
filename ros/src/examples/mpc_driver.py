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
import casadi as ca

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================
MIN_SPEED = 2.0
MAX_SPEED = 7.0
MAX_STEER_ANGLE = 0.5
MAX_ACCEL = 0.4

# 안정성과 부드러움을 극대화하기 위한 최종 튜닝
N = 10                      # 사용자가 찾은 최적의 예측 구간
DT = 0.1
CONTROL_RATE = 10

W_CTE = 60.0                # 경로 추종 정확도 (안정성을 위해 소폭 하향)
W_EPSI = 60.0               # 헤딩 오차 정확도 (안정성을 위해 소폭 하향)
W_V = 10.0                  # 속도 오차 가중치

# 부드러움을 위한 가중치를 극단적으로 상향
W_STEER = 150.0             # 조향각 사용량 가중치
W_ACCEL = 100.0             # 가속/감속 사용량 가중치
W_STEER_RATE = 400.0        # 조향각 '변화량' 가중치 (좌우 흔들림 억제)
W_ACCEL_RATE = 300.0        # 가속/감속 '변화량' 가중치 (울컥거림 억제)

# 차량 및 경로 파라미터
VEHICLE_LENGTH = 1.55
MAX_BRAKE = 1.0
LOCAL_PATH_WINDOW = 300
CURVATURE_LOOKAHEAD = 150
CURVATURE_GAIN = 5.0
INTERPOLATION_FACTOR = 20
# ==============================================================================

class UnifiedMPCController:
    def __init__(self):
        rospy.init_node("unified_mpc_controller")
        
        self.path, self.path_curvatures = None, None
        self.current_pose, self.current_speed, self.is_path_generated = np.zeros(3), 0.0, False
        self.last_closest_idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        self.history_x, self.history_y, self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [], [], [], [], [], []
        self.ref_path_plot, = self.ax.plot([], [], 'c.', markersize=4, label="MPC Reference")
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.track_callback, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)
        rospy.Timer(rospy.Duration(1.0 / CONTROL_RATE), self.control_loop)
        
        rospy.loginfo("Unified MPC Controller 노드 시작됨.")

    def setup_plot(self):
        self.ax.set_title("Unified MPC Controller")
        self.ax.set_xlabel("X (m)"), self.ax.set_ylabel("Y (m)")
        self.path_plot, = self.ax.plot([], [], 'g--', label="Interpolated Path")
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=4, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'o', color='gold', markersize=4, label="Yellow Cones")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ko', markersize=6, label="Car")
        self.ax.grid(True), self.ax.legend(), self.ax.set_aspect('equal', adjustable='box')

    def track_callback(self, track_msg: Track):
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
        dx, dy = np.gradient(self.path[:, 0]), np.gradient(self.path[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        self.path_curvatures = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

    def odom_callback(self, odom_msg: Odometry):
        linear_vel = odom_msg.twist.twist.linear
        self.current_speed = math.sqrt(linear_vel.x**2 + linear_vel.y**2)
        q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
        self.history_x.append(self.current_pose[0]), self.history_y.append(self.current_pose[1])

    def control_loop(self, event):
        if not self.is_path_generated: return
        solution = self.solve_mpc()
        if solution is None:
            steering_angle, throttle, brake = 0.0, 0.0, 1.0
        else:
            steering_angle = solution['steer']
            acceleration = solution['accel']
            if acceleration > 0:
                throttle, brake = np.clip(acceleration, 0.0, MAX_ACCEL), 0.0
            else:
                throttle, brake = 0.0, np.clip(abs(acceleration), 0.0, MAX_BRAKE)
        cmd_msg = ControlCommand(throttle=throttle, steering=-steering_angle, brake=brake)
        self.control_pub.publish(cmd_msg)

    def solve_mpc(self):
        opti = ca.Opti()
        X = opti.variable(4, N + 1)
        U = opti.variable(2, N)
        cost = 0
        x0 = np.append(self.current_pose, self.current_speed)
        opti.subject_to(X[:, 0] == x0)

        path_len = len(self.path)
        front_axle_pos = self.current_pose[:2] + VEHICLE_LENGTH * np.array([math.cos(self.current_pose[2]), math.sin(self.current_pose[2])])
        search_start_idx, search_end_idx = self.last_closest_idx, (self.last_closest_idx + LOCAL_PATH_WINDOW) % path_len
        if search_start_idx < search_end_idx: local_path_indices = np.arange(search_start_idx, search_end_idx)
        else: local_path_indices = np.concatenate([np.arange(search_start_idx, path_len), np.arange(0, search_end_idx)])
        local_path = self.path[local_path_indices]
        distances = np.linalg.norm(local_path - front_axle_pos, axis=1)
        local_closest_idx = np.argmin(distances)
        closest_idx = local_path_indices[local_closest_idx]
        self.last_closest_idx = closest_idx
        
        ref_path_indices = np.arange(closest_idx, closest_idx + N + 1) % path_len
        ref_path = self.path[ref_path_indices]
        self.ref_path_plot.set_data(ref_path[:, 0], ref_path[:, 1])

        for k in range(N):
            x_k, y_k, yaw_k, v_k = X[0, k], X[1, k], X[2, k], X[3, k]
            accel_k, steer_k = U[0, k], U[1, k]
            
            x_next = x_k + v_k * ca.cos(yaw_k) * DT
            y_next = y_k + v_k * ca.sin(yaw_k) * DT
            yaw_next = yaw_k + v_k / VEHICLE_LENGTH * ca.tan(steer_k) * DT
            v_next = v_k + accel_k * DT
            
            opti.subject_to(X[:, k+1] == ca.vertcat(x_next, y_next, yaw_next, v_next))
            
            ref_x, ref_y = ref_path[k, 0], ref_path[k, 1]
            ref_yaw = math.atan2(ref_path[k+1, 1] - ref_y, ref_path[k+1, 0] - ref_x)
            max_future_curvature = np.max(self.path_curvatures[np.arange(closest_idx, closest_idx + CURVATURE_LOOKAHEAD) % path_len])
            ref_v = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * min(max_future_curvature * CURVATURE_GAIN, 1.0)

            cte = (x_k - ref_x) * ca.sin(ref_yaw) - (y_k - ref_y) * ca.cos(ref_yaw)
            
            # [핵심 수정] CasADi가 처리할 수 있도록 삼각함수를 이용하여 각도 차이 계산
            epsi = ca.sin(yaw_k - ref_yaw)

            cost += W_CTE * cte**2
            cost += W_EPSI * epsi**2
            cost += W_V * (v_k - ref_v)**2
            cost += W_STEER * steer_k**2
            cost += W_ACCEL * accel_k**2
            
            if k > 0:
                cost += W_STEER_RATE * (steer_k - U[1, k-1])**2
                cost += W_ACCEL_RATE * (accel_k - U[0, k-1])**2

        opti.subject_to(opti.bounded(-MAX_STEER_ANGLE, U[1, :], MAX_STEER_ANGLE))
        opti.subject_to(opti.bounded(-MAX_BRAKE, U[0, :], MAX_ACCEL))

        opti.minimize(cost)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            return {'steer': sol.value(U[1, 0]), 'accel': sol.value(U[0, 0])}
        except:
            rospy.logwarn("MPC 솔루션 탐색 실패.")
            return None

    def update_plot_data(self):
        # [핵심 수정] 경로가 생성되기 전에는 플로팅을 시도하지 않도록 수정
        if not self.is_path_generated:
            return
            
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        self.blue_cone_plot.set_data(self.blue_cones_x, self.blue_cones_y)
        self.yellow_cone_plot.set_data(self.yellow_cones_x, self.yellow_cones_y)
        self.history_plot.set_data(self.history_x, self.history_y)
        self.car_plot.set_data([self.current_pose[0]], [self.current_pose[1]])

if __name__ == '__main__':
    try:
        controller = UnifiedMPCController()
        plt.show(block=False)
        while not rospy.is_shutdown():
            controller.update_plot_data()
            plt.pause(0.01)
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 종료되었습니다.")