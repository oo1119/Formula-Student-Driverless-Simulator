#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import rospy
import numpy as np
import math
from fs_msgs.msg import ControlCommand, Cone
from sensor_msgs.msg import PointCloud2, NavSatFix
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Empty
from sklearn.cluster import DBSCAN, KMeans
from collections import deque
from scipy.interpolate import CubicSpline
from tf.transformations import quaternion_matrix # 누락된 import 추가

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================
# 제어 관련
CONTROL_RATE = 50.0; MAX_STEER = 0.5; MAX_THROTTLE = 0.4
THROTTLE_PID_GAINS = {'Kp': 0.7, 'Ki': 0.1, 'Kd': 0.4}

# 지능형 속도 제어 파라미터
MAX_SPEED_ON_STRAIGHT = 4.0; MIN_SPEED_IN_CORNER = 2.0

# 퓨어 퍼슛 가변 룩어헤드 파라미터
VEHICLE_WHEELBASE = 1.55; LOOKAHEAD_BASE = 2.5; LOOKAHEAD_SPEED_RATIO = 0.7

# 인식 및 필터링 관련
LIDAR_RANGE_CUTOFF = 25.0; SPEED_FILTER_ALPHA = 0.2
LAT_TO_METER = 111132.0; LON_TO_METER = 111132.0
PATH_HISTORY_MAX_LEN = 30; CURVATURE_SENSITIVITY = 0.3

# 강제 출발 파라미터
CREEP_THROTTLE = 0.3; CREEP_SPEED_THRESHOLD = 0.5
# ==============================================================================

class PIDController:
    def __init__(self, gains, initial_target):
        self.gains = gains; self.target = initial_target; self.integral = 0; self.prev_error = 0; self.prev_value = 0.0; self.last_time = rospy.Time.now()
    def calculate(self, current_value):
        now, dt = rospy.Time.now(), (rospy.Time.now() - self.last_time).to_sec()
        if dt <= 0: return 0.0
        error = self.target - current_value; self.integral = np.clip(self.integral + error * dt, -1.0, 1.0); derivative = -(current_value - self.prev_value) / dt
        output = (self.gains['Kp']*error + self.gains['Ki']*self.integral + self.gains['Kd']*derivative)
        self.prev_error, self.last_time, self.prev_value = error, now, current_value
        return output

class FinalController:
    def __init__(self):
        rospy.init_node('final_controller')
        self.current_speed = 0.0; self.detected_cones_local = []; self.filtered_speed = 0.0
        self.prev_gps_pos = None; self.last_gps_time = None; self.last_valid_steering = 0.0
        self.startup_mode_active = True
        self.path_history = deque(maxlen=PATH_HISTORY_MAX_LEN)
        
        # --- [핵심 수정] self.vehicle_pose 초기화 누락 수정 ---
        self.vehicle_pose = np.eye(4)
        # ---------------------------------------------------

        self.pid_controller = PIDController(THROTTLE_PID_GAINS, MIN_SPEED_IN_CORNER)
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/gps', NavSatFix, self.gps_callback)
        rospy.Subscriber('/fsds/lidar/Lidar1', PointCloud2, self.lidar_callback)
        rospy.loginfo("Final Controller 노드 시작됨.")
        self.start_driving()

    def gps_callback(self, msg: NavSatFix):
        current_time = rospy.Time.now(); raw_speed = 0.0
        if self.prev_gps_pos is not None and self.last_gps_time is not None:
            dt = (current_time - self.last_gps_time).to_sec()
            if dt > 0.01:
                dx = (msg.latitude - self.prev_gps_pos.latitude) * LAT_TO_METER; dy = (msg.longitude - self.prev_gps_pos.longitude) * LON_TO_METER
                distance = math.sqrt(dx**2 + dy**2); raw_speed = distance / dt
        self.filtered_speed = SPEED_FILTER_ALPHA * raw_speed + (1 - SPEED_FILTER_ALPHA) * self.filtered_speed
        self.current_speed = self.filtered_speed
        self.prev_gps_pos, self.last_gps_time = msg, current_time
        
        # Odometry 대신 GPS와 IMU(가정)로 pose를 업데이트해야 하지만,
        # 현재 IMU 방향 데이터가 없으므로 GPS 위치만으로 포즈의 위치 부분을 업데이트 합니다.
        # 방향은 [0,0,0,1] (회전 없음)으로 가정합니다. 이는 직선 주행 시에는 문제가 적지만
        # 회전 시 오차를 유발할 수 있습니다. (향후 IMU 융합으로 개선 가능)
        q = [0, 0, 0, 1]
        self.vehicle_pose = quaternion_matrix(q)
        # 위도, 경도를 미터 단위로 변환하여 사용해야 하지만, 시뮬레이터의 Track Drive 환경은
        # 평평한 평면에 가깝고 크기가 작으므로, GPS 좌표를 그대로 월드 좌표처럼 사용합니다.
        # 실제 환경에서는 반드시 좌표 변환(e.g., UTM)이 필요합니다.
        self.vehicle_pose[0, 3] = msg.latitude
        self.vehicle_pose[1, 3] = msg.longitude
        self.vehicle_pose[2, 3] = msg.altitude

    def lidar_callback(self, msg: PointCloud2):
        raw_points = [p[:2] for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True) if np.hypot(p[0],p[1]) < LIDAR_RANGE_CUTOFF]
        if not raw_points: self.detected_cones_local = []; return
        db = DBSCAN(eps=0.5, min_samples=2).fit(raw_points)
        cone_centroids = [np.mean(np.array(raw_points)[db.labels_ == l], axis=0) for l in set(db.labels_) if l != -1 and np.hypot(*np.ptp(np.array(raw_points)[db.labels_ == l], axis=0)) <= 1.0]
        self.detected_cones_local = cone_centroids

    def control_logic(self):
        if self.startup_mode_active:
            if self.current_speed > CREEP_SPEED_THRESHOLD:
                self.startup_mode_active = False; rospy.loginfo("Startup complete.")
            else:
                rospy.loginfo_throttle(1.0, "Startup mode: Creeping forward.")
                self.publish_control(CREEP_THROTTLE, 0.0)
                # --- [핵심 수정] 올바른 인자 개수 및 데이터 형식으로 함수 호출 ---
                self.update_plot_data(self.detected_cones_local, [], [], None, 0.0)
                return

        path_cones = self.detected_cones_local; blue_path_cones, yellow_path_cones = [], []; path_found = False; goal_point = None; path_waypoints = []
        
        if len(path_cones) >= 4:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(path_cones)
            center_y_coords = kmeans.cluster_centers_[:, 1]; left_cluster_label = np.argmax(center_y_coords)
            blue_path_cones = np.array([c for i,c in enumerate(path_cones) if kmeans.labels_[i] == left_cluster_label])
            yellow_path_cones = np.array([c for i,c in enumerate(path_cones) if kmeans.labels_[i] != left_cluster_label])
        
        dynamic_target_speed = MIN_SPEED_IN_CORNER
        
        if len(blue_path_cones) >= 1 and len(yellow_path_cones) >= 1:
            current_center_points_local = []
            if len(blue_path_cones) < len(yellow_path_cones):
                for b_cone in blue_path_cones:
                    distances = np.linalg.norm(yellow_path_cones - b_cone, axis=1); y_cone = yellow_path_cones[np.argmin(distances)]; current_center_points_local.append((b_cone + y_cone) / 2.0)
            else:
                for y_cone in yellow_path_cones:
                    distances = np.linalg.norm(blue_path_cones - y_cone, axis=1); b_cone = blue_path_cones[np.argmin(distances)]; current_center_points_local.append((b_cone + y_cone) / 2.0)
            if current_center_points_local:
                points_to_transform = np.array(current_center_points_local).T
                points_to_transform = np.vstack([points_to_transform, np.zeros(points_to_transform.shape[1]), np.ones(points_to_transform.shape[1])])
                points_global = np.dot(self.vehicle_pose, points_to_transform).T[:, :2]
                self.path_history.extend(points_global)

        if len(self.path_history) >= 8:
            inv_vehicle_pose = np.linalg.inv(self.vehicle_pose)
            history_to_transform = np.array(self.path_history).T
            history_to_transform = np.vstack([history_to_transform, np.zeros(history_to_transform.shape[1]), np.ones(history_to_transform.shape[1])])
            history_local = np.dot(inv_vehicle_pose, history_to_transform).T[:, :2]
            relevant_history = np.array([p for p in history_local if p[0] > -1.0])
            if len(relevant_history) >= 8:
                sorted_indices = np.argsort(relevant_history[:, 0]); sorted_history = relevant_history[sorted_indices]
                unique_x, unique_indices = np.unique(sorted_history[:, 0], return_index=True)
                if len(unique_x) >= 4:
                    unique_history = sorted_history[unique_indices]
                    spline_path = CubicSpline(unique_history[:, 0], unique_history[:, 1])
                    x_for_curvature = 0.5; y_prime = spline_path(x_for_curvature, 1); y_double_prime = spline_path(x_for_curvature, 2)
                    curvature = abs(y_double_prime) / (1 + y_prime**2)**(3/2)
                    speed_factor = max(0, 1.0 - curvature / CURVATURE_SENSITIVITY)
                    dynamic_target_speed = MIN_SPEED_IN_CORNER + (MAX_SPEED_ON_STRAIGHT - MIN_SPEED_IN_CORNER) * speed_factor
                    x_points = np.arange(0.5, LIDAR_RANGE_CUTOFF, 0.5); y_points = spline_path(x_points)
                    path_waypoints = np.vstack([x_points, y_points]).T; path_found = True
        
        steering_command = self.last_valid_steering
        if path_found:
            dynamic_lookahead = LOOKAHEAD_BASE + self.current_speed * LOOKAHEAD_SPEED_RATIO
            distances = np.linalg.norm(path_waypoints, axis=1); goal_point_candidates = path_waypoints[distances > dynamic_lookahead]
            if len(goal_point_candidates) > 0:
                goal_point = goal_point_candidates[0]; alpha = math.atan2(goal_point[1], goal_point[0])
                steering_command = -1 * math.atan(2 * VEHICLE_WHEELBASE * math.sin(alpha) / dynamic_lookahead)
            else: path_found = False
        
        self.pid_controller.target = dynamic_target_speed; throttle = self.pid_controller.calculate(self.current_speed)
        if not path_found:
            rospy.logwarn_throttle(1.0, "Path not found, holding last steering!")
            throttle = 0.0
            self.path_history.clear()
        
        self.last_valid_steering = steering_command
        self.publish_control(throttle, steering_command)
        self.update_plot_data(blue_path_cones, yellow_path_cones, path_waypoints, goal_point, dynamic_target_speed)

    def publish_control(self, throttle, steering):
        cmd=ControlCommand(); cmd.throttle=np.clip(throttle,0,MAX_THROTTLE); cmd.steering=np.clip(steering,-MAX_STEER,MAX_STEER)
        self.control_pub.publish(cmd)
    def start_driving(self):
        rospy.sleep(1.0); go_pub=rospy.Publisher('/fsds/signal/go',Empty,queue_size=1); go_pub.publish(Empty())
        rospy.loginfo("주행 시작 'Go' 신호를 발행했습니다!")
    def run(self):
        rate = rospy.Rate(CONTROL_RATE); self.fig, self.ax = plt.subplots(figsize=(10, 10)); self.setup_plot(); plt.show(block=False)
        while not rospy.is_shutdown():
            self.control_logic()
            try: plt.draw(); plt.pause(0.001)
            except Exception: pass
            rate.sleep()
    def setup_plot(self):
        self.ax.set_title("Spline Path Controller"); self.ax.set_xlabel("X (local, m)"); self.ax.set_ylabel("Y (local, m)")
        self.blue_cones_plot, = self.ax.plot([], [], 'bo', markersize=6, label="Blue Cones")
        self.yellow_cones_plot, = self.ax.plot([], [], 'o', color='gold', markersize=6, label="Yellow Cones")
        self.center_path_plot, = self.ax.plot([], [], 'g--', linewidth=2, label="Spline Center Path")
        self.car_plot, = self.ax.plot([0], [0], 'r>', markersize=15, label="Car")
        self.goal_point_plot, = self.ax.plot([], [], 'rx', markersize=12, mew=3, label="Goal Point")
        self.speed_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plot_range = LIDAR_RANGE_CUTOFF; self.ax.set_xlim(-plot_range/3, plot_range); self.ax.set_ylim(-plot_range/2, plot_range/2)
        self.ax.grid(True); self.ax.legend(loc='upper right'); self.ax.set_aspect('equal', 'box')
    def update_plot_data(self, blue, yellow, path_waypoints, goal_point, target_speed):
        if len(blue)>0: self.blue_cones_plot.set_data(np.array(blue)[:,0], np.array(blue)[:,1])
        else: self.blue_cones_plot.set_data([], [])
        if len(yellow)>0: self.yellow_cones_plot.set_data(np.array(yellow)[:,0], np.array(yellow)[:,1])
        else: self.yellow_cones_plot.set_data([], [])
        if len(path_waypoints) > 0: self.center_path_plot.set_data(path_waypoints[:, 0], path_waypoints[:, 1])
        else: self.center_path_plot.set_data([], [])
        if goal_point is not None: self.goal_point_plot.set_data(goal_point[0], goal_point[1])
        else: self.goal_point_plot.set_data([], [])
        self.speed_text.set_text(f'Target: {target_speed:.2f} m/s\nCurrent: {self.current_speed:.2f} m/s')

if __name__ == '__main__':
    try:
        controller = FinalController()
        controller.run()
    except rospy.ROSInterruptException:
        pass