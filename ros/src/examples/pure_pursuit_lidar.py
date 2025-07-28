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

# ==============================================================================
# ======================== 튜닝 파라미터 (Tuning Parameters) =======================
# ==============================================================================
# 제어 관련
CONTROL_RATE = 50.0; MAX_STEER = 0.5; MAX_THROTTLE = 0.35
THROTTLE_PID_GAINS = {'Kp': 0.6, 'Ki': 0.1, 'Kd': 0.3}

# 지능형 속도 제어 파라미터
MAX_SPEED_ON_STRAIGHT = 4.0
MIN_SPEED_IN_CORNER = 2.0

# 퓨어 퍼슛 가변 룩어헤드 파라미터
VEHICLE_WHEELBASE = 1.55
LOOKAHEAD_BASE = 2.5
LOOKAHEAD_SPEED_RATIO = 0.7

# 인식 및 필터링 관련
LIDAR_RANGE_CUTOFF = 25.0
PATH_FILTER_ALPHA = 0.6
SPEED_FILTER_ALPHA = 0.2
LAT_TO_METER = 111132.0; LON_TO_METER = 111132.0

# 강제 출발 파라미터
CREEP_THROTTLE = 0.3
CREEP_SPEED_THRESHOLD = 0.5
# ==============================================================================

class PIDController:
    def __init__(self, gains, initial_target):
        self.gains = gains; self.target = initial_target; self.integral = 0; self.prev_error = 0; self.prev_value = 0.0; self.last_time = rospy.Time.now()
    def calculate(self, current_value):
        now, dt = rospy.Time.now(), (rospy.Time.now() - self.last_time).to_sec()
        if dt <= 0: return 0.0
        error = self.target - current_value; self.integral = np.clip(self.integral + error * dt, -1.0, 1.0)
        derivative = -(current_value - self.prev_value) / dt
        output = (self.gains['Kp']*error + self.gains['Ki']*self.integral + self.gains['Kd']*derivative)
        self.prev_error, self.last_time, self.prev_value = error, now, current_value
        return output

class FinalController:
    def __init__(self):
        rospy.init_node('final_controller')
        self.current_speed = 0.0; self.detected_cones_local = []; self.filtered_speed = 0.0; self.filtered_center_coeffs = None
        self.prev_gps_pos = None; self.last_gps_time = None; self.last_valid_steering = 0.0
        self.startup_mode_active = True
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

    def lidar_callback(self, msg: PointCloud2):
        raw_points = [p[:2] for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True) if np.hypot(p[0],p[1]) < LIDAR_RANGE_CUTOFF]
        if not raw_points: self.detected_cones_local = []; return
        db = DBSCAN(eps=0.5, min_samples=2).fit(raw_points)
        cone_centroids = [np.mean(np.array(raw_points)[db.labels_ == l], axis=0) for l in set(db.labels_) if l != -1 and np.hypot(*np.ptp(np.array(raw_points)[db.labels_ == l], axis=0)) <= 1.0]
        self.detected_cones_local = cone_centroids

    def control_logic(self):
        if self.startup_mode_active:
            if self.current_speed > CREEP_SPEED_THRESHOLD:
                self.startup_mode_active = False; rospy.loginfo("Startup complete. Switching to intelligent control mode.")
            else:
                rospy.loginfo_throttle(1.0, "Startup mode: Creeping forward to find path.")
                self.publish_control(CREEP_THROTTLE, 0.0)
                self.update_plot_data(self.detected_cones_local, [], [], None, 0.0)
                return

        path_cones = self.detected_cones_local; blue_path_cones, yellow_path_cones = [], []; path_found = False; goal_point = None; path_waypoints = []
        
        if len(path_cones) >= 4:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(path_cones)
            center_y_coords = kmeans.cluster_centers_[:, 1]; left_cluster_label = np.argmax(center_y_coords)
            blue_path_cones = np.array([c for i,c in enumerate(path_cones) if kmeans.labels_[i] == left_cluster_label])
            yellow_path_cones = np.array([c for i,c in enumerate(path_cones) if kmeans.labels_[i] != left_cluster_label])

        dynamic_target_speed = MIN_SPEED_IN_CORNER
        if len(blue_path_cones) >= 2 and len(yellow_path_cones) >= 2:
            blue_coeffs_1d = np.polyfit(blue_path_cones[:, 0], blue_path_cones[:, 1], 1); yellow_coeffs_1d = np.polyfit(yellow_path_cones[:, 0], yellow_path_cones[:, 1], 1)
            center_coeffs_1d = (blue_coeffs_1d + yellow_coeffs_1d) / 2.0
            if len(blue_path_cones) >= 3 and len(yellow_path_cones) >= 3:
                blue_coeffs_2d = np.polyfit(blue_path_cones[:, 0], blue_path_cones[:, 1], 2); yellow_coeffs_2d = np.polyfit(yellow_path_cones[:, 0], yellow_path_cones[:, 1], 2)
                center_coeffs_2d = (blue_coeffs_2d + yellow_coeffs_2d) / 2.0
                curvature = abs(center_coeffs_2d[0]); speed_factor = max(0, 1.0 - curvature / 0.015)
                dynamic_target_speed = MIN_SPEED_IN_CORNER + (MAX_SPEED_ON_STRAIGHT - MIN_SPEED_IN_CORNER) * speed_factor
            if self.filtered_center_coeffs is None: self.filtered_center_coeffs = center_coeffs_1d
            else: self.filtered_center_coeffs = PATH_FILTER_ALPHA * center_coeffs_1d + (1 - PATH_FILTER_ALPHA) * self.filtered_center_coeffs
            x_points = np.arange(0.5, LIDAR_RANGE_CUTOFF, 0.5); y_points = np.polyval(self.filtered_center_coeffs, x_points)
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
            rospy.logwarn_throttle(1.0, "Path not found, holding last steering!"); throttle = 0.0
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
        rate = rospy.Rate(CONTROL_RATE)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        plt.show(block=False)
        while not rospy.is_shutdown():
            self.control_logic()
            try: plt.draw(); plt.pause(0.001)
            except Exception: pass
            rate.sleep()

    def setup_plot(self):
        self.ax.set_title("Intelligent Speed & Path Controller"); self.ax.set_xlabel("X (local, m)"); self.ax.set_ylabel("Y (local, m)")
        self.blue_cones_plot, = self.ax.plot([], [], 'bo', markersize=6, label="Blue Cones")
        self.yellow_cones_plot, = self.ax.plot([], [], 'o', color='gold', markersize=6, label="Yellow Cones")
        self.center_path_plot, = self.ax.plot([], [], 'g--', linewidth=2, label="Center Path")
        self.car_plot, = self.ax.plot([0], [0], 'r>', markersize=15, label="Car")
        self.goal_point_plot, = self.ax.plot([], [], 'rx', markersize=12, mew=3, label="Goal Point")
        self.speed_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plot_range = LIDAR_RANGE_CUTOFF; self.ax.set_xlim(-plot_range/3, plot_range); self.ax.set_ylim(-plot_range/2, plot_range/2)
        self.ax.grid(True); self.ax.legend(loc='upper right'); self.ax.set_aspect('equal', 'box')

    def update_plot_data(self, blue, yellow, path_waypoints, goal_point, target_speed):
        # --- [핵심 수정] 데이터를 Numpy 배열로 변환 ---
        # blue, yellow는 list of lists일 수 있으므로, 슬라이싱 전에 np.array로 변환
        if len(blue)>0:
            blue_arr = np.array(blue)
            self.blue_cones_plot.set_data(blue_arr[:,0], blue_arr[:,1])
        else: self.blue_cones_plot.set_data([], [])
        
        if len(yellow)>0:
            yellow_arr = np.array(yellow)
            self.yellow_cones_plot.set_data(yellow_arr[:,0], yellow_arr[:,1])
        else: self.yellow_cones_plot.set_data([], [])
        # ----------------------------------------

        if len(path_waypoints) > 0: self.center_path_plot.set_data(path_waypoints[:, 0], path_waypoints[:, 1])
        else: self.center_path_plot.set_data([], [])
        if goal_point is not None: self.goal_point_plot.set_data([goal_point[0]], [goal_point[1]])
        else: self.goal_point_plot.set_data([], [])
        self.speed_text.set_text(f'Target: {target_speed:.2f} m/s\nCurrent: {self.current_speed:.2f} m/s')

if __name__ == '__main__':
    try:
        controller = FinalController()
        controller.run()
    except rospy.ROSInterruptException:
        pass