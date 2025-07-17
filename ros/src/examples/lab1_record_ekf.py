#!/usr/bin/env python3
# =============================================================
#  LAB 1 - EKF SLAM & GPS PATH RECORDING (Final Version 2.1)
#  - Perception: DBSCAN Clustering
#  - Steering: 2nd-Order Polynomial Path Fitting with PD Control
#  - Control: State-Aware Throttle Control
# =============================================================
import rospy
import os, math, threading
import numpy as np, utm
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt, matplotlib.animation as animation
from fs_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, NavSatFix
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Empty
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# --- 파라미터 ---
# 파일 및 제어 관련
PATH_FILENAME = "lab1_recorded_path.csv"
CONTROL_RATE = 20.0
VEHICLE_LENGTH = 1.5
MAX_STEER = 0.8

# 속도 제어 관련
TARGET_SPEED = 1.5
THROTTLE_FF_GAIN = 0.15
THROTTLE_PID_GAINS = {'Kp': 0.3, 'Ki': 0.05, 'Kd': 0.1}

# 스티어링 제어 관련
STEERING_PD_GAINS = {'Kp': 1.0, 'Kd': 0.6}
LOCAL_PATH_BOX = {'x_min': 1.0, 'x_max': 10.0, 'y_abs_max': 6.0}
THROTTLE_CUTOFF_THRESHOLD = 1.2

# 인식(Perception) 관련
DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 4
MAX_CLUSTER_SIZE = 0.6

# SLAM 및 센서 관련
LIDAR_RANGE_CUTOFF = 15.0
MAP_UPDATE_THRESHOLD = 1.0
MEASUREMENT_NOISE = 0.1
INITIAL_POSE_UNCERTAINTY = 0.5

class PIDController:
    def __init__(self, gains, target):
        self.gains, self.target = gains, target
        self.integral, self.prev_value, self.last_time = 0, 0, rospy.Time.now()

    def calculate(self, current_value):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        if dt <= 0: return 0.0

        error = self.target - current_value
        derivative = (self.prev_value - current_value) / dt
        
        output = (self.gains['Kp'] * error + self.gains['Ki'] * self.integral + self.gains['Kd'] * derivative)
        
        if not (output >= 1.0 and error > 0) and not (output <= 0.0 and error < 0):
            self.integral += error * dt
            self.integral = np.clip(self.integral, -2.0, 2.0)

        self.prev_value, self.last_time = current_value, now
        return output

class Lab1_EKF_Recorder:
    def __init__(self):
        rospy.init_node('lab1_recorder_node')
        rospy.loginfo("LAB 1: 최종 코드 실행 (v2.1)")
        
        path_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_file_full = os.path.join(path_dir, PATH_FILENAME)
        
        self.current_x, self.current_y, self.current_yaw, self.current_speed = 0.0, 0.0, 0.0, 0.0
        self.points, self.origin_utm = None, None
        self.recorded_path, self.slam_map_blue, self.slam_map_yellow = [], [], []
        self.next_cone_id, self.lap_start_pos = 0, None
        self.start_time = rospy.Time.now()
        self.prev_steering_error = 0.0

        self.points_lock = threading.Lock()
        self.map_lock = threading.Lock()
        self.pid_controller = PIDController(THROTTLE_PID_GAINS, TARGET_SPEED)
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/fsds/lidar/Lidar2', PointCloud2, self.lidar_callback)
        rospy.Subscriber('/fsds/gps', NavSatFix, self.gps_callback)
        rospy.Timer(rospy.Duration(1.0/CONTROL_RATE), self.control_loop)
        rospy.on_shutdown(self.save_recorded_path)

        self.setup_plot()
        self.start_driving()

    def control_loop(self, event=None):
        self.update_slam_map_ekf()
        
        steering_command, steering_error = self.calculate_robust_steering()
        
        if abs(steering_error) > THROTTLE_CUTOFF_THRESHOLD:
            stability_factor = 0.0
        else:
            stability_factor = 1.0

        feed_forward_throttle = TARGET_SPEED * THROTTLE_FF_GAIN
        pid_throttle = self.pid_controller.calculate(self.current_speed)
        base_throttle = feed_forward_throttle + pid_throttle

        throttle = base_throttle * stability_factor
        
        self.publish_control(throttle, steering_command)

        if self.lap_start_pos is None and self.current_speed > 0.5:
            self.lap_start_pos = (self.current_x, self.current_y)
        
        if self.lap_start_pos and (rospy.Time.now() - self.start_time).to_sec() > 20.0:
            if np.hypot(self.current_x - self.lap_start_pos[0], self.current_y - self.lap_start_pos[1]) < 4.0:
                rospy.loginfo("LAB 1 완주! 경로 기록을 종료합니다.")
                rospy.signal_shutdown("Lap finished.")
    
    def calculate_robust_steering(self):
        with self.map_lock:
            blue_cones = [c['state'].flatten() for c in self.slam_map_blue]
            yellow_cones = [c['state'].flatten() for c in self.slam_map_yellow]

        T_world_to_car = np.array([
            [math.cos(self.current_yaw), math.sin(self.current_yaw), -self.current_x*math.cos(self.current_yaw) - self.current_y*math.sin(self.current_yaw)],
            [-math.sin(self.current_yaw), math.cos(self.current_yaw), self.current_x*math.sin(self.current_yaw) - self.current_y*math.cos(self.current_yaw)]
        ])
        
        def transform_cones(cones):
            if not cones: return np.array([])
            cones_np = np.array(cones)
            ones = np.ones((cones_np.shape[0], 1))
            cones_homogeneous = np.hstack((cones_np, ones))
            transformed = T_world_to_car @ cones_homogeneous.T
            return transformed.T

        blue_local = transform_cones(blue_cones)
        yellow_local = transform_cones(yellow_cones)

        def filter_cones(cones):
            if len(cones) == 0: return np.array([])
            return cones[(cones[:, 0] > LOCAL_PATH_BOX['x_min']) & 
                         (cones[:, 0] < LOCAL_PATH_BOX['x_max']) & 
                         (np.abs(cones[:, 1]) < LOCAL_PATH_BOX['y_abs_max'])]

        blue_filtered = filter_cones(blue_local)
        yellow_filtered = filter_cones(yellow_local)
        
        center_points = []
        if len(blue_filtered) > 0 and len(yellow_filtered) > 0:
            distances = cdist(yellow_filtered, blue_filtered)
            min_blue_indices = np.argmin(distances, axis=1)
            for i, blue_idx in enumerate(min_blue_indices):
                midpoint = (yellow_filtered[i] + blue_filtered[blue_idx]) / 2.0
                center_points.append(midpoint)
        
        if len(center_points) >= 3:
            center_points = np.array(center_points)
            coeffs = np.polyfit(center_points[:, 0], center_points[:, 1], 2)
            target_angle = math.atan(coeffs[1])
        else:
            def get_line_fit(cones):
                if len(cones) < 2: return None
                return np.polyfit(cones[:, 0], cones[:, 1], 1)

            blue_line = get_line_fit(blue_filtered)
            yellow_line = get_line_fit(yellow_filtered)

            if blue_line is not None and yellow_line is not None:
                target_angle = math.atan((blue_line[0] + yellow_line[0]) / 2.0)
            elif blue_line is not None:
                target_angle = math.atan(blue_line[0]) - 0.3
            elif yellow_line is not None:
                target_angle = math.atan(yellow_line[0]) + 0.3
            else:
                return 0.0, 0.0 # [수정됨] 항상 2개의 값을 반환

        steering_error = target_angle
        derivative_error = steering_error - self.prev_steering_error
        self.prev_steering_error = steering_error
        
        steering = STEERING_PD_GAINS['Kp'] * steering_error + STEERING_PD_GAINS['Kd'] * derivative_error
        return steering, steering_error

    def lidar_callback(self, msg):
        raw_points = [p[:2] for p in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True) if np.hypot(p[0], p[1]) < LIDAR_RANGE_CUTOFF]
        if not raw_points: return

        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(raw_points)
        labels = db.labels_

        cone_candidates = []
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1: continue
            class_member_mask = (labels == k)
            cluster_points = np.array(raw_points)[class_member_mask]
            if np.hypot(*np.ptp(cluster_points, axis=0)) > MAX_CLUSTER_SIZE: continue
            centroid = np.mean(cluster_points, axis=0)
            cone_candidates.append(centroid)
        with self.points_lock: self.points = cone_candidates
    
    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def gps_callback(self, msg):
        easting, northing, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        if self.origin_utm is None: self.origin_utm = (easting, northing)
        x = easting - self.origin_utm[0]
        y = northing - self.origin_utm[1]
        if not self.recorded_path or np.hypot(x - self.recorded_path[-1][0], y - self.recorded_path[-1][1]) > 0.1:
            self.recorded_path.append([x, y])

    def update_slam_map_ekf(self):
        with self.points_lock:
            if self.points is None: return
            points_copy = list(self.points)
        if not points_copy: return

        R = np.identity(2) * MEASUREMENT_NOISE**2
        T = np.array([[math.cos(self.current_yaw), -math.sin(self.current_yaw), self.current_x],
                      [math.sin(self.current_yaw), math.cos(self.current_yaw), self.current_y],
                      [0, 0, 1]])
        
        detected_cones = [{'pos': np.dot(T, [p[0], p[1], 1])[:2], 'color': 'b' if p[1] > 0 else 'y'} for p in points_copy]
        
        with self.map_lock:
            for d_cone in detected_cones:
                map_to_update = self.slam_map_blue if d_cone['color'] == 'b' else self.slam_map_yellow
                
                if not map_to_update:
                    map_to_update.append({'id': self.next_cone_id, 'state': d_cone['pos'].reshape(2, 1), 'P': np.identity(2) * INITIAL_POSE_UNCERTAINTY**2})
                    self.next_cone_id += 1
                    continue

                distances = [np.hypot(d_cone['pos'][0] - m['state'][0], d_cone['pos'][1] - m['state'][1]) for m in map_to_update]
                
                if min(distances) < MAP_UPDATE_THRESHOLD:
                    match_idx = np.argmin(distances)
                    map_cone = map_to_update[match_idx]
                    H = np.identity(2)
                    y = d_cone['pos'].reshape(2, 1) - map_cone['state']
                    S = H @ map_cone['P'] @ H.T + R
                    K = map_cone['P'] @ H.T @ np.linalg.inv(S)
                    map_cone['state'] += K @ y
                    map_cone['P'] = (np.identity(2) - K @ H) @ map_cone['P']
                else:
                    map_to_update.append({'id': self.next_cone_id, 'state': d_cone['pos'].reshape(2, 1), 'P': np.identity(2) * INITIAL_POSE_UNCERTAINTY**2})
                    self.next_cone_id += 1
                    
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.slam_blue_plot, = self.ax.plot([], [], 'bo', markersize=5, label="Blue Cones (EKF)")
        self.slam_yellow_plot, = self.ax.plot([], [], 'yo', markersize=5, label="Yellow Cones (EKF)")
        self.gps_path_plot, = self.ax.plot([], [], 'g-', linewidth=2, label="Recorded GPS Path")
        self.car_plot, = self.ax.plot([], [], 'ko', markersize=8, label="Car")
        self.ax.set_title("LAB 1: Final Code (Polynomial Fit)")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', 'box')
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

    def update_plot(self, frame):
        with self.map_lock:
            blue_states = [c['state'].flatten() for c in self.slam_map_blue]
            yellow_states = [c['state'].flatten() for c in self.slam_map_yellow]
        
        if blue_states: self.slam_blue_plot.set_data(np.array(blue_states)[:, 0], np.array(blue_states)[:, 1])
        if yellow_states: self.slam_yellow_plot.set_data(np.array(yellow_states)[:, 0], np.array(yellow_states)[:, 1])
        if self.recorded_path: self.gps_path_plot.set_data(np.array(self.recorded_path)[:, 0], np.array(self.recorded_path)[:, 1])
        self.car_plot.set_data(self.current_x, self.current_y)
        
        all_x = [self.current_x] + [c[0] for c in blue_states] + [c[0] for c in yellow_states]
        all_y = [self.current_y] + [c[1] for c in blue_states] + [c[1] for c in yellow_states]
        if len(all_x) > 1:
            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)
            x_margin = max((x_max - x_min) * 0.2, 10)
            y_margin = max((y_max - y_min) * 0.2, 10)
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
        return []

    def start_driving(self):
        go_pub = rospy.Publisher('/fsds/signal/go', Empty, queue_size=1)
        rospy.sleep(1.0)
        go_pub.publish(Empty())
        rospy.loginfo("주행 시작 'Go' 신호를 발행했습니다!")
        
    def save_recorded_path(self):
        if self.recorded_path:
            rospy.loginfo(f"경로를 '{self.path_file_full}'에 저장합니다...")
            np.savetxt(self.path_file_full, np.array(self.recorded_path), delimiter=",", fmt='%.4f')
            rospy.loginfo("경로 저장 완료!")
            
    def publish_control(self, throttle, steering):
        cmd = ControlCommand(header=rospy.Header(stamp=rospy.Time.now()))
        cmd.throttle = np.clip(throttle, 0.0, 1.0)
        cmd.steering = np.clip(steering, -MAX_STEER, MAX_STEER)
        self.control_pub.publish(cmd)

if __name__ == '__main__':
    try:
        lab = Lab1_EKF_Recorder()
        plt.show()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close('all')