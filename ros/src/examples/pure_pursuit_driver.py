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
CRUISE_SPEED = 5.0
SPEED_KP = 1.0
LOOKAHEAD_DISTANCE = 3.0
STEERING_GAIN = 0.9
VEHICLE_LENGTH = 1.55
MAX_STEER_ANGLE = 0.45
LOCAL_PATH_WINDOW = 200
INTERPOLATION_FACTOR = 20
# ==============================================================================

class CruiseController:
    def __init__(self):
        rospy.init_node("cruise_controller")
        
        self.path = None
        self.current_pose, self.current_speed, self.is_path_generated = np.zeros(3), 0.0, False
        self.last_closest_idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        self.history_x, self.history_y, self.blue_cones_x, self.blue_cones_y, self.yellow_cones_x, self.yellow_cones_y = [], [], [], [], [], []
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.track_callback, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback, queue_size=1)
        
        rospy.loginfo("Cruise Controller (True Speed) 노드 시작됨.")

    def setup_plot(self):
        self.ax.set_title("Cruise Controller (True Speed)")
        self.ax.set_xlabel("X (m)"), self.ax.set_ylabel("Y (m)")
        self.path_plot, = self.ax.plot([], [], 'g--', label="Interpolated Path")
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=4, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'o', color='gold', markersize=4, label="Yellow Cones")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ko', markersize=6, label="Car")
        self.lookahead_plot, = self.ax.plot([], [], 'mx', markersize=8, mew=2, label="Lookahead Point")
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
            self.is_path_generated = True
            margin = 10
            self.ax.set_xlim(self.path[:, 0].min() - margin, self.path[:, 0].max() + margin)
            self.ax.set_ylim(self.path[:, 1].min() - margin, self.path[:, 1].max() + margin)
            rospy.loginfo("경로 생성 완료!")

    def odom_callback(self, odom_msg: Odometry):
        if not self.is_path_generated: return
        
        # [핵심 수정] 차량의 실제 속도(Speed)를 전진/측면 속도를 모두 고려하여 계산
        linear_vel = odom_msg.twist.twist.linear
        self.current_speed = math.sqrt(linear_vel.x**2 + linear_vel.y**2)
        
        q = odom_msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, yaw])
        self.history_x.append(self.current_pose[0]), self.history_y.append(self.current_pose[1])
        self.control_loop()

    def control_loop(self):
        # ... (이하 제어 로직은 이전과 동일)
        front_axle_pos = self.current_pose[:2] + VEHICLE_LENGTH * np.array([math.cos(self.current_pose[2]), math.sin(self.current_pose[2])])
        path_len = len(self.path)
        search_start_idx = self.last_closest_idx
        search_end_idx = (self.last_closest_idx + LOCAL_PATH_WINDOW) % path_len
        if search_start_idx < search_end_idx:
            local_path_indices = np.arange(search_start_idx, search_end_idx)
        else:
            local_path_indices = np.concatenate([np.arange(search_start_idx, path_len), np.arange(0, search_end_idx)])
        local_path = self.path[local_path_indices]
        distances = np.linalg.norm(local_path - front_axle_pos, axis=1)
        local_closest_idx = np.argmin(distances)
        closest_idx = local_path_indices[local_closest_idx]
        self.last_closest_idx = closest_idx
        target_idx = closest_idx
        for i in range(len(local_path_indices)):
            check_idx = local_path_indices[(local_closest_idx + i) % len(local_path_indices)]
            dist_from_car = np.linalg.norm(self.path[check_idx] - front_axle_pos)
            if dist_from_car >= LOOKAHEAD_DISTANCE:
                target_idx = check_idx
                break
        if target_idx == closest_idx: target_idx = (closest_idx + 10) % path_len
        target_point = self.path[target_idx]
        alpha = math.atan2(target_point[1] - front_axle_pos[1], target_point[0] - front_axle_pos[0]) - self.current_pose[2]
        delta = math.atan2(2.0 * VEHICLE_LENGTH * math.sin(alpha), LOOKAHEAD_DISTANCE)
        steering_angle = np.clip(STEERING_GAIN * delta, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        speed_error = CRUISE_SPEED - self.current_speed
        p_control_output = SPEED_KP * speed_error
        
        if p_control_output > 0:
            throttle, brake = np.clip(p_control_output, 0.0, 1.0), 0.0
        else:
            throttle, brake = 0.0, np.clip(abs(p_control_output), 0.0, 1.0)
        
        cmd_msg = ControlCommand(throttle=throttle, steering=-steering_angle, brake=brake)
        self.control_pub.publish(cmd_msg)
        self.update_plot_data(target_point)

    def update_plot_data(self, target_point):
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        self.blue_cone_plot.set_data(self.blue_cones_x, self.blue_cones_y)
        self.yellow_cone_plot.set_data(self.yellow_cones_x, self.yellow_cones_y)
        self.history_plot.set_data(self.history_x, self.history_y)
        self.car_plot.set_data([self.current_pose[0]], [self.current_pose[1]])
        self.lookahead_plot.set_data([target_point[0]], [target_point[1]])

if __name__ == '__main__':
    try:
        controller = CruiseController()
        plt.show(block=False)
        while not rospy.is_shutdown():
            plt.pause(0.01)
    except rospy.ROSInterruptException:
        rospy.loginfo("노드가 종료되었습니다.")