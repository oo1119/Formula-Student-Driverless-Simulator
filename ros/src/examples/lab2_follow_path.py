#!/usr/bin/env python3
# =============================================================
#  LAB 2 - PURE PURSUIT PATH FOLLOWING
# =============================================================
import rospy
import os, time, math, threading
import numpy as np
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt, matplotlib.animation as animation
from fs_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

# --- 파라미터 ---
PATH_FILENAME = "lab1_ekf_path.csv" # 1랩에서 저장한 파일 이름과 동일해야 함
TARGET_SPEED = 3.0
CONTROL_RATE = 20.0
VEHICLE_LENGTH = 1.5
MAX_STEER = 0.8
LOOKAHEAD_BASE = 2.0
LOOKAHEAD_SPEED_RATIO = 0.8
THROTTLE_PID_GAINS = {'Kp': 1.2, 'Ki': 0.1, 'Kd': 0.2}

class PIDController:
    # ... (위 1번 스크립트의 PIDController 클래스와 동일) ...
    def __init__(self, gains, target):
        self.gains, self.target = gains, target
        self.integral, self.prev_error, self.last_time = 0, 0, rospy.Time.now()
    def calculate(self, current_value):
        now = rospy.Time.now(); dt = (now - self.last_time).to_sec()
        if dt <= 0: return 0.0
        error = self.target - current_value
        output = (self.gains['Kp']*error + self.gains['Ki']*self.integral + self.gains['Kd']*derivative)
        if not (output >= 1.0 and error > 0) and not (output <= 0.0 and error < 0):
            self.integral += error * dt
            self.integral = np.clip(self.integral, -2.0, 2.0)
        derivative = (error - self.prev_error) / dt
        self.prev_error, self.last_time = error, now
        return np.clip(output, 0.0, 1.0)

class Lab2_PurePursuit_Follower:
    def __init__(self):
        rospy.init_node('lab2_follower_node')
        rospy.loginfo("LAB 2: 경로 추종 노드 시작")

        path_dir = os.path.dirname(os.path.realpath(__file__))
        self.path_file_full = os.path.join(path_dir, PATH_FILENAME)
        if not os.path.exists(self.path_file_full):
            rospy.logerr(f"경로 파일 '{self.path_file_full}' 없음. Lab 1을 먼저 실행하세요."); rospy.signal_shutdown("경로 파일 없음")
        self.reference_path = np.loadtxt(self.path_file_full, delimiter=",")
        rospy.loginfo(f"총 {len(self.reference_path)}개의 경로 포인트를 불러왔습니다.")

        self.current_x, self.current_y, self.current_yaw, self.current_speed = 0.0,0.0,0.0,0.0
        self.history_x, self.history_y, self.lap_start_pos = [], [], None
        self.pid_controller = PIDController(THROTTLE_PID_GAINS, TARGET_SPEED)
        
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)
        rospy.Timer(rospy.Duration(1.0/CONTROL_RATE), self.control_loop)

        self.setup_plot()
        self.start_driving()

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x
        self.current_x, self.current_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
    def control_loop(self, event=None):
        if not self.history_x: self.lap_start_pos = (self.current_x, self.current_y)
        self.history_x.append(self.current_x); self.history_y.append(self.current_y)
        
        lookahead = LOOKAHEAD_BASE + LOOKAHEAD_SPEED_RATIO * self.current_speed
        steering, target_idx = self.calculate_steering_angle(lookahead)
        throttle = self.pid_controller.calculate(self.current_speed)
        self.publish_control(throttle, steering)
        
        if target_idx is not None:
            self.lookahead_plot.set_data([self.reference_path[target_idx, 0]], [self.reference_path[target_idx, 1]])

        if self.lap_start_pos and (rospy.Time.now() - self.start_time).to_sec() > 10.0:
            if np.hypot(self.current_x - self.lap_start_pos[0], self.current_y - self.lap_start_pos[1]) < 4.0:
                rospy.loginfo("LAB 2 완주!"); rospy.signal_shutdown("Lap finished.")

    def calculate_steering_angle(self, lookahead):
        dx, dy = self.reference_path[:, 0] - self.current_x, self.reference_path[:, 1] - self.current_y
        distances = np.hypot(dx, dy); closest_idx = np.argmin(distances)
        target_idx = -1
        for i in range(closest_idx, len(distances)):
            if distances[i] >= lookahead: target_idx = i; break
        else: target_idx = len(distances) - 1
        if target_idx == -1: return 0.0, None
        
        target_x, target_y = self.reference_path[target_idx]
        alpha = math.atan2(target_y - self.current_y, target_x - self.current_x) - self.current_yaw
        delta = math.atan2(2.0 * VEHICLE_LENGTH * math.sin(alpha), lookahead)
        return -delta, target_idx

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.plot(self.reference_path[:, 0], self.reference_path[:, 1], 'g--', label="Reference Path")
        self.history_plot, = self.ax.plot([], [], 'r-', label="Actual Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ro', markersize=8, label="Car")
        self.lookahead_plot, = self.ax.plot([], [], 'mo', markersize=8, label="Lookahead")
        self.ax.set_title(f"LAB 2: Pure Pursuit (Target: {TARGET_SPEED} m/s)")
        self.ax.legend(); self.ax.grid(True); self.ax.set_aspect('equal', 'box')
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50)

    def update_plot(self, frame):
        if self.history_x: self.history_plot.set_data(self.history_x, self.history_y)
        self.car_plot.set_data(self.current_x, self.current_y)
        margin = 15
        self.ax.set_xlim(self.current_x-margin, self.current_x+margin); self.ax.set_ylim(self.current_y-margin, self.current_y+margin)
        return []

    def start_driving(self):
        self.start_time = rospy.Time.now()
        go_pub = rospy.Publisher('/fsds/signal/go', Empty, queue_size=1)
        rospy.sleep(1.0); go_pub.publish(Empty()); rospy.loginfo("주행 시작 'Go' 신호를 발행했습니다!")

    def publish_control(self, throttle, steering):
        cmd = ControlCommand(header=rospy.Header(stamp=rospy.Time.now()))
        cmd.throttle = np.clip(throttle, 0.0, 1.0); cmd.steering = np.clip(steering, -MAX_STEER, MAX_STEER)
        self.control_pub.publish(cmd)

if __name__ == '__main__':
    try:
        lab = Lab2_PurePursuit_Follower(); plt.show(); rospy.spin()
    except rospy.ROSInterruptException: pass
    finally: plt.close('all')