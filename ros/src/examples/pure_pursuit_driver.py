#!/usr/bin/env python3
import rospy
import math
import numpy as np
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ROS 메시지 타입 임포트
from fs_msgs.msg import ControlCommand, Track, Cone
from nav_msgs.msg import Odometry

class PurePursuitVisualizer:
    def __init__(self):
        rospy.init_node('pure_pursuit_visualizer')

        # === 파라미터 설정 ===
        self.LOOKAHEAD_DISTANCE = 3.0
        self.VEHICLE_LENGTH = 0.49
        self.TARGET_THROTTLE = 0.5

        # === 변수 초기화 ===
        self.path = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.history_x = []
        self.history_y = []

        # === 시각화 설정 (matplotlib) ===
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        # 맵 요소는 한 번만 그림
        self.blue_cone_plot, = self.ax.plot([], [], 'bo', markersize=5, label="Blue Cones")
        self.yellow_cone_plot, = self.ax.plot([], [], 'yo', markersize=5, label="Yellow Cones")
        self.path_plot, = self.ax.plot([], [], 'g--', label="Center Path")
        # 실시간 업데이트될 요소
        self.history_plot, = self.ax.plot([], [], 'r-', label="Car Trajectory")
        self.car_plot, = self.ax.plot([], [], 'ro', markersize=8, label="Car")
        
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("FSDS Pure Pursuit Visualization")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        
        # 애니메이션 설정
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=100, blit=True)

        # === ROS Publisher & Subscriber ===
        self.control_pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
        rospy.Subscriber('/fsds/testing_only/track', Track, self.path_callback)
        rospy.Subscriber('/fsds/testing_only/odom', Odometry, self.odom_callback)
        rospy.loginfo("Pure Pursuit Visualizer 노드 시작됨.")

    def init_plot(self):
        """애니메이션 초기화 함수: 배경(맵)을 먼저 그림"""
        self.history_plot.set_data([], [])
        self.car_plot.set_data([], [])
        return self.history_plot, self.car_plot

    def path_callback(self, track_msg):
        if self.path is not None: return
        rospy.loginfo("트랙 데이터를 수신하여 경로를 생성합니다...")

        blue_cones = [c for c in track_msg.track if c.color == Cone.BLUE]
        yellow_cones = [c for c in track_msg.track if c.color == Cone.YELLOW]
        
        if not blue_cones or not yellow_cones: return

        # 맵(콘, 경로) 플롯 데이터 설정
        self.blue_cone_plot.set_data([c.location.x for c in blue_cones], [c.location.y for c in blue_cones])
        self.yellow_cone_plot.set_data([c.location.x for c in yellow_cones], [c.location.y for c in yellow_cones])
        
        path_points = []
        for i in range(min(len(blue_cones), len(yellow_cones))):
            mid_x = (blue_cones[i].location.x + yellow_cones[i].location.x) / 2.0
            mid_y = (blue_cones[i].location.y + yellow_cones[i].location.y) / 2.0
            path_points.append([mid_x, mid_y])
        
        self.path = np.array(path_points)
        self.path_plot.set_data(self.path[:, 0], self.path[:, 1])
        
        # 전체 맵이 보이도록 축 범위 설정
        x_margin, y_margin = 10, 10
        self.ax.set_xlim(self.path[:, 0].min() - x_margin, self.path[:, 0].max() + x_margin)
        self.ax.set_ylim(self.path[:, 1].min() - y_margin, self.path[:, 1].max() + y_margin)

        rospy.loginfo(f"{len(self.path)}개의 웨이포인트 경로 생성 완료.")

    def odom_callback(self, odom_msg):
        if self.path is None:
            rospy.logwarn_throttle(1.0, "경로 데이터를 기다리는 중...")
            return

        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.history_x.append(self.current_x)
        self.history_y.append(self.current_y)

        orientation_q = odom_msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        
        steering_angle = self.calculate_steering_angle()
        
        control_cmd = ControlCommand()
        control_cmd.header.stamp = rospy.Time.now()
        control_cmd.throttle = self.TARGET_THROTTLE
        control_cmd.steering = steering_angle
        control_cmd.brake = 0.0
        
        self.control_pub.publish(control_cmd)

    def calculate_steering_angle(self):
        dx = self.path[:, 0] - self.current_x
        dy = self.path[:, 1] - self.current_y
        distances = np.hypot(dx, dy)
        closest_idx = np.argmin(distances)

        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            dist_from_car = np.hypot(self.path[i, 0] - self.current_x, self.path[i, 1] - self.current_y)
            if dist_from_car >= self.LOOKAHEAD_DISTANCE:
                target_idx = i
                break
        
        target_x, target_y = self.path[target_idx]
        relative_y = (target_y - self.current_y) * math.cos(self.current_yaw) - (target_x - self.current_x) * math.sin(self.current_yaw)
        delta = math.atan2(2.0 * self.VEHICLE_LENGTH * relative_y, self.LOOKAHEAD_DISTANCE**2)
        
        return np.clip(delta, -1.0, 1.0)

    def update_plot(self, frame):
        """애니메이션 프레임 업데이트 함수"""
        # 오류 수정: 단일 값이 아닌 리스트 형태로 데이터 전달
        self.car_plot.set_data([self.current_x], [self.current_y])
        self.history_plot.set_data(self.history_x, self.history_y)
        return self.history_plot, self.car_plot

if __name__ == '__main__':
    try:
        controller = PurePursuitVisualizer()
        plt.show() # 플롯 창을 띄움
        rospy.spin()
    except rospy.ROSInterruptException:
        pass