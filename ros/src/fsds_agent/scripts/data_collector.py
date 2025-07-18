#!/usr/bin/env python3
import rospy
import os
import rospkg
import cv2
import csv
import time
from sensor_msgs.msg import Image
from fs_msgs.msg import ControlCommand
from geometry_msgs.msg import TwistWithCovarianceStamped
from cv_bridge import CvBridge, CvBridgeError

class DataCollector:
    def __init__(self):
        rospy.init_node('passive_data_collector_node')

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('fsds_agent')
        self.data_path = os.path.join(self.pkg_path, "data")
        rospy.loginfo(f"Data will be saved in: {self.data_path}")
        
        self.bridge = CvBridge()
        self.frame_counter = 0

        # 최신 데이터를 저장할 변수
        self.latest_cam1_msg = None
        self.latest_cam2_msg = None
        self.latest_gss_msg = None
        self.latest_steering = 0.0

        # --- [수정됨] 데이터 저장 설정 (이어쓰기 방식) ---
        self.log_file_path = os.path.join(self.data_path, 'driving_log_passive.csv')
        
        # 1. 파일이 이미 존재하는지 확인
        file_exists = os.path.isfile(self.log_file_path)
        
        # 2. 파일을 'a' (append) 모드로 열기
        self.log_file = open(self.log_file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # 3. 파일이 새로 생성된 경우에만 헤더를 씀
        if not file_exists:
            self.csv_writer.writerow(['cam1_path', 'cam2_path', 'steering', 'speed'])

        # 각 토픽에 대한 개별 Subscriber
        rospy.Subscriber('/fsds/control_command', ControlCommand, self.control_callback)
        rospy.Subscriber('/fsds/cameracam1', Image, self.cam1_callback)
        rospy.Subscriber('/fsds/cameracam2', Image, self.cam2_callback)
        rospy.Subscriber('/fsds/gss', TwistWithCovarianceStamped, self.gss_callback)

        # 일정한 주기로 데이터 저장을 시도하는 Timer
        self.collection_rate = 10  # 1초에 10번 (10Hz) 데이터 수집
        self.save_timer = rospy.Timer(rospy.Duration(1.0 / self.collection_rate), self.save_data_loop)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Passive Data Collector is running. Recording will start upon receiving first sensor data.")

    def control_callback(self, msg):
        self.latest_steering = msg.steering
    def cam1_callback(self, msg):
        self.latest_cam1_msg = msg
    def cam2_callback(self, msg):
        self.latest_cam2_msg = msg
    def gss_callback(self, msg):
        self.latest_gss_msg = msg

    def save_data_loop(self, event):
        """타이머에 의해 주기적으로 호출되어 최신 데이터 스냅샷을 저장"""
        
        # [수정됨] 모든 시작 조건 제거. 모든 센서 데이터가 한 번이라도 들어왔으면 바로 기록 시작.
        if not all([self.latest_cam1_msg, self.latest_cam2_msg, self.latest_gss_msg]):
            return 

        if self.frame_counter == 0:
            rospy.loginfo("--- First full sensor set received. Recording all data now! ---")

        try:
            cv_cam1 = self.bridge.imgmsg_to_cv2(self.latest_cam1_msg, "bgr8")
            cv_cam2 = self.bridge.imgmsg_to_cv2(self.latest_cam2_msg, "bgr8")
            steering = self.latest_steering
            speed = self.latest_gss_msg.twist.twist.linear.x
        except (CvBridgeError, AttributeError) as e:
            rospy.logwarn_throttle(1.0, f"Could not process data for saving: {e}")
            return

        timestamp = f"{int(time.time()*1000)}_{self.frame_counter:05d}"
        cam1_filename = f"cam1/{timestamp}.jpg"
        cam2_filename = f"cam2/{timestamp}.jpg"
        
        cv2.imwrite(os.path.join(self.data_path, cam1_filename), cv_cam1)
        cv2.imwrite(os.path.join(self.data_path, cam2_filename), cv_cam2)

        self.csv_writer.writerow([cam1_filename, cam2_filename, steering, speed])
        self.frame_counter += 1

        if self.frame_counter % 100 == 0:
            rospy.loginfo(f"Collected {self.frame_counter} frames...")

    def cleanup(self):
        if self.log_file:
            self.log_file.close()
            rospy.loginfo(f"Driving log saved with {self.frame_counter} entries. Data collection finished.")

if __name__ == '__main__':
    try:
        DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass