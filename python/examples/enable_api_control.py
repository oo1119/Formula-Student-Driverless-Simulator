#!/usr/bin/env python3
import sys
import os

# FSDS Python 라이브러리 경로 설정
try:
    fsds_lib_path = os.path.join(os.path.expanduser("~"), "FSDS/Formula-Student-Driverless-Simulator", "python")
    sys.path.insert(0, fsds_lib_path)
    import fsds
except ImportError:
    fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
    sys.path.insert(0, fsds_lib_path)
    import fsds

# FSDS 클라이언트에 연결
client = fsds.FSDSClient()
client.confirmConnection()

# API 제어 활성화
print("API 제어를 활성화합니다...")
client.enableApiControl(True)
print("✅ API 제어가 활성화되었습니다. 이제 자율주행 노드를 실행할 수 있습니다.")