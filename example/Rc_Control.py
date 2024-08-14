import sys
import os
import cv2
import concurrent.futures
from datetime import datetime
import warnings
import time
import requests
import numpy as np

# 필요한 모듈을 /Users/apple/Desktop/Python/Smarcle/MakersDay/StereoCam/실시간/src 경로에서 임포트
sys.path.insert(0, '/Users/apple/Desktop/Python/Smarcle/MakersDay/StereoCam/실시간/src')

from capture import capture_frame_from_esp32
from processing import process_frames
from visualization import annotate_image_with_distances
from class_map import CLASS_MAP
import model as yolo_model

# pyserial 모듈 임포트
try:
    import serial
except ImportError:
    print("pyserial 모듈이 설치되지 않았습니다. 'pip install pyserial' 명령어로 설치하세요.")
    sys.exit(1)

# 모든 경고 무시
warnings.filterwarnings("ignore")

# Arduino 시리얼 포트 설정
try:
    arduino = serial.Serial("/dev/cu.usbmodem11301", 9600, timeout=.1)  # 포트 이름과 보드레이트 설정
except AttributeError:
    print("serial 모듈에서 'Serial' 속성을 찾을 수 없습니다. 모듈이 제대로 설치되었는지 확인하세요.")
    sys.exit(1)

# 탐지할 객체의 초기 카테고리
TARGET_CATEGORY = 'bottle'  # 다른 카테고리로 변경하려면 이 값을 수정

# 거리 임계값 설정 (단위: cm)
MIN_DISTANCE_THRESHOLD = 10  # 10cm 이하의 거리는 무시

def send_distance_to_arduino(distance):
    """
    아두이노에 거리를 전송하는 함수
    """
    if distance > MIN_DISTANCE_THRESHOLD:
        arduino.write(bytes(f"{distance}\n", 'utf-8'))
        print(f"Distance sent to Arduino: {distance} cm")

def check_arduino_signal():
    """
    아두이노에서 신호를 받는 함수
    """
    if arduino.in_waiting > 0:
        return arduino.readline().decode('utf-8').strip()
    return None

def main():
    # ESP32 카메라 주소 (해상도 QVGA)
    left_cam_url = "http://192.168.0.14/capture"
    right_cam_url = "http://192.168.0.13/capture"

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/yolov5s.pt'
    model = yolo_model.load_model(weights_path)

    # 스테레오 비전 설정
    fl = 2.043636363636363
    tantheta = 0.5443642625

    # img_width = 640  # VGA 해상도 기준
    img_width = 480  # HVGA 해상도 기준

    with concurrent.futures.ThreadPoolExecutor() as executor:
        is_moving = False
        while True:
            try:
                # 아두이노 신호 확인
                arduino_signal = check_arduino_signal()
                if arduino_signal:
                    if arduino_signal == 'moving':
                        is_moving = True
                        print("Moving signal received, pausing detection.")
                    elif arduino_signal == 'find':
                        is_moving = False
                        print("Find signal received, resuming detection.")
                    elif arduino_signal == 'check' and not is_moving:
                        # 병렬 처리로 프레임을 처리
                        future = executor.submit(process_frames, left_cam_url, right_cam_url, model, fl, tantheta, img_width)
                        result = future.result()

                        if result is None:
                            continue

                        img1, labels1, boxes1, distances, disparities = result

                        # 현재 시간 출력
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Time: {current_time}")

                        # 'bottle'이 탐지되었는지 확인하고 거리 전송
                        for label in labels1:
                            if label in distances:
                                category = CLASS_MAP.get(int(label), "Unknown")
                                distance = distances.get(label, "Unknown")
                                if category == TARGET_CATEGORY and distance > MIN_DISTANCE_THRESHOLD:
                                    send_distance_to_arduino(distance)
                                    break  # 하나의 객체만 처리

                        # 이미지에 주석 추가
                        img1_annotated = annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)

                        # 최종 이미지 파이썬 창에 띄우기
                        cv2.imshow("Annotated Image", img1_annotated)

                        # 초당 30프레임 설정 (33ms 대기)
                        if cv2.waitKey(33) & 0xFF == ord('q'):
                            break

            except KeyError as e:
                print(f"KeyError: {e} - Skipping this frame.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
