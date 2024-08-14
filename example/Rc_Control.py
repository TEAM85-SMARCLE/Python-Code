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

from capture import capture_frame_from_esp32  # 프레임 캡처 관련 모듈 임포트
from processing import process_frames  # 프레임 처리 관련 모듈 임포트
from visualization import annotate_image_with_distances  # 이미지에 거리 주석 추가하는 모듈 임포트
from class_map import CLASS_MAP  # 클래스 매핑을 위한 모듈 임포트
import model as yolo_model  # YOLO 모델 관련 모듈 임포트

# pyserial 모듈 임포트
try:
    import serial  # 시리얼 통신 모듈 임포트
except ImportError:
    print("pyserial 모듈이 설치되지 않았습니다. 'pip install pyserial' 명령어로 설치하세요.")
    sys.exit(1)  # pyserial 모듈이 없으면 프로그램 종료

# 모든 경고 무시
warnings.filterwarnings("ignore")  # 경고 메시지 무시 설정

# Arduino 시리얼 포트 설정
try:
    arduino = serial.Serial("/dev/cu.usbmodem11301", 9600, timeout=.1)  # 아두이노 시리얼 포트와 보드레이트 설정
except AttributeError:
    print("serial 모듈에서 'Serial' 속성을 찾을 수 없습니다. 모듈이 제대로 설치되었는지 확인하세요.")
    sys.exit(1)  # 시리얼 포트 설정 실패 시 프로그램 종료

# 탐지할 객체의 초기 카테고리
TARGET_CATEGORY = 'bottle'  # 탐지할 객체 카테고리 설정 (병)

# 거리 임계값 설정 (단위: cm)
MIN_DISTANCE_THRESHOLD = 10  # 10cm 이하의 거리는 무시

def send_distance_to_arduino(distance):
    """
    아두이노에 거리를 전송하는 함수
    """
    if distance > MIN_DISTANCE_THRESHOLD:  # 거리가 임계값보다 클 경우
        arduino.write(bytes(f"{distance}\n", 'utf-8'))  # 아두이노로 거리 전송
        print(f"Distance sent to Arduino: {distance} cm")  # 전송된 거리 출력

def check_arduino_signal():
    """
    아두이노에서 신호를 받는 함수
    """
    if arduino.in_waiting > 0:  # 아두이노에서 대기 중인 데이터가 있으면
        return arduino.readline().decode('utf-8').strip()  # 데이터를 읽어서 반환
    return None  # 대기 중인 데이터가 없으면 None 반환

def main():
    # ESP32 카메라 주소 (해상도 QVGA)
    left_cam_url = "http://192.168.0.14/capture"  # 왼쪽 카메라 주소
    right_cam_url = "http://192.168.0.13/capture"  # 오른쪽 카메라 주소

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/yolov5s.pt'  # YOLO 모델 가중치 파일 경로
    model = yolo_model.load_model(weights_path)  # YOLO 모델 로드

    # 스테레오 비전 설정
    fl = 2.043636363636363  # 스테레오 비전의 초점 거리 설정
    tantheta = 0.5443642625  # 스테레오 비전의 각도 설정

    # img_width = 640  # VGA 해상도 기준 (주석 처리)
    img_width = 480  # HVGA 해상도 기준

    with concurrent.futures.ThreadPoolExecutor() as executor:  # 스레드 풀을 사용한 병렬 처리 설정
        is_moving = False  # 이동 상태를 나타내는 플래그 변수 초기화
        while True:
            try:
                # 아두이노 신호 확인
                arduino_signal = check_arduino_signal()  # 아두이노 신호 체크
                if arduino_signal:  # 신호가 있을 경우
                    if arduino_signal == 'moving':  # 이동 중 신호
                        is_moving = True  # 이동 상태로 설정
                        print("Moving signal received, pausing detection.")  # 이동 중임을 출력
                    elif arduino_signal == 'find':  # 탐지 신호
                        is_moving = False  # 탐지 상태로 설정
                        print("Find signal received, resuming detection.")  # 탐지 상태로 전환
                    elif arduino_signal == 'check' and not is_moving:  # 탐지 신호가 'check'이고 이동 중이 아닐 때
                        # 병렬 처리로 프레임을 처리
                        future = executor.submit(process_frames, left_cam_url, right_cam_url, model, fl, tantheta, img_width)  # 프레임 처리 비동기 실행
                        result = future.result()  # 결과 가져오기

                        if result is None:  # 결과가 없을 경우
                            continue  # 다음 반복으로 넘어감

                        img1, labels1, boxes1, distances, disparities = result  # 결과 분해

                        # 현재 시간 출력
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간 포맷팅
                        print(f"Time: {current_time}")  # 현재 시간 출력

                        # 'bottle'이 탐지되었는지 확인하고 거리 전송
                        for label in labels1:  # 모든 탐지된 레이블에 대해
                            if label in distances:  # 해당 레이블에 대한 거리가 있을 경우
                                category = CLASS_MAP.get(int(label), "Unknown")  # 클래스 매핑
                                distance = distances.get(label, "Unknown")  # 거리 가져오기
                                if category == TARGET_CATEGORY and distance > MIN_DISTANCE_THRESHOLD:  # 타겟 카테고리이고 거리가 임계값 이상일 경우
                                    send_distance_to_arduino(distance)  # 아두이노에 거리 전송
                                    break  # 하나의 객체만 처리하고 루프 종료

                        # 이미지에 주석 추가
                        img1_annotated = annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)  # 이미지에 거리 주석 추가

                        # 최종 이미지 파이썬 창에 띄우기
                        cv2.imshow("Annotated Image", img1_annotated)  # 주석이 추가된 이미지 창에 띄우기

                        # 초당 30프레임 설정 (33ms 대기)
                        if cv2.waitKey(33) & 0xFF == ord('q'):  # 'q'를 누르면 루프 종료
                            break

            except KeyError as e:  # KeyError 발생 시
                print(f"KeyError: {e} - Skipping this frame.")  # 에러 메시지 출력하고 해당 프레임 건너뜀
                continue
            except Exception as e:  # 기타 예외 발생 시
                print(f"An unexpected error occurred: {e}")  # 예외 메시지 출력하고 루프 종료
                break

    cv2.destroyAllWindows()  # 모든 윈도우 창 닫기

if __name__ == '__main__':
    main()  # 메인 함수 실행
