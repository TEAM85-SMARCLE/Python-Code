import cv2
import requests
import numpy as np
import model as yolo_model
import detection as det
import distance_calculation as dist_calc
import visualization as vis
from class_map import CLASS_MAP
from datetime import datetime
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")

def capture_frame_from_esp32(cam_url):
    """
    ESP32 카메라 서버에서 프레임을 캡처하여 반환합니다.
    """
    img_resp = requests.get(cam_url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    return img

def main():
    # ESP32 카메라 주소
    left_cam_url = "http://192.168.0.14/capture"
    right_cam_url = "http://192.168.0.13/capture"

    # YOLOv5 모델 로드
    weights_path = '/Users/apple/Desktop/Python/yolov5s.pt'
    model = yolo_model.load_model(weights_path)

    # 스테레오 비전 설정
    fl = 2.043636363636363
    tantheta = 0.5443642625
    img_width = 800  # SVGA 해상도 기준

    while True:
        try:
            # ESP32에서 이미지 캡처
            img1 = capture_frame_from_esp32(left_cam_url)
            img2 = capture_frame_from_esp32(right_cam_url)

            if img1 is None or img2 is None:
                print("Failed to capture images from ESP32 cameras.")
                continue

            # 객체 탐지
            results1 = det.detect_objects(model, img1)
            results2 = det.detect_objects(model, img2)

            # 바운더리 박스와 라벨 추출
            labels1, boxes1 = det.get_bounding_boxes(results1)
            labels2, boxes2 = det.get_bounding_boxes(results2)

            # 거리 및 시차 계산
            distances, disparities = dist_calc.compute_distances_and_disparity(labels1, boxes1, labels2, boxes2, fl, tantheta, img_width)

            # 현재 시간 출력
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {current_time}")

            # 시차 및 거리 출력
            for label in disparities:
                category = CLASS_MAP.get(int(label), "Unknown")
                distance = distances.get(label, "Unknown")
                print(f"Category: {category}, Distance: {distance:.2f} meters")

            print("-"*50)

            # 이미지에 주석 추가
            img1_annotated = vis.annotate_image_with_distances(img1, labels1, boxes1, distances, CLASS_MAP)

            # 최종 이미지 파이썬 창에 띄우기
            cv2.imshow("Annotated Image", img1_annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
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
