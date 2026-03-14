import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import urllib.request
import os
import sys

def download_file(url, filename, desc):
    if not os.path.exists(filename):
        print(f"{desc} 다운로드하는 중... ({url})")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            print("다운로드 완료.")
        except urllib.error.HTTPError as e:
            print(f"다운로드 실패 (HTTP Error): {e}")
            if e.code == 429:
                print("너무 많은 요청으로 이미지를 다운로드할 수 없습니다. 다른 이미지를 사용해 주세요.")
            sys.exit(1)
        except Exception as e:
            print(f"다운로드 실패: {e}")
            sys.exit(1)
    return filename

def draw_landmarks(image, landmarks):
    """
    OpenCV를 사용해 직접 MediaPipe 랜드마크를 그립니다. (tasks API에서는 solutions.drawing_utils를 대체함)
    """
    h, w, c = image.shape
    # MediaPipe Pose 연결선 정의
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    # 1. 뼈대(선) 그리기
    for connection in connections:
        p1_idx, p2_idx = connection
        if p1_idx < len(landmarks) and p2_idx < len(landmarks):
            l1 = landmarks[p1_idx]
            l2 = landmarks[p2_idx]
            if getattr(l1, 'presence', 1.0) > 0.3 and getattr(l2, 'presence', 1.0) > 0.3:
                pt1 = (int(l1.x * w), int(l1.y * h))
                pt2 = (int(l2.x * w), int(l2.y * h))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # 2. 관절(점) 그리기
    for idx, landmark in enumerate(landmarks):
        if getattr(landmark, 'presence', 1.0) > 0.3:
            pt = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(image, pt, 5, (0, 0, 255), -1)

def main():
    parser = argparse.ArgumentParser(description='MediaPipe를 이용한 가벼운 포즈 추정 테스트 (Tasks API)')
    parser.add_argument('--image', type=str, help='테스트할 입력 이미지 경로 (입력하지 않으면 기본 이미지 자동 다운로드)')
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("입력 이미지가 제공되지 않아 기본 이미지를 사용합니다.")
        # Wikimedia Commons의 안정적인 다른 이미지로 변경 (429 에러 방지)
        image_url = "https://images.unsplash.com/photo-1552674605-15cff24c6018?q=80&w=600&auto=format&fit=crop"
        os.makedirs("images", exist_ok=True)
        image_path = download_file(image_url, os.path.join("images", "default_pose_image.jpg"), "기본 이미지")

    if not os.path.exists(image_path):
        print(f"오류: 이미지를 찾을 수 없습니다. 경로: {image_path}")
        return

    # 모델 가중치 파일 다운로드 (가벼운 모델 - pose_landmarker_lite.task)
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    os.makedirs("models", exist_ok=True)
    model_path = download_file(model_url, os.path.join("models", "pose_landmarker_lite.task"), "포즈 추정 모델 파일(가벼운 모델)")

    print(f"이미지 처리 중...: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("오류: 이미지를 읽을 수 없습니다. (지원되지 않는 이미지 포맷이거나 파일이 깨졌을 수 있습니다.)")
        return

    # MediaPipe Pose Landmarker 초기화 (Tasks API)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    
    try:
        detector = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        print(f"모델을 초기화하는 중 오류가 발생했습니다: {e}")
        return

    # OpenCV 이미지를 MediaPipe 포맷으로 변환 (BGR -> RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # 포즈 추정 수행
    detection_result = detector.detect(mp_image)

    # 결과 시각화
    annotated_image = image.copy()
    if detection_result.pose_landmarks:
        print(f"포즈 추출 성공! {len(detection_result.pose_landmarks)} 명의 사람이 탐지되었습니다.")
        # 모든 탐지된 사람의 랜드마크 그리기
        for pose_landmarks in detection_result.pose_landmarks:
            draw_landmarks(annotated_image, pose_landmarks)
    else:
        print("포즈를 찾을 수 없습니다.")

    # 결과 저장
    os.makedirs("images", exist_ok=True)
    output_filename = os.path.join("images", "result_pose.jpg")
    cv2.imwrite(output_filename, annotated_image)
    print(f"시각화 결과가 저장되었습니다: {output_filename}")

    # 결과 화면에 출력 (API 형식 사용을 위해 창 띄우기는 생략하고 파일로만 저장됨)

if __name__ == '__main__':
    main()
