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
            sys.exit(1)
        except Exception as e:
            print(f"다운로드 실패: {e}")
            sys.exit(1)
    return filename

def draw_boxes(image, detection_result):
    """
    OpenCV를 사용하여 감지된 각 객체 주변에 사각형 테두리와 라벨을 그립니다.
    """
    for detection in detection_result.detections:
        # Bounding box 그리기
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

        # 라벨과 스코어 그리기
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (bbox.origin_x + 8, bbox.origin_y + 24)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 0, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='MediaPipe를 이용한 객체 탐지 테스트 (Tasks API)')
    parser.add_argument('--image', type=str, help='테스트할 입력 이미지 경로')
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("입력 이미지가 제공되지 않아 기본 이미지를 사용합니다.")
        image_url = "https://images.unsplash.com/photo-1517841905240-472988babdf9?q=80&w=600&auto=format&fit=crop"
        os.makedirs("images", exist_ok=True)
        image_path = download_file(image_url, os.path.join("images", "default_object_image.jpg"), "기본 이미지")

    model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
    os.makedirs("models", exist_ok=True)
    model_path = download_file(model_url, os.path.join("models", "efficientdet_lite0.tflite"), "객체 탐지 모델 파일")

    image = cv2.imread(image_path)
    if image is None:
        print("오류: 이미지를 읽을 수 없습니다.")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)

    annotated_image = image.copy()
    draw_boxes(annotated_image, detection_result)

    os.makedirs("images", exist_ok=True)
    output_filename = os.path.join("images", "result_object.jpg")
    cv2.imwrite(output_filename, annotated_image)
    print(f"시각화 결과가 저장되었습니다: {output_filename}")

if __name__ == '__main__':
    main()
