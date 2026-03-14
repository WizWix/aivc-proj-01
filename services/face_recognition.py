import argparse
import os
import sys
import cv2
import numpy as np
import urllib.request

# 모델 다운로드 URL
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"다운로드 중: {filename} ... ({url})")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("다운로드 완료.")
        except Exception as e:
            print(f"다운로드 실패: {e}")
            sys.exit(1)

def resize_image(image, max_size=800):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    return image

def detect_face(detector, image):
    """
    주어진 이미지에서 얼굴을 검출하여 반환
    """
    height, width, _ = image.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    # 가장 점수가 높은(확실한) 첫 번째 얼굴 반환
    return faces[0]

def extract_feature(recognizer, image, face):
    """
    정렬된 얼굴에서 특징 벡터(Feature) 추출
    """
    # 얼굴 정렬 (Align)
    aligned_face = recognizer.alignCrop(image, face)
    # 특징 추출
    feature = recognizer.feature(aligned_face)
    return feature

def main():
    parser = argparse.ArgumentParser(description='OpenCV SFace 기반 얼굴 인식 유사도 테스트 스크립트')
    parser.add_argument('--img1', type=str, required=True, help='첫 번째 이미지 경로')
    parser.add_argument('--img2', type=str, required=True, help='두 번째 이미지 경로')
    # 코사인 임계값은 보통 0.363 이상이면 동일인. 유클리디안 거리는 작을수록 동일인(임계값 1.128)
    parser.add_argument('--threshold', type=float, default=0.363, 
                        help='유사도(코사인) 거리 임계값. 기본값: 0.363. (이 값보다 크면 동일인물)')
    args = parser.parse_args()

    # 이미지 존재 여부 확인
    for path in [args.img1, args.img2]:
        if not os.path.exists(path):
            print(f"오류: 이미지를 찾을 수 없습니다: {path}")
            sys.exit(1)

    print("OpenCV 모델 파일 확인 및 다운로드...")
    os.makedirs("models", exist_ok=True)
    yunet_model = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_model = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_file(YUNET_URL, yunet_model)
    download_file(SFACE_URL, sface_model)

    print("얼굴 검출기 및 인식기 초기화 중...")
    detector = cv2.FaceDetectorYN.create(
        model=yunet_model,
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(
        model=sface_model,
        config=""
    )

    print("\n이미지 로드 중...")
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    
    if img1 is None or img2 is None:
        print("이미지를 읽어오는데 실패했습니다. 올바른 포맷인지 확인해 주세요.")
        sys.exit(1)

    img1 = resize_image(img1)
    img2 = resize_image(img2)

    print(f"첫 번째 이미지({args.img1}) 조정된 크기: {img1.shape}")
    print(f"두 번째 이미지({args.img2}) 조정된 크기: {img2.shape}")

    print("첫 번째 이미지에서 얼굴 검출 중...")
    face1 = detect_face(detector, img1)
    if face1 is None:
        print(f"오류: {args.img1} 에서 얼굴을 검출하지 못했습니다.")
        sys.exit(1)

    print("두 번째 이미지에서 얼굴 검출 중...")
    face2 = detect_face(detector, img2)
    if face2 is None:
        print(f"오류: {args.img2} 에서 얼굴을 검출하지 못했습니다.")
        sys.exit(1)

    print("각 얼굴의 특징 추출 중...")
    feat1 = extract_feature(recognizer, img1, face1)
    feat2 = extract_feature(recognizer, img2, face2)

    print("유사도 계산 중...")
    # 코사인 유사도 (값이 높을수록 비슷함, 1.0 = 동일)
    cosine_similarity = recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
    
    is_same = cosine_similarity >= args.threshold

    print("\n================ 결과 ================")
    print(f"동일인물 여부: {'예 (True)' if is_same else '아니오 (False)'}")
    print(f"코사인 유사도 (Cosine Similarity): {cosine_similarity:.4f}")
    print(f"임계값 (Threshold): {args.threshold:.4f} 이상일 때 동일인물 판단")
    print("======================================")


if __name__ == '__main__':
    main()
