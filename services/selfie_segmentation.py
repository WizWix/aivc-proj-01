import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
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
        except Exception as e:
            print(f"다운로드 실패: {e}")
            # Don't exit here, let the caller handle it
    return filename

from PIL import Image, ImageOps
import io

def fix_image_orientation(image_path):
    """
    Corrects image orientation based on EXIF data.
    """
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        # Convert to BGR for OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"Warning: Failed to fix orientation: {e}")
        return cv2.imread(image_path)

def process_selfie(image, blur_intensity=21):
    """
    이미지에서 사람을 분리하고 배경을 흐릿하게 만듭니다.
    blur_intensity: 흐림 정도 (홀수)
    """
    # Ensure blur_intensity is odd
    if blur_intensity % 2 == 0:
        blur_intensity += 1

    # Optional: Resize large images to improve model consistency
    h, w = image.shape[:2]
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    model_path = os.path.join("models", "selfie_segmenter.tflite")
    if not os.path.exists(model_path):
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        os.makedirs("models", exist_ok=True)
        download_file(model_url, model_path, "셀피 세그멘테이션 모델")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    # MediaPipe Segmenter 설정
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                          output_confidence_masks=True)
    
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Convert OpenCV BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 세그멘테이션 수행
        segmentation_result = segmenter.segment(mp_image)
        
        # Get the confidence mask (typically index 0 for person in selfie segmenter)
        confidence_mask = segmentation_result.confidence_masks[0].numpy_view()
        
        # Ensure mask is 2D (squeeze if (H, W, 1) or similar)
        if len(confidence_mask.shape) > 2:
            confidence_mask = np.squeeze(confidence_mask)
            
        # Create a 3-channel mask for easy multiplication
        mask = np.stack((confidence_mask,) * 3, axis=-1)
        
        # 배경 흐림 처리
        bg_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
        
        # 알파 블렌딩 (Alpha Blending)을 사용한 합성
        # Ensure mask and images have compatible shapes (H, W, 3)
        output_image = (image.astype(float) * mask + bg_image.astype(float) * (1.0 - mask)).astype(np.uint8)
        
        return output_image

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MediaPipe를 이용한 셀피 세그멘테이션 테스트')
    parser.add_argument('--image', type=str, help='테스트할 입력 이미지 경로')
    parser.add_argument('--blur', type=int, default=21, help='배경 흐림 강도 (기본값: 21)')
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("입력 이미지가 제공되지 않아 기본 이미지를 사용합니다.")
        image_url = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=600&auto=format&fit=crop"
        os.makedirs("images", exist_ok=True)
        image_path = download_file(image_url, os.path.join("images", "default_selfie.jpg"), "기본 이미지")

    image = cv2.imread(image_path)
    if image is None:
        print("오류: 이미지를 읽을 수 없습니다.")
        return

    try:
        result = process_selfie(image, args.blur)
        output_path = os.path.join("images", "result_selfie.jpg")
        cv2.imwrite(output_path, result)
        print(f"결과가 저장되었습니다: {output_path}")
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")

if __name__ == '__main__':
    main()
