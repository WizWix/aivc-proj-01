import os
import shutil
import tempfile
import base64
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np

# Import AI services (Assuming they are in the 'services' directory)
from services import ocr, face_recognition, image_classification, pose_estimation, object_detection, selfie_segmentation

app = FastAPI(
    title="AI 서비스 허브",
    description="OCR, 얼굴 인식, 이미지 분류, 포즈 추정 등 다양한 AI 서비스를 제공하는 통합 플랫폼입니다.",
    version="1.0.0"
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/ocr", summary="텍스트 추출 (OCR)", description="이미지에서 한국어 및 영어 텍스트를 추출합니다.")
async def run_ocr(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = ocr.extract_text(tmp_path)
        serializable_results = []
        for (bbox, text, prob) in results:
            serializable_results.append({
                "text": text,
                "confidence": float(prob),
                "bbox": [[int(coord) for coord in point] for point in bbox]
            })
        return {"results": serializable_results, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def resize_image(image, max_size=800):
    """
    Resizes an image if its dimensions exceed max_size.
    Matches the logic used in face_recognition.py.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    return image

@app.post("/api/face-recognition", summary="얼굴 인식 유사도", description="두 이미지 속 인물의 얼굴 유사도를 측정합니다.")
async def run_face_recognition(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2:
        shutil.copyfileobj(img1.file, tmp1)
        shutil.copyfileobj(img2.file, tmp2)
        path1, path2 = tmp1.name, tmp2.name

    try:
        os.makedirs("models", exist_ok=True)
        yunet_model = os.path.join("models", "face_detection_yunet_2023mar.onnx")
        sface_model = os.path.join("models", "face_recognition_sface_2021dec.onnx")
        
        detector = cv2.FaceDetectorYN.create(yunet_model, "", (320, 320), 0.6, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(sface_model, "")

        im1 = cv2.imread(path1)
        im2 = cv2.imread(path2)
        
        if im1 is None or im2 is None:
             return {"error": "이미지 파일을 읽을 수 없습니다.", "similarity": 0.0, "is_same": False}

        # Apply resizing to improve detection reliability for large images
        im1 = resize_image(im1)
        im2 = resize_image(im2)
        
        face1 = face_recognition.detect_face(detector, im1)
        face2 = face_recognition.detect_face(detector, im2)
        
        if face1 is None or face2 is None:
            return {"error": "이미지 중 하나에서 얼굴을 검출하지 못했습니다.", "similarity": 0.0, "is_same": False}

        feat1 = face_recognition.extract_feature(recognizer, im1, face1)
        feat2 = face_recognition.extract_feature(recognizer, im2, face2)

        cosine_similarity = recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return {
            "similarity": float(cosine_similarity),
            "is_same": bool(cosine_similarity >= 0.363),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [path1, path2]:
            if os.path.exists(p):
                os.remove(p)

@app.post("/api/image-classification", summary="이미지 분류", description="이미지 속 객체를 분류합니다 (MobileNet V3 Small 사용).")
async def run_classification(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import torch
        from torchvision import models
        import cv2

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights).to(device)
        model.eval()

        img = cv2.imread(tmp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            label = weights.meta['categories'][predicted_class]
            
        return {
            "predicted_class_index": predicted_class,
            "label": label,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/pose-estimation", summary="포즈 추정", description="사람의 포즈와 랜드마크를 감지합니다 (MediaPipe 사용).")
async def run_pose_estimation(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = os.path.join("models", "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
             return {"error": "모델 파일이 없습니다. 먼저 원본 스크립트를 실행하여 다운로드하세요."}

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        detector = vision.PoseLandmarker.create_from_options(options)

        image = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = detector.detect(mp_image)
        
        # Visualization
        annotated_image = image.copy()
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                pose_estimation.draw_landmarks(annotated_image, pose_landmarks)
        
        # Convert to base64 to show as image in the UI
        _, buffer = cv2.imencode('.png', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        num_poses = len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0
        return {
            "detected_poses": num_poses,
            "result_image": f"data:image/png;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/object-detection", summary="객체 탐지", description="이미지 속 객체를 탐지하고 테두리를 표시합니다 (MediaPipe EfficientDet 사용).")
async def run_object_detection(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = os.path.join("models", "efficientdet_lite0.tflite")
        if not os.path.exists(model_path):
            # Proactively download model if missing
            url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
            os.makedirs("models", exist_ok=True)
            object_detection.download_file(url, model_path, "객체 탐지 모델 파일")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
        detector = vision.ObjectDetector.create_from_options(options)

        image = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = detector.detect(mp_image)
        
        # Visualization
        annotated_image = image.copy()
        object_detection.draw_boxes(annotated_image, detection_result)
        
        # Convert to base64 to show as image in the UI
        _, buffer = cv2.imencode('.png', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        num_detections = len(detection_result.detections)
        return {
            "detected_objects": num_detections,
            "result_image": f"data:image/png;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Selfie Segmentation Sub-app ---
selfie_app = FastAPI(title="AI Selfie Segmentation Service")

# Note: Templates are shared from the main app's directory
@selfie_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_selfie_root(request: Request):
    return templates.TemplateResponse("selfie_index.html", {"request": request})

@selfie_app.post("/api/process", summary="셀피 배경 흐림", description="인물을 감지하고 배경을 흐릿하게 처리합니다.")
async def run_selfie_segmentation(file: UploadFile = File(...), blur_intensity: int = 21):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Use the orientation fix instead of just cv2.imread
        image = selfie_segmentation.fix_image_orientation(tmp_path)
        
        if image is None:
             raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다.")
             
        processed_image = selfie_segmentation.process_selfie(image, blur_intensity)
        
        _, buffer = cv2.imencode('.png', processed_image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {
            "result_image": f"data:image/png;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Mount the sub-app
app.mount("/selfie-segmentation", selfie_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
