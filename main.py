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
from pydantic import BaseModel

# Import AI services
from services import ocr, face_recognition, image_classification, pose_estimation, object_detection, selfie_segmentation, sentiment_analysis

app = FastAPI(
    title="AI 서비스 허브",
    description="최첨단 AI 모델들을 한곳에서 체험해보세요.",
    version="1.1.0"
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def resize_image(image, max_size=800):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    return image

# --- OCR Sub-app ---
ocr_app = FastAPI(title="AI OCR Service")
@ocr_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_ocr_root(request: Request):
    return templates.TemplateResponse("ocr_index.html", {"request": request})

@ocr_app.post("/api/process")
async def run_ocr(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        results = ocr.extract_text(tmp_path)
        serializable_results = [{"text": t, "confidence": float(p), "bbox": [[int(c) for c in pt] for pt in b]} for (b, t, p) in results]
        return {"results": serializable_results, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Face Recognition Sub-app ---
face_app = FastAPI(title="AI Face Recognition Service")
@face_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_face_root(request: Request):
    return templates.TemplateResponse("face_index.html", {"request": request})

@face_app.post("/api/process")
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
        im1, im2 = cv2.imread(path1), cv2.imread(path2)
        if im1 is None or im2 is None: return {"error": "이미지를 읽을 수 없습니다.", "similarity": 0.0, "is_same": False}
        im1, im2 = resize_image(im1), resize_image(im2)
        face1 = face_recognition.detect_face(detector, im1)
        face2 = face_recognition.detect_face(detector, im2)
        if face1 is None or face2 is None: return {"error": "얼굴을 검출하지 못했습니다.", "similarity": 0.0, "is_same": False}
        feat1 = face_recognition.extract_feature(recognizer, im1, face1)
        feat2 = face_recognition.extract_feature(recognizer, im2, face2)
        sim = recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        return {"similarity": float(sim), "is_same": bool(sim >= 0.363), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [path1, path2]:
            if os.path.exists(p): os.remove(p)

# --- Image Classification Sub-app ---
class_app = FastAPI(title="AI Image Classification Service")
@class_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_class_root(request: Request):
    return templates.TemplateResponse("class_index.html", {"request": request})

@class_app.post("/api/process")
async def run_classification(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        import torch
        from torchvision import models
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
            pred = torch.argmax(output, dim=1).item()
            label = weights.meta['categories'][pred]
        return {"label": label, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Pose Estimation Sub-app ---
pose_app = FastAPI(title="AI Pose Estimation Service")
@pose_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_pose_root(request: Request):
    return templates.TemplateResponse("pose_index.html", {"request": request})

@pose_app.post("/api/process")
async def run_pose(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = os.path.join("models", "pose_landmarker_lite.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        detector = vision.PoseLandmarker.create_from_options(options)
        image = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = detector.detect(mp_image)
        ann = image.copy()
        if res.pose_landmarks:
            for lm in res.pose_landmarks: pose_estimation.draw_landmarks(ann, lm)
        _, buf = cv2.imencode('.png', ann)
        img_str = base64.b64encode(buf).decode('utf-8')
        return {"result_image": f"data:image/png;base64,{img_str}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Object Detection Sub-app ---
obj_app = FastAPI(title="AI Object Detection Service")
@obj_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_obj_root(request: Request):
    return templates.TemplateResponse("obj_index.html", {"request": request})

@obj_app.post("/api/process")
async def run_obj(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = os.path.join("models", "efficientdet_lite0.tflite")
        if not os.path.exists(model_path):
             object_detection.download_file("https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite", model_path, "Object Detection Model")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
        detector = vision.ObjectDetector.create_from_options(options)
        image = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = detector.detect(mp_image)
        ann = image.copy()
        object_detection.draw_boxes(ann, res)
        _, buf = cv2.imencode('.png', ann)
        img_str = base64.b64encode(buf).decode('utf-8')
        return {"result_image": f"data:image/png;base64,{img_str}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Selfie Segmentation Sub-app ---
selfie_app = FastAPI(title="AI Selfie Segmentation Service")
@selfie_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_selfie_root(request: Request):
    return templates.TemplateResponse("selfie_index.html", {"request": request})

@selfie_app.post("/api/process")
async def run_selfie(file: UploadFile = File(...), blur_intensity: int = 21):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        image = selfie_segmentation.fix_image_orientation(tmp_path)
        if image is None: raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
        processed = selfie_segmentation.process_selfie(image, blur_intensity)
        _, buf = cv2.imencode('.png', processed)
        img_str = base64.b64encode(buf).decode('utf-8')
        return {"result_image": f"data:image/png;base64,{img_str}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- Sentiment Analysis Sub-app ---
sentiment_app = FastAPI(title="AI Sentiment Analysis Service")
class SentimentRequest(BaseModel): text: str
@sentiment_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_sentiment_root(request: Request):
    return templates.TemplateResponse("sentiment_index.html", {"request": request})

@sentiment_app.post("/api/process")
async def run_sentiment(request: SentimentRequest):
    try:
        return sentiment_analysis.analyze_sentiment(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount All Sub-apps
app.mount("/ocr", ocr_app)
app.mount("/face-recognition", face_app)
app.mount("/image-classification", class_app)
app.mount("/pose-estimation", pose_app)
app.mount("/object-detection", obj_app)
app.mount("/selfie-segmentation", selfie_app)
app.mount("/sentiment-analysis", sentiment_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
