# 🤖 AI 서비스 허브 (AI Service Hub)

최첨단 인공지능 모델들을 한곳에서 간편하게 체험해볼 수 있는 통합 AI 모델 허브입니다. FastAPI 기반의 웹 인터페이스를 통해 다양한 이미지 처리 및 자연어 처리 인공지능 기능을 제공합니다.

---

## 🌟 주요 기능

이 프로젝트는 다음과 같은 7가지 핵심 AI 서비스를 제공합니다:

1.  **감정 분석 (Sentiment Analysis)**: KcELECTRA 및 RoBERTa 모델을 활용하여 한국어/영어 텍스트의 감정을 다각도로 분석합니다.
2.  **셀피 세그멘테이션 (Selfie Segmentation)**: MediaPipe를 사용하여 인물과 배경을 분리하고, 배경을 실시간으로 흐릿하게(Blur) 처리합니다.
3.  **텍스트 추출 (OCR)**: EasyOCR 엔진을 사용하여 이미지 내의 다양한 텍스트를 인식하고 디지털 텍스트로 변환합니다.
4.  **얼굴 인식 및 유사도 측정 (Face Recognition)**: OpenCV SFace 모델을 사용하여 두 장의 이미지 속 인물이 동일인인지 판별하고 유사도를 측정합니다.
5.  **이미지 분류 (Image Classification)**: MobileNet V3 모델을 사용하여 이미지 속에 어떤 객체가 있는지 카테고리별로 분류합니다.
6.  **객체 탐지 (Object Detection)**: MediaPipe Object Detector를 통해 이미지 내 여러 사물의 위치를 사각형 박스로 표시하고 이름을 탐지합니다.
7.  **포즈 추정 (Pose Estimation)**: MediaPipe Pose Landmarker로 사람의 골격(관절) 위치를 파악하여 동작을 분석합니다.

---

## 🛠 기술 스택

-   **Backend**: FastAPI, Python 3.x
-   **Frontend**: Jinja2 Templates, Vanilla CSS, JavaScript
-   **AI Libraries**:
    -   OpenCV (얼굴 인식, 이미지 처리)
    -   MediaPipe (포즈 추정, 객체 탐지, 세그멘테이션)
    -   PyTorch & Torchvision (이미지 분류)
    -   Transformers (감정 분석)
    -   EasyOCR (텍스트 추출)
-   **Server**: Uvicorn
-   **Deployment**: Docker, Docker Compose

---

## 🚀 시작하기

### 1. 환경 설정 및 설치

먼저 레포지토리를 클론하고 가상환경을 설정합니다.

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 가상환경 활성화 (Linux/macOS)
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

```bash
python main.py
```

서버가 실행되면 웹 브라우저에서 `http://localhost:8000`으로 접속하여 서비스를 이용할 수 있습니다.

---

## 🐳 Docker 사용하기

Docker를 사용하여 쉽고 빠르게 실행 환경을 구축할 수 있습니다.

### Docker Compose로 실행

```bash
docker-compose up --build
```

실행 후 `http://localhost:8000`에서 접속 가능합니다.

---

## 📂 프로젝트 구조

```text
.
├── main.py              # FastAPI 메인 애플리케이션 및 라우팅
├── services/            # 각 AI 서비스 로직 (OCR, Face, Pose 등)
├── models/              # AI 모델 파일 저장 경로 (.onnx, .tflite, .task 등)
├── templates/           # HTML 템플릿 파일 (Jinja2)
├── static/              # CSS, JS 및 정적 리소스
├── scripts/             # 관리 및 유틸리티 스크립트
├── Dockerfile           # 컨테이너 빌드 설정
└── requirements.txt     # 프로젝트 의존성 라이브러리
```

---

## 📜 라이선스

이 프로젝트의 각 서비스에서 활용하는 오픈소스 모델들의 라이선스는 해당 라이브러리(MediaPipe, EasyOCR, OpenCV 등)의 관련 규정을 따릅니다.
