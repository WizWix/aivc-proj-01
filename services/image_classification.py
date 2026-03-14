import torch
import torchvision.models as models
import time

def test_lightweight_model():
    # 1. 사용할 디바이스 확인 (GPU가 있으면 CUDA, 없으면 CPU 사용)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 디바이스: {device}")
    
    if device.type != 'cuda':
        print("경고: CUDA를 사용할 수 없어 CPU로 실행됩니다. PyTorch가 CUDA를 지원하는 버전인지 확인하세요.")

    # 2. 가벼운 대표적 이미지 분류 모델 불러오기 (MobileNet V3 Small)
    # 파라미터 수가 적고 연산량이 적어 매우 가볍고 빠릅니다.
    print("MobileNet V3 Small 모델을 불러오는 중...")
    try:
        # 최신 torchvision 버전 권장 방식
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except AttributeError:
        # 구버전 torchvision 호환용
        model = models.mobilenet_v3_small(pretrained=True)
        
    # 모델을 선택한 디바이스(GPU/CPU)로 이동
    model = model.to(device)
    # 모델을 평가(추론) 모드로 설정 (Dropout 및 BatchNorm 비활성화)
    model.eval()

    # 3. 임의의 입력 데이터(더미 데이터) 생성
    # 형태: (배치 크기=1, 채널 수=3(RGB), 높이=224, 너비=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 4. GPU 워밍업
    # 처음 추론할 때는 초기화 작업 등으로 인해 시간이 더 걸리므로, 시간을 재기 전에 몇 번 미리 실행해 줍니다.
    print("추론 워밍업 중...")
    with torch.no_grad(): # 추론 시에는 기울기(gradient) 계산을 비활성화하여 메모리와 연산 속도 최적화
        for _ in range(5):
            _ = model(dummy_input)

    # 5. 본격적인 추론 속도 측정
    print("평균 추론 속도 측정 중...")
    num_tests = 100
    
    # GPU를 사용하는 경우 확실하게 측정을 위해 동기화
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_tests):
            output = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # 밀리초(ms) 단위로 변환하여 평균 속도 계산
    avg_inference_time = (end_time - start_time) / num_tests * 1000 

    print("=" * 40)
    print(f"테스트 모델: MobileNet V3 Small")
    print(f"평균 추론 시간 ({num_tests}회): {avg_inference_time:.2f} ms")
    
    # 6. 결과 텐서 확인 (결과는 ImageNet 1000개 클래스에 대한 로짓 값)
    print(f"출력 텐서 형태: {output.shape}")
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"더미 데이터 예측 클래스 인덱스: {predicted_class}")
    print("=" * 40)

if __name__ == "__main__":
    test_lightweight_model()
