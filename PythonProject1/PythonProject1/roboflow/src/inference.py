# inference.py

from ultralytics import YOLO

def run_inference():
    # 1. 학습된 모델 (best.pt) 파일 경로 지정
    # 학습이 끝났다면 이 경로에 best.pt 파일이 생성되어 있습니다.
    model_path = 'runs/detect/floating_matters_v33/weights/best.pt'
    model = YOLO(model_path)

    # 2. 🚨 추론할 동영상/이미지 파일 경로 지정 🚨
    # 여기에 분석하고 싶은 동영상 파일의 전체 경로를 넣어주세요.
    source_path = r'C:\Users\hongyoseb\Desktop\roboflow\test_yolo.mp4' # <<< 예시입니다! 실제 동영상 경로로 변경해야 합니다.

    # 3. 모델 추론 시작 (GPU 사용)
    # source: 분석할 파일 경로
    # save=True: 탐지 결과(바운딩 박스가 그려진)를 'runs/detect/predict' 폴더에 저장합니다.
    # show=True: 실시간으로 결과를 화면에 표시합니다.
    # conf=0.25: 탐지 임계값 (신뢰도가 25% 이상인 객체만 탐지)
    # device='0': GPU 0번을 사용합니다.
    results = model.predict(
        source=source_path,
        save=True,
        show=True,
        conf=0.25, 
        device='0' 
    )

    print("\n[완료] 동영상 분석 및 탐지 결과 저장이 완료되었습니다.")
    print("결과는 PyCharm 프로젝트 폴더 내의 'runs/detect/predict' 폴더를 확인하세요.")

if __name__ == '__main__':
    run_inference()