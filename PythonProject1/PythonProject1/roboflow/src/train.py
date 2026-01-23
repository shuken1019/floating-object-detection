from ultralytics import YOLO

def main():
    # 1. 모델 불러오기
    # 'yolov8n.pt'는 가장 가볍고 빠른 모델입니다. (nano 버전)
    # 성능을 높이려면 'yolov8s.pt'나 'yolov8m.pt'로 바꿔보세요.
    model = YOLO('yolov8n.pt')

    # 2. 모델 학습 시작
    # data: data.yaml 파일의 경로
    # epochs: 학습 반복 횟수 (보통 50~100번 정도면 결과가 나옵니다)
    # imgsz: 이미지 크기 (데이터셋이 512로 전처리되어 있으므로 512 설정)
    results = model.train(
        data=r'C:\Users\hongyoseb\Desktop\roboflow\data.yaml',
        epochs=100,
        imgsz=512,
        batch=16,
        name='floating_matters_v3',
        device = '0'
    )

if __name__ == '__main__':
    main()