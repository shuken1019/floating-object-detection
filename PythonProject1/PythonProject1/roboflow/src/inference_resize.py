# inference_resize.py (동영상 크기를 줄여서 저장하는 코드)

from ultralytics import YOLO
import cv2
import os


def run_inference_with_resize():
    # --- 1. 설정값 확인 및 수정 ---

    # 🚨 [필수 수정] 원하는 출력 동영상 해상도로 변경하세요.
    # 예시: (1280, 720) 또는 (640, 480)
    OUTPUT_SIZE = (640, 480)

    # 🚨 [필수 수정] 추론할 동영상 파일의 실제 전체 경로로 변경하세요.
    source_path = r'C:\Users\hongyoseb\Desktop\roboflow\test_yolo.mp4'

    # [확인 완료] 학습된 모델 (best.pt) 파일 경로. 학습이 완료된 'v33' 폴더를 사용합니다.
    # 만약 best.pt가 없다면 이 경로를 확인하고 수정해야 합니다.
    model_path = 'runs/detect/floating_matters_v33/weights/best.pt'

    # --- 2. 초기화 및 모델 로드 ---
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"오류: 모델 파일이 존재하지 않습니다: {model_path}")
        print("해당 경로에 best.pt 파일이 있는지 확인하거나, train.py를 다시 실행하세요.")
        return

    # 동영상 캡처 객체 생성
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다: {source_path}")
        return

    # 동영상 저장 객체 설정
    output_dir = 'runs/detect/resized_output'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, 'output_resized.mp4')

    # 프레임 레이트 (FPS)와 코덱 설정
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V 코덱 사용

    # VideoWriter 객체 생성 (OUTPUT_SIZE로 저장)
    out = cv2.VideoWriter(output_filename, fourcc, fps, OUTPUT_SIZE)

    print(f"\n[시작] 동영상 추론을 시작합니다. 출력 크기: {OUTPUT_SIZE}")

    # --- 3. 프레임별 추론 및 저장 루프 ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO 추론 실행 (GPU 사용)
        # device='0'으로 GPU 사용, verbose=False로 콘솔 출력 줄임
        results = model.predict(
            source=frame,
            save=False,
            show=False,
            conf=0.25,
            device='0',
            verbose=False
        )

        # 2. 결과 프레임 가져오기 (바운딩 박스가 그려진 원본 크기 프레임)
        # results[0].plot()이 OpenCV 이미지(numpy 배열)를 반환합니다.
        annotated_frame = results[0].plot()

        # 3. 출력 크기로 리사이즈 (핵심!)
        resized_frame = cv2.resize(annotated_frame, OUTPUT_SIZE)

        # 4. 리사이즈된 프레임을 저장
        out.write(resized_frame)

        # 5. 리사이즈된 프레임을 화면에 표시
        cv2.imshow("Resized Inference", resized_frame)

        # 'q' 키를 누르거나 1밀리초 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 4. 종료 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[완료] 동영상 분석 및 리사이즈 저장이 완료되었습니다. 파일: {output_filename}")


if __name__ == '__main__':
    run_inference_with_resize()