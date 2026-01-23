# export.py
from ultralytics import YOLO

# 학습된 모델 경로를 지정합니다.
# 주의: 'runs/detect/' 뒤의 폴더 이름은 'name' 인수에 따라 다를 수 있습니다.
model_path = 'runs/detect/floating_matters_v3/weights/best.pt'

model = YOLO(model_path)

# ONNX로 내보내기 (best.onnx 파일 생성)
model.export(format='onnx')