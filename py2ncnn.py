from ultralytics import YOLO   
model = YOLO("best.pt")
# 明确指定 format="ncnn"
model.export(format="ncnn")  # 不是 model.export() 或其他默认值