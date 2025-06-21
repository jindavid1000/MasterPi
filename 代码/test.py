import cv2
from ultralytics import YOLO

def main():
    # 1. 加载模型（自动下载预训练权重）
    model = YOLO("yolov8n.pt")  # 自动适配Mac Metal加速
    
    # 2. 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 3. 实时检测循环
    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8检测（自动使用GPU加速）
        results = model.track(frame, persist=True, verbose=False)
        
        # 绘制检测结果
        cv2.imshow("YOLOv8 Detection", results[0].plot())

        # 退出检测
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()