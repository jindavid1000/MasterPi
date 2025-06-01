from ultralytics import YOLO
import cv2
import time
import os
import glob

def list_available_models():
    """列出所有可用的模型"""
    # 搜索所有训练目录中的best.pt
    model_paths = glob.glob('runs/detect/train*/weights/best.pt')
    models = {}
    
    for i, path in enumerate(model_paths, 1):
        # 从路径中提取train编号
        train_num = path.split('/')[2].replace('train', '')
        models[i] = path
        print(f"{i}. train{train_num} 模型")
    
    return models

def select_model():
    """让用户选择要使用的模型"""
    print("\n可用的模型：")
    models = list_available_models()
    
    if not models:
        print("错误：没有找到任何训练好的模型！")
        return None
        
    while True:
        try:
            choice = int(input("\n请选择要使用的模型 (输入编号): "))
            if choice in models:
                return models[choice]
            else:
                print("无效的选择，请重试！")
        except ValueError:
            print("请输入有效的数字！")

def main():
    # 选择模型
    model_path = select_model()
    if model_path is None:
        return
        
    print(f"\n使用模型: {model_path}")
    
    # 加载选择的模型
    model = YOLO(model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置窗口名称
    window_name = 'Real-time Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 颜色映射 (BGR格式)
    colors = {
        'red': (0, 0, 255),    # 红色
        'green': (0, 255, 0),  # 绿色
        'blue': (255, 0, 0)    # 蓝色
    }
    
    # 类别名称
    class_names = ['red', 'green', 'blue']
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
            
        # 开始计时
        start_time = time.time()
        
        # 进行检测
        results = model(frame, conf=0.5)  # 置信度阈值设为 0.5
        
        # 计算FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 在每一帧上绘制检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取坐标和类别
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 转换为整数坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取类别名称和颜色
                class_name = class_names[class_id]
                color = colors[class_name]
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示FPS和当前使用的模型版本
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Model: {os.path.dirname(model_path)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow(window_name, frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 