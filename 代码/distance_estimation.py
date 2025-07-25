#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import json

class DistanceEstimator:
    def __init__(self, known_width, model_path="/Users/jty_1/Desktop/MasterPi/代码/best.pt"):
        """
        初始化距离估计器
        
        参数:
            known_width: 目标物体的实际宽度(厘米)
            model_path: YOLO模型路径
        """
        self.known_width = known_width
        self.focal_length = 1506.67  # 初始焦距
        
        if model_path.endswith(".pt"):
            self.model = YOLO(model_path)
            self.export_to_ncnn()
        else:
            self.model = YOLO(model_path)

        self.calibration_file = "calibration_params.json"
        self.load_calibration_params()

    def export_to_ncnn(self):
        """
        将YOLO模型导出为NCNN格式
        """
        try:
            self.model.export(format="ncnn")
            print("模型已成功导出为NCNN格式")
        except Exception as e:
            print(f"模型导出失败: {e}")
        
    def load_calibration_params(self):
        """
        加载校正参数
        """
        try:
            with open(self.calibration_file, 'r') as f:
                params = json.load(f)
                self.pixel_ranges = params['pixel_ranges']
                self.compensation_factors = params['compensation_factors']
                self.focal_length = params['focal_length']
                print(f"已加载校正参数，焦距: {self.focal_length:.2f}像素")
        except (FileNotFoundError, json.JSONDecodeError):
            self.pixel_ranges = []
            self.compensation_factors = []
            print("未找到校正参数文件，使用默认参数")
    
    def get_compensation_factor(self, pixel_width):
        """
        根据像素宽度获取补偿因子
        """
        if not self.pixel_ranges:
            # 使用默认的分段补偿
            if pixel_width > 800:
                return 0.85
            elif pixel_width > 500:
                return 0.9
            elif pixel_width > 300:
                return 0.95
            return 1.0
            
        # 找到最接近的像素范围
        distances = [abs(px - pixel_width) for px in self.pixel_ranges]
        closest_idx = distances.index(min(distances))
        
        # 如果差距太大，使用线性插值
        if min(distances) > 50:
            # 找到最近的两个点进行插值
            sorted_ranges = sorted(self.pixel_ranges)
            sorted_factors = [f for _, f in sorted(zip(self.pixel_ranges, self.compensation_factors))]
            
            # 找到插值区间
            for i in range(len(sorted_ranges)):
                if sorted_ranges[i] > pixel_width:
                    if i == 0:
                        return sorted_factors[0]
                    if i == len(sorted_ranges):
                        return sorted_factors[-1]
                    # 线性插值
                    x1, x2 = sorted_ranges[i-1], sorted_ranges[i]
                    y1, y2 = sorted_factors[i-1], sorted_factors[i]
                    return y1 + (y2 - y1) * (pixel_width - x1) / (x2 - x1)
            
            return self.compensation_factors[closest_idx]
        
        return self.compensation_factors[closest_idx]
    
    def get_object_width(self, frame, x1, y1, x2, y2):
        """
        通过二值化处理和最小外接矩形计算目标的实际宽度
        
        参数:
            frame: 输入图像
            x1, y1, x2, y2: 边界框坐标
        
        返回:
            实际宽度（像素）和旋转角度
        """
        # 提取ROI
        roi = frame[y1:y2, x1:x2]
        
        # 转换到灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Otsu's二值化方法
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作改善二值化结果
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 反转图像，使目标为白色（255）
        binary = cv2.bitwise_not(binary)
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return x2 - x1, 0  # 如果没有检测到轮廓，返回边界框宽度
            
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 获取矩形的中心点、宽度、高度和角度
        center, (width, height), angle = rect
        
        # 选择较短的边作为宽度（因为立方体旋转时，较短的边更可能是真实宽度）
        actual_width = min(width, height)
        
        # 在调试窗口中显示结果
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
        
        # 在原始ROI上也画出最小外接矩形
        roi_with_rect = roi.copy()
        cv2.drawContours(roi_with_rect, [box], 0, (0, 255, 0), 2)
        
        # 水平拼接图像
        combined_image = np.hstack([roi_with_rect, debug_image])
        
        # 创建与合并图像等宽的信息面板
        debug_info = np.zeros((50, combined_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(debug_info, f"Width: {actual_width:.1f}px Angle: {angle:.1f}deg",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 合并显示
        debug_view = np.vstack([combined_image, debug_info])
        cv2.imshow('Object Analysis', debug_view)
        
        return actual_width, angle
    
    def estimate_distance(self, pixel_width):
        """
        估计距离，使用校正参数
        """
        # 基础距离计算
        base_distance = (self.known_width * self.focal_length) / pixel_width
        
        # 获取补偿因子
        compensation = self.get_compensation_factor(pixel_width)
        
        return base_distance * compensation

    def get_color_name(self, frame, x1, y1, x2, y2):
        """
        获取指定区域的主要颜色
        """
        # 提取ROI
        roi = frame[y1:y2, x1:x2]
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 颜色范围定义
        color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255)],
            'green': [(35, 100, 100), (85, 255, 255)],
            'blue': [(100, 100, 100), (140, 255, 255)],
            'yellow': [(20, 100, 100), (35, 255, 255)]
        }
        
        max_pixels = 0
        detected_color = 'unknown'
        
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            pixels = cv2.countNonZero(mask)
            
            if pixels > max_pixels:
                max_pixels = pixels
                detected_color = color
        
        return detected_color if max_pixels > 100 else 'unknown'
    
    def process_frame(self, frame, target_class=None):
        """
        处理视频帧，使用YOLO检测物体并分析颜色
        """
        # 运行YOLO检测
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        # 处理每个检测结果
        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # 如果指定了目标类别，则只处理该类别
            if target_class is not None and cls != target_class:
                continue
            
            # 获取物体颜色
            color = self.get_color_name(frame, x1, y1, x2, y2)
            
            # 通过二值化获取等效宽度和角度
            pixel_width, angle = self.get_object_width(frame, x1, y1, x2, y2)
            
            # 估计距离
            distance = self.estimate_distance(pixel_width)
            
            # 在图像上绘制检测结果
            class_name = results.names[int(cls)]
            label = f"{class_name} ({color})"
            if distance is not None:
                label += f" {distance:.1f}cm"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示等效宽度和角度（用于调试）
            cv2.putText(frame, f"Width: {pixel_width:.1f}px Angle: {angle:.1f}deg",
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'class': class_name,
                'color': color,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'distance': distance,
                'pixel_width': pixel_width,
                'angle': angle
            })
        
        return frame, detections

def main():
    parser = argparse.ArgumentParser(description='距离估计工具')
    parser.add_argument('--known-width', type=float, required=True,
                      help='目标物体的实际宽度(厘米)')
    parser.add_argument('--model', type=str, default='/Users/jty/Desktop/moli/模型/runs/detect/train2/weights/best.pt',
                      help='YOLO模型路径')
    parser.add_argument('--target-class', type=int,
                      help='目标类别ID（可选）')
    parser.add_argument('--use-ncnn', action='store_true',
                        help='使用NCNN模型进行推理')
    args = parser.parse_args()
    
    model_path = args.model
    if args.use_ncnn:
        model_path = model_path.replace('.pt', '_ncnn_model')

    # 初始化距离估计器
    estimator = DistanceEstimator(known_width=args.known_width,
                                model_path=model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按'q'键退出")
    
    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
        
        # 处理帧
        processed_frame, detections = estimator.process_frame(frame, args.target_class)
        
        # 显示帧
        cv2.imshow('Distance Estimation', processed_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 