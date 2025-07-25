#!/usr/bin/env python3
import cv2
import numpy as np
from distance_estimation import DistanceEstimator
import matplotlib.pyplot as plt
import json
import os

class DistanceTester:
    def __init__(self, known_width, model_path="/Users/jty_1/Desktop/MasterPi/代码/best.pt"):
        """
        初始化距离测试器
        """
        self.estimator = DistanceEstimator(known_width=known_width, model_path=model_path)
        self.distances = []  # 实际距离
        self.measurements = []  # 测量距离
        self.pixel_widths = []  # 像素宽度
        self.calibration_file = "calibration_params.json"

        if model_path.endswith(".pt"):
            self.export_to_ncnn(model_path)

    def export_to_ncnn(self, model_path):
        """
        将YOLO模型导出为NCNN格式
        """
        try:
            model = YOLO(model_path)
            model.export(format="ncnn")
            print("模型已成功导出为NCNN格式")
        except Exception as e:
            print(f"模型导出失败: {e}")
        
    def load_calibration_params(self):
        """
        加载已有的校正参数
        """
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                return json.load(f)
        return {
            'pixel_ranges': [],
            'compensation_factors': [],
            'focal_length': 1506.67
        }
        
    def save_calibration_params(self, params):
        """
        保存校正参数
        """
        with open(self.calibration_file, 'w') as f:
            json.dump(params, f, indent=4)
            
    def merge_calibration_params(self, old_params, new_params):
        """
        融合新旧校正参数
        """
        # 合并数据点
        all_ranges = old_params['pixel_ranges'] + new_params['pixel_ranges']
        all_factors = old_params['compensation_factors'] + new_params['compensation_factors']
        
        # 按像素范围排序
        sorted_pairs = sorted(zip(all_ranges, all_factors))
        all_ranges = [pair[0] for pair in sorted_pairs]
        all_factors = [pair[1] for pair in sorted_pairs]
        
        # 合并重叠区域的补偿因子（取平均值）
        merged_ranges = []
        merged_factors = []
        i = 0
        while i < len(all_ranges):
            current_range = all_ranges[i]
            current_factors = [all_factors[i]]
            j = i + 1
            while j < len(all_ranges) and abs(all_ranges[j] - current_range) < 50:  # 50像素范围内认为是同一区域
                current_factors.append(all_factors[j])
                j += 1
            merged_ranges.append(current_range)
            merged_factors.append(sum(current_factors) / len(current_factors))
            i = j
            
        # 计算新的焦距（取加权平均）
        new_focal = (old_params['focal_length'] + new_params['focal_length']) / 2
        
        return {
            'pixel_ranges': merged_ranges,
            'compensation_factors': merged_factors,
            'focal_length': new_focal
        }
    
    def collect_data(self):
        """
        收集测试数据
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        print("数据收集开始:")
        print("1. 将目标物体放置在不同距离")
        print("2. 按's'保存当前测量值")
        print("3. 输入实际距离(厘米)")
        print("4. 按'q'结束收集")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 处理帧
            processed_frame, detections = self.estimator.process_frame(frame)
            
            # 如果检测到物体,显示距离
            if detections:
                detection = detections[0]  # 使用第一个检测结果
                distance = detection['distance']
                pixel_width = detection['pixel_width']
                
                # 显示当前测量值
                cv2.putText(processed_frame, 
                           f"Measured Distance: {distance:.1f}cm",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, 
                           f"Pixel Width: {pixel_width:.1f}px",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Distance Test', processed_frame)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            elif key == ord('s') and detections:
                # 保存当前测量值
                detection = detections[0]
                print(f"\n当前测量距离: {detection['distance']:.1f}cm")
                print(f"像素宽度: {detection['pixel_width']:.1f}px")
                
                try:
                    actual_distance = float(input("请输入实际距离(厘米): "))
                    self.distances.append(actual_distance)
                    self.measurements.append(detection['distance'])
                    self.pixel_widths.append(detection['pixel_width'])
                    print("数据点已保存!")
                except ValueError:
                    print("输入无效,请输入数字")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_data(self):
        """
        分析收集的数据并生成校正参数
        """
        if not self.distances:
            print("没有数据可分析")
            return
            
        # 转换为numpy数组
        distances = np.array(self.distances)
        measurements = np.array(self.measurements)
        pixel_widths = np.array(self.pixel_widths)
        
        # 计算误差
        errors = measurements - distances
        relative_errors = (errors / distances) * 100
        
        # 创建图表
        plt.figure(figsize=(15, 5))
        
        # 1. 距离对比图
        plt.subplot(131)
        plt.scatter(distances, measurements, alpha=0.5)
        plt.plot([min(distances), max(distances)], 
                [min(distances), max(distances)], 'r--')
        plt.xlabel('实际距离 (cm)')
        plt.ylabel('测量距离 (cm)')
        plt.title('距离对比')
        
        # 2. 误差分析
        plt.subplot(132)
        plt.scatter(distances, relative_errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('实际距离 (cm)')
        plt.ylabel('相对误差 (%)')
        plt.title('相对误差分布')
        
        # 3. 像素宽度与距离的关系
        plt.subplot(133)
        plt.scatter(pixel_widths, distances, alpha=0.5)
        plt.xlabel('像素宽度')
        plt.ylabel('实际距离 (cm)')
        plt.title('像素宽度-距离关系')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\n统计分析:")
        print(f"平均相对误差: {np.mean(relative_errors):.2f}%")
        print(f"相对误差标准差: {np.std(relative_errors):.2f}%")
        print(f"最大相对误差: {np.max(np.abs(relative_errors)):.2f}%")
        
        # 计算补偿因子
        compensation_factors = distances / measurements
        
        # 生成新的校正参数
        new_params = {
            'pixel_ranges': pixel_widths.tolist(),
            'compensation_factors': compensation_factors.tolist(),
            'focal_length': self.estimator.focal_length
        }
        
        # 加载并融合已有的校正参数
        old_params = self.load_calibration_params()
        merged_params = self.merge_calibration_params(old_params, new_params)
        
        # 保存融合后的参数
        self.save_calibration_params(merged_params)
        print("\n校正参数已保存到", self.calibration_file)
        
        # 打印参数分析
        print("\n校正参数分析:")
        print(f"焦距: {merged_params['focal_length']:.2f}像素")
        print("像素范围和补偿因子:")
        for px, factor in zip(merged_params['pixel_ranges'], merged_params['compensation_factors']):
            print(f"像素宽度: {px:.1f}, 补偿因子: {factor:.3f}")

def main():
    parser = argparse.ArgumentParser(description='距离测试和校正工具')
    parser.add_argument('--known-width', type=float, required=True,
                      help='目标物体的实际宽度(厘米)')
    parser.add_argument('--model', type=str, default='/Users/jty/Desktop/moli/模型/runs/detect/train2/weights/best.pt',
                      help='YOLO模型路径')
    parser.add_argument('--use-ncnn', action='store_true',
                        help='使用NCNN模型进行推理')
    args = parser.parse_args()
    
    model_path = args.model
    if args.use_ncnn:
        model_path = model_path.replace('.pt', '_ncnn_model')

    # 创建测试器实例
    tester = DistanceTester(known_width=args.known_width, model_path=model_path)
    
    # 收集数据
    tester.collect_data()
    
    # 分析数据并保存校正参数
    tester.analyze_data()

if __name__ == "__main__":
    main() 