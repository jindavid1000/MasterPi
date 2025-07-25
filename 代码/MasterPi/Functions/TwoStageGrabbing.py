#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/MasterPi/')
import cv2
import time
import torch
import numpy as np
import threading
import HiwonderSDK.Board as Board
import MecanumControl.mecanum as mecanum
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
from ultralytics import YOLO

# 初始化机械臂
AK = ArmIK()
# 底盘控制
chassis = mecanum.MecanumChassis()
# 夹爪舵机的通道
GRIPPER_SERVO = 1
GRIPPER_ROTATION_SERVO = 2  # 夹爪水平旋转舵机
# 夹爪打开与关闭的脉宽值
GRIPPER_OPEN = 2300
GRIPPER_CLOSE = 1500
# 夹爪旋转的脉宽范围
ROTATION_MIN = 500
ROTATION_MAX = 2500
ROTATION_MIDDLE = 1500

# 相机参数
FOCAL_LENGTH = 1506.67  # 相机焦距
KNOWN_WIDTH = 3.0  # 目标物体的实际宽度(cm)

# 机械结构参数
CAMERA_GRIPPER_VERTICAL_OFFSET = 3.0  # 摄像头到夹爪的垂直距离(cm)
GRIPPER_CENTER_DEPTH_OFFSET = 1.5  # 夹爪旋转中心相对摄像头的后向偏移(cm)
GRIPPER_ARM_LENGTH = 5.5  # 夹爪臂长(cm)

# 状态变量
is_running = False
target_detected = False
target_direction = None  # 目标方向
target_distance = None  # 目标距离
stage = 'INIT'  # 当前阶段：INIT, SEARCH_ANGLE, VERTICAL, GRAB
frame_center_x = 320
frame_center_y = 240

# 加载YOLO模型
model = YOLO("/Users/jty/Desktop/moli/模型/runs/detect/train2/weights/best.pt")

def angle_to_pulse(angle):
    """将角度转换为舵机脉冲值"""
    # 将角度映射到500-2500的脉冲范围
    pulse = int(1500 + (angle / 90.0) * 1000)
    return max(ROTATION_MIN, min(ROTATION_MAX, pulse))

def set_gripper_rotation(angle):
    """设置夹爪旋转角度"""
    pulse = angle_to_pulse(angle)
    Board.setPWMServoPulse(GRIPPER_ROTATION_SERVO, pulse, 500)
    time.sleep(0.5)

def initMove():
    """初始化位置"""
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_OPEN, 500)
    set_gripper_rotation(0)  # 夹爪旋转归零
    # 机械臂移动到斜向观察位置
    AK.setPitchRangeMoving((0, 15, 25), -30, -60, 0, 1000)  # 调整角度以便斜向观察
    time.sleep(1)

def moveToVerticalView():
    """移动到垂直观察位置"""
    AK.setPitchRangeMoving((0, 10, 30), 0, -90, 90, 1000)  # 垂直向下观察
    time.sleep(1)

def openGripper():
    """打开夹爪"""
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_OPEN, 500)
    time.sleep(0.5)

def closeGripper():
    """关闭夹爪"""
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_CLOSE, 500)
    time.sleep(0.5)

def estimate_distance(pixel_width):
    """估计目标距离"""
    if pixel_width == 0:
        return None
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
    
    # 添加非线性补偿
    if pixel_width > 800:
        distance *= 0.85
    elif pixel_width > 500:
        distance *= 0.9
    elif pixel_width > 300:
        distance *= 0.95
        
    return distance

def get_object_width(frame, x1, y1, x2, y2):
    """通过二值化处理和最小外接矩形计算目标的实际宽度和角度"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
        
    # 转换到灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用自适应阈值进行二值化
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    # 找到最大的轮廓
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    width = min(rect[1])  # 取较短边作为宽度
    angle = rect[2]
    
    # 调整角度，使其在-90到90度之间
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
        
    return width, angle

def detect_object(frame):
    """使用YOLO检测目标"""
    results = model(frame, verbose=False)[0]
    detections = []
    
    for r in results.boxes.data:
        x1, y1, x2, y2, score, class_id = r
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 获取目标颜色
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        # 转换到HSV空间判断颜色
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 红色的HSV阈值
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        if cv2.countNonZero(mask) > (roi.shape[0] * roi.shape[1] * 0.3):  # 如果红色区域超过30%
            # 获取实际宽度和角度
            pixel_width, angle = get_object_width(frame, x1, y1, x2, y2)
            if pixel_width is None:
                continue
                
            # 估计距离
            distance = estimate_distance(pixel_width)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'size': (x2 - x1, y2 - y1),
                'angle': angle,
                'distance': distance,
                'score': score
            })
            
    return detections

def process_frame(frame):
    """处理每一帧图像"""
    global stage, target_detected, target_direction, target_distance
    
    if frame is None:
        return frame
        
    detections = detect_object(frame)
    
    if len(detections) > 0:
        target_detected = True
        detection = detections[0]  # 使用第一个检测结果
        x1, y1, x2, y2 = detection['bbox']
        center_x, center_y = detection['center']
        
        # 在图像上绘制检测框和中心点
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        if stage == 'SEARCH_ANGLE':
            # 记录目标方向和距离
            target_direction = detection['angle']
            target_distance = detection['distance']
            cv2.putText(frame, f"Angle: {target_direction:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {target_distance:.1f}cm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif stage == 'VERTICAL':
            # 计算与图像中心的偏差
            offset_x = center_x - frame_center_x
            offset_y = center_y - frame_center_y
            
            cv2.putText(frame, f"Offset X: {offset_x}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Offset Y: {offset_y}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 判断是否对准目标
            if abs(offset_x) < 15 and abs(offset_y) < 15:  # 允许15像素的误差
                cv2.putText(frame, "Target Locked!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                stage = 'GRAB'  # 切换到抓取阶段
            else:
                # 使用机械臂调整位置
                current_coords = AK.getCoords()  # 获取当前坐标
                if current_coords is not None:
                    x, y, z = current_coords
                    
                    # 将像素偏移转换为实际距离偏移（基于距离估计）
                    # 假设视场角为60度，图像宽度为640像素
                    pixel_to_distance = target_distance * np.tan(np.radians(30)) / 320
                    
                    # 计算需要的位置调整
                    dx = -offset_x * pixel_to_distance  # 左右调整
                    dy = -offset_y * pixel_to_distance  # 前后调整
                    
                    # 限制调整范围
                    dx = np.clip(dx, -2, 2)  # 限制单次调整不超过2cm
                    dy = np.clip(dy, -2, 2)
                    
                    # 移动机械臂
                    AK.setPitchRangeMoving((x + dx, y + dy, z), 0, -90, target_direction, 1000)
    else:
        target_detected = False
        
    # 显示当前阶段
    cv2.putText(frame, f"Stage: {stage}", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def grab_object():
    """执行抓取流程"""
    global is_running, stage, target_detected, target_direction, target_distance
    
    if is_running:
        return
        
    try:
        is_running = True
        stage = 'INIT'
        
        # 初始化
        initMove()
        openGripper()
        time.sleep(1)
        
        # 第一阶段：斜向寻找目标
        stage = 'SEARCH_ANGLE'
        timeout = time.time() + 10  # 10秒超时
        
        while not target_detected and time.time() < timeout:
            time.sleep(0.1)
        
        if not target_detected:
            print("未找到目标")
            return
            
        # 记录目标方向和距离
        object_angle = target_direction
        object_distance = target_distance
        print(f"目标方向: {object_angle}度")
        print(f"目标距离: {object_distance}cm")
        
        # 调整夹爪角度
        set_gripper_rotation(object_angle)
        time.sleep(0.5)
        
        # 第二阶段：切换到垂直视角
        stage = 'VERTICAL'
        moveToVerticalView()
        time.sleep(1)
        
        # 等待对准目标
        timeout = time.time() + 10
        while stage != 'GRAB' and time.time() < timeout:
            if not target_detected:
                print("目标丢失")
                return
            time.sleep(0.1)
            
        if stage != 'GRAB':
            print("对准目标超时")
            return
            
        # 执行抓取
        print("开始抓取")
        
        # 计算实际工作位置
        # 考虑夹爪的机械结构，从摄像头位置计算实际抓取点
        # 夹爪中心位置比摄像头低 CAMERA_GRIPPER_VERTICAL_OFFSET，后移 GRIPPER_CENTER_DEPTH_OFFSET
        work_x = 0  # 默认在中心线上
        work_y = -GRIPPER_CENTER_DEPTH_OFFSET  # 考虑后向偏移
        work_z = object_distance - CAMERA_GRIPPER_VERTICAL_OFFSET  # 考虑垂直偏移
        
        # 确保不会碰撞
        min_safe_height = CAMERA_GRIPPER_VERTICAL_OFFSET + 2  # 预留2cm安全距离
        work_z = max(min_safe_height, work_z)
        
        # 移动到抓取位置
        AK.setPitchRangeMoving((work_x, work_y, work_z), 0, -90, object_angle, 1000)
        time.sleep(1)
        closeGripper()
        
        # 抬起物体到安全高度
        lift_height = CAMERA_GRIPPER_VERTICAL_OFFSET + 10  # 抬高10cm
        AK.setPitchRangeMoving((0, -GRIPPER_CENTER_DEPTH_OFFSET, lift_height), 0, -90, object_angle, 1000)
        
    finally:
        is_running = False
        stage = 'INIT'

def run(img):
    """主运行函数"""
    return process_frame(img)

def init():
    """初始化"""
    initMove()

def exit():
    """退出清理"""
    global is_running
    is_running = False 