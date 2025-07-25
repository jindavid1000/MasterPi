#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/MasterPi/')
import cv2
import time
import numpy as np
import threading
import HiwonderSDK.Board as Board
import MecanumControl.mecanum as mecanum
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *

import HiwonderSDK.Misc as Misc

# 颜色阈值，用于识别红色物体
color_range = {
    'red': [(0, 100, 60), (10, 255, 255)]
}

# PID参数
class PID:
    def __init__(self, P=0.1, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()
        
    def clear(self):
        """重置"""
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0

    def update(self, feedback_value):
        """计算PID输出"""
        error = self.SetPoint - feedback_value
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if delta_time >= self.sample_time:
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time
            self.DTerm = 0.0
            
            if delta_time > 0:
                self.DTerm = delta_error / delta_time
                
            self.last_time = self.current_time
            self.last_error = error
            
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            
        return self.output

# 初始化机械臂
AK = ArmIK()
# 底盘控制
chassis = mecanum.MecanumChassis()
# 夹爪舵机的通道
GRIPPER_SERVO = 1
# 夹爪打开与关闭的脉宽值
GRIPPER_OPEN = 2300
GRIPPER_CLOSE = 1500
# 创建PID控制器
pid_controller = PID(P=0.5, I=0.01, D=0.05)
pid_controller.SetPoint = 320  # 图像中心x坐标

# 初始化状态
is_running = False
target_detected = False
adjustment_complete = False
frame_center_x = 320
frame_center_y = 240

def initMove():
    """初始化位置"""
    # 初始化夹爪为打开状态
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_OPEN, 500)
    # 机械臂移动到初始位置（准备位置）
    AK.setPitchRangeMoving((0, 10, 20), 0, -90, 90, 1000)
    time.sleep(1)

def setBuzzer(timer):
    """设置蜂鸣器"""
    Board.setBuzzer(0)
    Board.setBuzzer(1)
    time.sleep(timer)
    Board.setBuzzer(0)

def openGripper():
    """打开夹爪"""
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_OPEN, 500)
    time.sleep(0.5)

def closeGripper():
    """关闭夹爪"""
    Board.setPWMServoPulse(GRIPPER_SERVO, GRIPPER_CLOSE, 500)
    time.sleep(0.5)

def process(frame):
    """处理图像帧，识别目标颜色"""
    global target_detected, adjustment_complete
    
    if frame is None:
        return None
    
    # 将图像转换为HSV颜色空间
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    
    # 提取红色区域
    mask = cv2.inRange(frame_hsv, 
                       np.array(color_range['red'][0]), 
                       np.array(color_range['red'][1]))
    
    # 进行形态学操作改善掩膜
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_x, target_y, target_w, target_h = None, None, None, None
    
    # 如果检测到轮廓
    if len(contours) > 0:
        target_detected = True
        
        # 取最大的轮廓
        c = max(contours, key=cv2.contourArea)
        # 计算最小外接矩形
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算中心坐标和宽高
        target_x, target_y, target_w, target_h = cv2.boundingRect(c)
        target_x = target_x + target_w // 2
        target_y = target_y + target_h // 2
        
        # 计算目标与中心的偏移量
        offset_x = target_x - frame_center_x
        
        # 使用PID控制器计算出舵机的调整量
        pid_output = pid_controller.update(target_x)
        
        # 判断是否已经对准目标
        if abs(offset_x) < 15:  # 允许15像素的误差
            adjustment_complete = True
            cv2.putText(frame, "Target Locked!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            adjustment_complete = False
            # 底盘转向调整
            if offset_x > 0:
                chassis.set_velocity(0, 0, -int(min(abs(pid_output), 30)))  # 向右转
            else:
                chassis.set_velocity(0, 0, int(min(abs(pid_output), 30)))   # 向左转
        
        # 绘制目标框和中心点
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1)
        
        # 计算目标距离（基于像素宽度的估算）
        # 假设已知物体宽度为5cm
        known_width = 5.0  # 目标物体的实际宽度，单位cm
        # 假设焦距为500
        focal_length = 500  # 这需要通过实际标定获得
        # 计算距离
        if target_w > 0:
            distance = (known_width * focal_length) / target_w
            cv2.putText(frame, f"Distance: {distance:.2f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        target_detected = False
        adjustment_complete = False
        # 目标丢失，停止底盘
        chassis.set_velocity(0, 0, 0)
    
    return frame

def grab_object():
    """夹取物体的执行逻辑"""
    global is_running, target_detected, adjustment_complete
    
    if is_running:
        return
    
    is_running = True
    initMove()  # 初始化位置
    openGripper()  # 打开夹爪
    
    try:
        # 等待目标识别和位置调整完成
        timeout = time.time() + 15  # 设置超时时间15秒
        
        # 等待目标识别和调整完成
        while not (target_detected and adjustment_complete) and time.time() < timeout:
            time.sleep(0.1)
        
        if target_detected and adjustment_complete:
            # 停止底盘运动
            chassis.set_velocity(0, 0, 0)
            time.sleep(0.5)
            
            # 机械臂向前移动到抓取位置
            AK.setPitchRangeMoving((0, 25, 10), 0, -90, 90, 1500)
            time.sleep(1.5)
            
            # 闭合夹爪抓取物体
            closeGripper()
            
            # 机械臂抬起
            AK.setPitchRangeMoving((0, 15, 20), 0, -90, 90, 1500)
            time.sleep(1.5)
            
            # 机械臂回到初始位置
            AK.setPitchRangeMoving((0, 10, 20), 0, -90, 90, 1000)
            time.sleep(1)
            
            # 提示成功
            setBuzzer(0.1)
            print("Grab success!")
        else:
            # 超时或未检测到目标
            print("Target not detected or position adjustment failed")
    except Exception as e:
        print(f"Error in grab_object: {e}")
    finally:
        is_running = False

# 主要运行函数
def run(img):
    global is_running
    
    img = process(img)
    
    if img is None:
        return img
    
    # 在图像上显示当前状态
    if is_running:
        cv2.putText(img, "Status: Running", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(img, "Status: Standby", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 在图像上显示指导提示
    cv2.putText(img, "Press 'G' to start grabbing", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 检查键盘输入
    key = cv2.waitKey(1)
    if key == ord('g') or key == ord('G'):
        if not is_running:
            # 启动抓取线程
            threading.Thread(target=grab_object, daemon=True).start()
    
    return img

# 初始化函数
def init():
    print("Grabbing function initialized")
    initMove()

# 退出函数
def exit():
    global is_running
    is_running = False
    chassis.set_velocity(0, 0, 0)
    print("Grabbing function exited")

if __name__ == '__main__':
    init() 