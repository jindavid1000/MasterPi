#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/MasterPi/')
import cv2
import time
import Camera
import threading
import yaml_handle
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Misc as Misc
import HiwonderSDK.Sonar as Sonar
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *

debug = False
AK = ArmIK()
iHWSONAR = None
board = None
if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# 加载opencv的人脸识别级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

lab_data = None
servo_data = None

def load_config():
    global lab_data, servo_data
    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)

load_config()

# 夹持器夹取时闭合的角度(the closing angle of the gripper while grasping an object)
servo1 = 1500
x_pulse = 1500

# 初始位置
def initMove():
    Board.setPWMServoPulse(1, servo1, 800)
    AK.setPitchRangeMoving((0, 8, 10), -90, -90, 0, 1500)

d_pulse = 20
start_greet = False
action_finish = True

# 变量重置(reset variables)
def reset():
    global d_pulse
    global start_greet
    global x_pulse
    global action_finish

    start_greet = False
    action_finish = True
    x_pulse = 500
    initMove()

__isRunning = False

# 初始化调用(call the initialization of the app)
def init():
    print("ColorDetect Init")
    load_config()
    initMove()

# 开始玩法调用(the app starts the game calling)
def start():
    global __isRunning
    __isRunning = True
    print("ColorDetect Start")

def move():
    global start_greet
    global action_finish
    global d_pulse, x_pulse

    while True:
        if __isRunning:
            if start_greet:
                start_greet = False
                action_finish = False
                Board.setPWMServoPulse(3, 500, int(0.05 * 1000))  # 将时间参数转换为整数
                time.sleep(0.4)
                Board.setPWMServoPulse(3, 700, int(0.05 * 1000))  # 将时间参数转换为整数
                action_finish = True
                time.sleep(0.5)
            else:
                if x_pulse >= 1900 or x_pulse <= 1100:
                    d_pulse = -d_pulse

                x_pulse += d_pulse
                # 将时间参数转换为整数
                Board.setPWMServoPulse(6, int(x_pulse), int(0.05 * 1000))  # 假设时间单位需要转换为毫秒
                time.sleep(0.05)
        else:
            time.sleep(0.01)

# 运行子线程(run sub-thread)
threading.Thread(target=move, args=(), daemon=True).start()

# 运行子线程(run sub-thread)
# threading.Thread(target=move, args=(), daemon=True).start()  # 注释掉重复的线程启动

size = (320, 240)

# 定义 faceDetection
faceDetection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

center_x, center_y, area = -1, -1, 0  # 初始化全局变量

def run(img):
    global __isRunning
    global start_greet
    global action_finish
    global center_x, center_y, area

    if not __isRunning:  # 检测是否开启玩法，没有开启则返回原图像(Detect if the game is started, if not, return the original image)
        return img

    img_h, img_w = img.shape[:2]

    # 调整参数以提高人脸识别的准确性
    results = faceDetection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)  # 直接使用BGR图像，避免颜色空间转换

    if len(results) > 0:  # 如果检测到人脸
        for (x, y, w, h) in results:  # 返回人脸索引index(第几张脸)，和关键点的坐标信息(Return the face index (which face) and the coordinate information of the keypoints)
            cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 2)  # 在每一帧图像上绘制矩形框(draw a rectangle on each frame of the image)

            # 获取识别框的信息, xy为左上角坐标点(Get information about the recognition box, where xy is the coordinates of the upper left corner)
            center_x = int(x + (w / 2))
            center_y = int(y + (h / 2))
            area = int(w * h)
            if action_finish:
                start_greet = True
    else:
        center_x, center_y, area = -1, -1, 0
        start_greet = False  # 没有检测到人脸时，将 start_greet 置为 False
    return img

# 图像显示线程函数
def display_thread():
    global cap
    while True:
        ret, img = cap.read()
        if ret:
            # 降低分辨率
            img = cv2.resize(img, (160, 120))
            frame = img.copy()
            Frame = run(frame)
            frame_resize = cv2.resize(Frame, (320, 240))
            cv2.imshow('frame', frame_resize)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    init()
    start()
    # 使用 gstreamer 优化视频流读取（需要根据实际情况调整，这里只是示例）
    # cap = cv2.VideoCapture('v4l2src device=/dev/video0 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture('http://127.0.0.1:8080?action=stream')
    # cap = cv2.VideoCapture(0)

    # 启动图像显示线程
    display_thread = threading.Thread(target=display_thread, daemon=True)
    display_thread.start()

    try:
        while True:
            time.sleep(0.1)  # 主线程可以做一些其他的事情
    except KeyboardInterrupt:
        pass

    # 假设my_camera是之前定义的相机对象，这里进行关闭操作
    # my_camera.camera_close()
    cv2.destroyAllWindows()