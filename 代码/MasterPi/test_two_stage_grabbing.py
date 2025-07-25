#!/usr/bin/python3
# coding=utf8
import sys
import os
import time
import cv2
import numpy as np
import RPCServer
import Functions.TwoStageGrabbing as TwoStageGrabbing
import HiwonderSDK.Board as Board
from Camera import Camera

if sys.version_info.major == 2:
    print('请使用Python3运行此程序')
    sys.exit(0)

# 初始化相机
camera = Camera()
camera.camera_open()

# 蜂鸣器提示
Board.setBuzzer(0)
Board.setBuzzer(1)
time.sleep(0.1)
Board.setBuzzer(0)

# 初始化抓取功能
TwoStageGrabbing.init()
print("按下'g'键开始抓取，按下'q'键退出")

# 主循环
try:
    while True:
        img = camera.frame
        if img is not None:
            # 处理图像
            processed_img = TwoStageGrabbing.run(img)
            # 显示图像
            cv2.imshow("Two Stage Grabbing", processed_img)
            
            # 键盘控制
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('g') or key == ord('G'):
                # 启动抓取线程
                import threading
                grab_thread = threading.Thread(target=TwoStageGrabbing.grab_object)
                grab_thread.start()
            
        else:
            time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    # 清理
    TwoStageGrabbing.exit()
    camera.camera_close()
    cv2.destroyAllWindows()
    print("测试结束") 