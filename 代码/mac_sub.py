# mac_sub.py
import zmq
import cv2
import numpy as np

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://192.168.3.30:5555")  # 替换为树莓派IP
sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

while True:
    data = sub_socket.recv_pyobj()
    frame = cv2.imdecode(data["frame"], cv2.IMREAD_COLOR)
    
    # 绘制检测框
    for box in data["boxes"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("YOLO+ZMQ", frame)
    if cv2.waitKey(1) == 27: break