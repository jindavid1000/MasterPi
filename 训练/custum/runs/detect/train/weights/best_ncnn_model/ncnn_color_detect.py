import cv2
import numpy as np
import ncnn
import time

NCNN_PARAM = "model.ncnn.param"
NCNN_BIN = "model.ncnn.bin"
INPUT_SIZE = 416  # 你的训练/导出分辨率

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)

def get_main_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    main_hue = np.argmax(hist)
    if 0 <= main_hue < 15 or 160 <= main_hue < 180:
        return 'red'
    elif 35 <= main_hue < 85:
        return 'green'
    elif 100 <= main_hue < 140:
        return 'blue'
    else:
        return 'other'

# 初始化ncnn模型
net = ncnn.Net()
net.load_param(NCNN_PARAM)
net.load_model(NCNN_BIN)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按q退出")

frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 预处理
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    img = np.expand_dims(img, 0)        # NCHW
    img = np.ascontiguousarray(img)

    # ncnn推理
    with net.create_extractor() as ex:
        # ex.set_num_threads(4)  # 视树莓派CPU核心数调整
        ex.input("in0", ncnn.Mat(img.squeeze(0)).clone())
        t0 = time.time()
        _, out0 = ex.extract("out0")
        t1 = time.time()
        print(f"ncnn 推理耗时: {(t1-t0)*1000:.2f} ms")

    # 后处理
    arr = np.array(out0)
    print("arr.shape:", arr.shape)
    if arr.shape[0] == 5:
        arr = arr.transpose(1, 0)  # (N, 5)
    elif arr.shape[1] == 5:
        pass  # 已经是 (N, 5)
    else:
        raise ValueError(f"Unexpected output shape: {arr.shape}")
    boxes = []
    scores = []
    for det in arr:
        x, y, w, h, obj_conf = det
        score = obj_conf
        if score < 0.85:
            continue
        # 还原到原图坐标
        x1 = int((x - w / 2) / INPUT_SIZE * frame.shape[1])
        y1 = int((y - h / 2) / INPUT_SIZE * frame.shape[0])
        x2 = int((x + w / 2) / INPUT_SIZE * frame.shape[1])
        y2 = int((y + h / 2) / INPUT_SIZE * frame.shape[0])
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = nms(boxes, scores, 0.5)
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # 颜色分析
            roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if roi.size > 0:
                color = get_main_color(roi)
                cv2.putText(frame, f'{color}:{score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 帧率统计
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    # 显示FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ncnn Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 