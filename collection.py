import cv2
import os

# 定义类别和保存路径
CLASSES = {
    "1": "red",    # 按1保存为猫
    "2": "blue",    # 按2保存为狗
    "3": "green"  # 按3保存为人
}

# 创建类别子目录
for class_name in CLASSES.values():
    os.makedirs(f"captured_images/{class_name}", exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ 摄像头打开失败！请检查权限")
    exit()

print("📸 摄像头已就绪，按数字键保存对应类别：")
print("1: red | 2: blue | 3: green | q: 退出")

counters = {cls: 0 for cls in CLASSES.values()}  # 各类别计数器

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取画面")
        break

    # 显示实时画面（左上角提示操作）
    cv2.putText(frame, "1:Cat | 2:Dog | 3:Person | q:Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Press Number to Save", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("1"), ord("2"), ord("3")):  # 检测数字键
        class_id = chr(key)
        class_name = CLASSES[class_id]
        
        # 保存图片到对应类别目录
        img_path = f"captured_images/{class_name}/{counters[class_name]}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"✅ 保存 {class_name}: {img_path}")
        counters[class_name] += 1

    elif key == ord("q"):  # 退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 打印采集统计
print("\n📊 采集结果：")
for cls, cnt in counters.items():
    print(f"{cls}: {cnt}张")