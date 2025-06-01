import cv2
import os
import time

def main():
    # 创建保存图片的目录
    save_dir = 'custum/images/new_samples'
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置窗口名称
    window_name = 'Capture Samples (Press SPACE to capture, q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 计数器
    count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 显示当前帧
        cv2.putText(frame, f'Captured: {count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        # 按空格保存图片
        if key == ord(' '):
            # 生成文件名（使用时间戳避免重复）
            timestamp = int(time.time() * 1000)
            filename = f'{save_dir}/{timestamp}.jpg'
            
            # 保存图片
            cv2.imwrite(filename, frame)
            count += 1
            print(f'Saved: {filename}')
        
        # 按 'q' 退出
        elif key == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f'\nTotal captured: {count} images')
    print(f'Images saved in: {save_dir}')
    print('Now you can use labelImg to annotate these images')

if __name__ == '__main__':
    main() 