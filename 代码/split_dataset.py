import os
import random
import shutil

def split_dataset():
    # 设置目录
    train_dir = 'custum/labels/train'
    val_dir = 'custum/labels/val'
    image_train_dir = 'custum/images/train'
    image_val_dir = 'custum/images/val'
    
    # 确保验证集目录存在
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]
    
    # 随机选择20%作为验证集
    num_val = int(len(txt_files) * 0.2)
    val_files = random.sample(txt_files, num_val)
    
    print(f"总文件数: {len(txt_files)}")
    print(f"验证集数量: {num_val}")
    
    # 移动文件
    for txt_file in val_files:
        # 移动标签文件
        src_txt = os.path.join(train_dir, txt_file)
        dst_txt = os.path.join(val_dir, txt_file)
        shutil.move(src_txt, dst_txt)
        
        # 移动对应的图片文件
        img_file = os.path.splitext(txt_file)[0] + '.jpg'
        src_img = os.path.join(image_train_dir, img_file)
        dst_img = os.path.join(image_val_dir, img_file)
        
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            print(f"已移动 {txt_file} 和 {img_file} 到验证集")
        else:
            print(f"警告：找不到图片文件 {img_file}")

if __name__ == '__main__':
    split_dataset() 