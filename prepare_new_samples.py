import os
import shutil
import random

def prepare_samples():
    # 源目录和目标目录
    new_samples_dir = 'custum/images/new_samples'
    train_images_dir = 'custum/images/train'
    train_labels_dir = 'custum/labels/train'
    val_images_dir = 'custum/images/val'
    val_labels_dir = 'custum/labels/val'
    
    # 确保目录存在
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(new_samples_dir) if f.endswith('.txt')]
    image_files = [os.path.splitext(f)[0] + '.jpg' for f in label_files]
    
    # 随机选择 20% 作为验证集
    num_val = int(len(label_files) * 0.2)
    val_samples = set(random.sample(range(len(label_files)), num_val))
    
    # 移动文件
    for i, (lbl_file, img_file) in enumerate(zip(label_files, image_files)):
        src_lbl = os.path.join(new_samples_dir, lbl_file)
        src_img = os.path.join(new_samples_dir, img_file)
        
        if i in val_samples:
            # 移动到验证集
            dst_img = os.path.join(val_images_dir, img_file)
            dst_lbl = os.path.join(val_labels_dir, lbl_file)
        else:
            # 移动到训练集
            dst_img = os.path.join(train_images_dir, img_file)
            dst_lbl = os.path.join(train_labels_dir, lbl_file)
        
        # 移动标签
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
            print(f'Moved {src_lbl} to {dst_lbl}')
        
        # 移动图片（如果存在）
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            print(f'Moved {src_img} to {dst_img}')
    
    print('\nSample distribution:')
    print(f'Training samples: {len(label_files) - num_val}')
    print(f'Validation samples: {num_val}')

if __name__ == '__main__':
    prepare_samples() 