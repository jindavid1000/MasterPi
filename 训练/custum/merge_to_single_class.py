import os
from glob import glob
import shutil

def process_labels_and_images(label_dir, image_dir, out_label_dir, out_image_dir):
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)
    label_files = glob(os.path.join(label_dir, '*.txt'))
    for label_path in label_files:
        base = os.path.splitext(os.path.basename(label_path))[0]
        # 支持jpg和png
        for ext in ['.jpg', '.png', '.jpeg']:
            image_path = os.path.join(image_dir, base + ext)
            if os.path.exists(image_path):
                # 处理标签
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = '0'  # 类别编号改为0
                        new_lines.append(' '.join(parts) + '\n')
                # 保存新标签
                with open(os.path.join(out_label_dir, base + '.txt'), 'w') as f:
                    f.writelines(new_lines)
                # 复制图片
                shutil.copy(image_path, os.path.join(out_image_dir, base + ext))
                break  # 找到图片就不再查别的扩展名

# 处理train
process_labels_and_images(
    'labels/train', 'images/train',
    'singleclass/labels/train', 'singleclass/images/train'
)
# 处理val
process_labels_and_images(
    'labels/val', 'images/val',
    'singleclass/labels/val', 'singleclass/images/val'
)
print('处理完成！')