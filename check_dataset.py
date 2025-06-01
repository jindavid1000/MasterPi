import os
import glob

def check_dataset(images_dir, labels_dir):
    """检查图片和标签文件的对应关系"""
    # 获取所有图片文件
    image_files = set(os.path.splitext(os.path.basename(f))[0] 
                     for f in glob.glob(os.path.join(images_dir, '*.jpg')))
    
    # 获取所有标签文件
    label_files = set(os.path.splitext(os.path.basename(f))[0] 
                     for f in glob.glob(os.path.join(labels_dir, '*.txt')))
    
    # 检查不匹配的文件
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    print(f"\n检查目录: {images_dir} 和 {labels_dir}")
    print(f"图片文件数量: {len(image_files)}")
    print(f"标签文件数量: {len(label_files)}")
    
    if images_without_labels:
        print("\n没有对应标签的图片:")
        for img in sorted(images_without_labels):
            print(f"- {img}")
    
    if labels_without_images:
        print("\n没有对应图片的标签:")
        for lbl in sorted(labels_without_images):
            print(f"- {lbl}")
    
    # 检查标签文件内容
    print("\n检查标签文件内容...")
    empty_labels = []
    for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                empty_labels.append(os.path.basename(label_file))
    
    if empty_labels:
        print("\n空的标签文件:")
        for lbl in empty_labels:
            print(f"- {lbl}")

if __name__ == '__main__':
    # 检查训练集
    check_dataset('custum/images/train', 'custum/labels/train')
    
    # 检查验证集
    check_dataset('custum/images/val', 'custum/labels/val') 