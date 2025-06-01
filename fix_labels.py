import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil

def extract_label_from_path(path):
    """从图片路径中提取标签（green、red 或 blue）"""
    if not path:
        return 'blue'  # 默认标签
    
    if 'green' in path.lower():
        return 'green'
    elif 'red' in path.lower():
        return 'red'
    else:
        return 'blue'

def fix_xml_file(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # 获取图片路径
    image_path = ''
    for path in root.findall('path'):
        image_path = path.text
        break
    
    # 从路径中提取标签
    label = extract_label_from_path(image_path)
    
    # 修复每个 object 标签
    for obj in root.findall('object'):
        # 删除错误的 <n> 标签
        for n_tag in obj.findall('n'):
            obj.remove(n_tag)
        
        # 检查是否已存在 name 标签
        name_tag = obj.find('name')
        if name_tag is None:
            # 如果不存在，创建新的 name 标签
            name_tag = ET.SubElement(obj, 'name')
        
        # 设置正确的标签值
        name_tag.text = label
    
    # 使用 minidom 格式化 XML
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='    ')
    
    # 保存格式化后的 XML
    with open(xml_file_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"Fixed {xml_file_path}")

def process_directory(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(directory, filename)
            fix_xml_file(xml_file_path)
            count += 1
    print(f"\nTotal files fixed: {count}")

# 处理训练集目录
train_dir = 'custum/labels/train'
print("Processing training directory...")
process_directory(train_dir)

# 处理验证集目录
val_dir = 'custum/labels/val'
print("\nProcessing validation directory...")
process_directory(val_dir)

def fix_labels():
    # 目录定义
    new_samples_dir = 'custum/images/new_samples'  # 原标签目录
    train_images_dir = 'custum/images/train'       # 训练集图片目录
    train_labels_dir = 'custum/labels/train'       # 训练集标签目录
    val_images_dir = 'custum/images/val'           # 验证集图片目录
    val_labels_dir = 'custum/labels/val'           # 验证集标签目录
    
    # 确保标签目录存在
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(new_samples_dir) if f.endswith('.txt')]
    
    # 处理每个标签文件
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        image_name = base_name + '.jpg'
        
        # 检查图片是否在训练集中
        if os.path.exists(os.path.join(train_images_dir, image_name)):
            src_label = os.path.join(new_samples_dir, label_file)
            dst_label = os.path.join(train_labels_dir, label_file)
            shutil.move(src_label, dst_label)
            print(f'Moved label to train: {label_file}')
            
        # 检查图片是否在验证集中
        elif os.path.exists(os.path.join(val_images_dir, image_name)):
            src_label = os.path.join(new_samples_dir, label_file)
            dst_label = os.path.join(val_labels_dir, label_file)
            shutil.move(src_label, dst_label)
            print(f'Moved label to val: {label_file}')
        
        else:
            print(f'Warning: No matching image found for label: {label_file}')
    
    # 打印统计信息
    print('\nLabels distribution:')
    print(f'Training labels: {len(os.listdir(train_labels_dir))}')
    print(f'Validation labels: {len(os.listdir(val_labels_dir))}')
    print(f'Remaining labels in new_samples: {len(os.listdir(new_samples_dir))}')

if __name__ == '__main__':
    fix_labels() 