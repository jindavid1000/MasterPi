import os
import xml.etree.ElementTree as ET

def convert_coordinates(size, box):
    """将 VOC 格式的坐标转换为 YOLO 格式"""
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    
    # VOC 格式: xmin, ymin, xmax, ymax
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # 归一化
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    
    return (x, y, w, h)

def convert_xml_to_yolo():
    # 输入和输出目录
    xml_dir = 'custum/labels/xml_backup'
    output_dir = 'custum/images/new_samples'  # YOLO格式标签的输出目录
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有XML文件
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        try:
            print(f"\n处理文件: {xml_file}")
            # 解析XML文件
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像尺寸
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 创建对应的txt文件
            txt_name = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(output_dir, txt_name)
            
            with open(txt_path, 'w') as f:
                # 处理每个目标
                objects = root.findall('object')
                print(f"找到 {len(objects)} 个目标")
                
                for obj in objects:
                    try:
                        # 获取类别
                        n_tag = obj.find('n')
                        if n_tag is None:
                            print(f"警告：找不到 <n> 标签")
                            name_tag = obj.find('name')
                            if name_tag is not None:
                                class_name = name_tag.text.strip()
                                print(f"使用 <name> 标签代替: {class_name}")
                            else:
                                print("警告：也找不到 <name> 标签")
                                continue
                        else:
                            class_name = n_tag.text.strip()
                            print(f"找到类别: {class_name}")
                        
                        # 根据类别名称分配ID
                        if class_name == 'red':
                            class_id = 0
                        elif class_name == 'green':
                            class_id = 1
                        elif class_name == 'blue':
                            class_id = 2
                        else:
                            print(f"警告：未知类别 {class_name}")
                            continue
                        
                        # 获取边界框坐标
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                        
                        # 转换为YOLO格式
                        yolo_coords = convert_coordinates((width, height), (xmin, ymin, xmax, ymax))
                        
                        # 写入txt文件
                        f.write(f"{class_id} {' '.join([str(coord) for coord in yolo_coords])}\n")
                        print(f"已写入标签: 类别={class_id}, 坐标={yolo_coords}")
                    except Exception as e:
                        print(f"处理对象时出错: {str(e)}")
                        continue
            
            print(f"完成转换: {xml_file} -> {txt_name}")
        except Exception as e:
            print(f"处理文件时出错 {xml_file}: {str(e)}")
            continue

if __name__ == '__main__':
    convert_xml_to_yolo() 