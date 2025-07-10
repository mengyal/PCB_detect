
import os
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

def convert_to_yolo_format(size, box):
    """将PASCAL VOC的bbox坐标(xmin, ymin, xmax, ymax)转换为YOLO格式的归一化坐标(x_center, y_center, width, height)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotations(project_root):
    """
    遍历所有XML标注文件，将其转换为YOLO格式的TXT文件。
    """
    # 1. 定义类别列表，必须与yaml文件中的顺序一致
    classes = [
        "resistor", "IC", "diode", "capacitor", "transistor", "inductor",
        "LED", "connector", "clock", "switch", "battery", "buzzer",
        "display", "fuse", "relay", "potentiometer"
    ]

    # 2. 定义输入和输出目录
    annotations_dir = os.path.join(project_root, 'data', '2700', 'Annotations')
    labels_dir = os.path.join(project_root, 'data', '2700', 'labels')
    
    # 3. 确保输出目录存在
    os.makedirs(labels_dir, exist_ok=True)
    print(f"YOLO格式的标签文件将保存至: {labels_dir}")

    # 4. 获取所有XML文件
    xml_files = glob.glob(os.path.join(annotations_dir, '*.xml'))
    if not xml_files:
        print(f"错误: 在 '{annotations_dir}' 中未找到任何XML文件。")
        return

    print(f"找到 {len(xml_files)} 个XML文件，开始转换...")

    # 5. 遍历并转换每个XML文件
    for xml_file in tqdm(xml_files, desc="转换进度"):
        # 构建输出的txt文件路径
        base_filename = os.path.basename(xml_file)
        txt_filename = os.path.splitext(base_filename)[0] + '.txt'
        txt_filepath = os.path.join(labels_dir, txt_filename)

        with open(txt_filepath, "w") as out_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_to_yolo_format((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    print(f"\n转换完成！ {len(xml_files)} 个XML文件已成功转换为YOLO TXT格式。")

if __name__ == '__main__':
    # 动态计算项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 src/2-detect 向上返回两级到达项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    print(f"项目根目录检测为: {project_root}")
    
    convert_annotations(project_root)
