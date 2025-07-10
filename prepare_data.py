import os
import random

def create_yolo_dataset_files(project_root, train_split=0.8):
    """
    从图像文件夹创建YOLO所需的数据集文件(train.txt 和 val.txt)。
    路径是相对于项目根目录构建的。
    只包含具有相应标签文件的图像。
    """
    # 1. 定义相对于项目根目录的路径
    relative_image_dir = os.path.join('data', '2700', 'JPEGImages')
    relative_label_dir = os.path.join('data', '2700', 'labels') # 新增：标签目录
    relative_output_dir = os.path.join('data', '2700')

    # 2. 创建用于文件系统操作的绝对路径
    absolute_image_dir = os.path.join(project_root, relative_image_dir)
    absolute_label_dir = os.path.join(project_root, relative_label_dir) # 新增：标签绝对路径
    absolute_output_dir = os.path.join(project_root, relative_output_dir)

    # 3. 确保输出目录存在
    os.makedirs(absolute_output_dir, exist_ok=True)

    # 4. 使用绝对路径读取图片文件列表，并校验标签文件是否存在
    try:
        all_image_files = [f for f in os.listdir(absolute_image_dir) if f.endswith('.jpg')]
        image_files = []
        for img_file in all_image_files:
            label_file_name = os.path.splitext(img_file)[0] + '.txt'
            label_file_path = os.path.join(absolute_label_dir, label_file_name)
            if os.path.exists(label_file_path):
                image_files.append(img_file)
            else:
                print(f"警告：找不到图片 '{img_file}' 对应的标签文件，将跳过该图片。")
        
        if not image_files:
            print("错误：找不到任何带有标签的图片。请先运行标签转换脚本。")
            return

    except FileNotFoundError:
        print(f"错误：找不到图片目录 '{absolute_image_dir}'")
        print("请确认脚本相对于项目根目录的位置是否正确。")
        return

    random.shuffle(image_files)

    split_index = int(len(image_files) * train_split)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 5. 使用绝对路径创建 train.txt, 但写入相对路径
    with open(os.path.join(absolute_output_dir, 'train.txt'), 'w') as f:
        for file_name in train_files:
            # 写入YOLO需要的、相对于项目根目录的路径
            relative_file_path = os.path.join(relative_image_dir, file_name).replace('\\', '/')
            f.write(relative_file_path + '\n')

    # 6. 使用绝对路径创建 val.txt, 但写入相对路径
    with open(os.path.join(absolute_output_dir, 'val.txt'), 'w') as f:
        for file_name in val_files:
            relative_file_path = os.path.join(relative_image_dir, file_name).replace('\\', '/')
            f.write(relative_file_path + '\n')

    print(f"成功创建 train.txt 和 val.txt in {relative_output_dir}")
    print(f"总计有效（有标签）图片数: {len(image_files)}")
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")

if __name__ == '__main__':
    # 动态计算项目根目录
    # __file__ 是当前脚本的路径
    # os.path.dirname() 获取目录
    # os.path.abspath() 获取绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从src/1-preprocessing向上返回两级到达项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    print(f"项目根目录检测为: {project_root}")
    
    create_yolo_dataset_files(project_root)
