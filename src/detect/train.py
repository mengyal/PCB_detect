import os
import random
import yaml
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def run_training(project_root):
    """
    使用YOLOv8训练PCB元件检测模型。
    - 使用 yolov8s 模型以获得更好的性能。
    - 增加 epochs 和 early stopping patience。
    - 使用绝对路径以确保稳健性。
    """
    # 1. 定义相对于项目根目录的路径
    data_yaml_path = os.path.join(project_root, 'data', 'pcb_solder.yaml')
    models_dir = os.path.join(project_root, 'models', 'detect')

    # 2. 检查数据配置文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"错误：数据配置文件未找到于 '{data_yaml_path}'")
        print("请确保 pcb_solder.yaml 文件存在于 data/ 目录下。")
        return


    # 3. 加载预训练的YOLOv8s模型
    model = YOLO('yolov8s.pt')

    # 4. 训练模型
    print(f"开始使用 {data_yaml_path} 进行训练...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,         # 增加训练轮次
        imgsz=640,
        batch=8,            # 如果你的显存较小，可以降低batch size
        patience=20,        # 早停机制，20个epoch没有提升则停止
        name='yolov8s_pcb_solder',
        project=models_dir, # 指定模型和日志的保存目录
        exist_ok=True       # 如果目录存在，则覆盖
    )

    print(f"训练完成！模型已保存至 {models_dir}/yolov8s_pcb_solder 目录中。")

if __name__ == '__main__':
    # 动态计算项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 src/2-detect 向上返回两级到达项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    print(f"项目根目录检测为: {project_root}")
    
    # 在主进程保护下运行训练，这对于Windows上的多进程很重要
    run_training(project_root)
