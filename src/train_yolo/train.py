#!/usr/bin/env python3
"""
YOLO训练脚本
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        return True
    else:
        logger.info("GPU不可用，将使用CPU训练")
        return False

def train_yolo_model(dataset_path, model_size='n', epochs=100, batch_size=16, img_size=640, project_name='pcb_solder_detection'):
    """
    训练YOLO模型
    
    Args:
        dataset_path: 数据集配置文件路径
        model_size: 模型大小 (n, s, m, l, x)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图片大小
        project_name: 项目名称
    """
    logger.info("开始训练YOLO模型...")
    
    # 检查GPU
    gpu_available = check_gpu()
    
    # 设置设备
    device = 'cuda' if gpu_available else 'cpu'
    
    # 根据GPU内存调整批次大小
    if gpu_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 6:
            batch_size = min(batch_size, 8)
            logger.info(f"GPU内存较小，调整批次大小为: {batch_size}")
    else:
        batch_size = min(batch_size, 4)
        logger.info(f"使用CPU训练，调整批次大小为: {batch_size}")
    
    # 初始化模型
    model = YOLO(f'yolov8{model_size}.pt')
    
    # 训练参数
    train_args = {
        'data': dataset_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': 'runs/train',
        'name': project_name,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        'patience': 50,
        'save': True,
        'cache': False,
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
    }
    
    # 开始训练
    try:
        results = model.train(**train_args)
        logger.info("模型训练完成！")
        
        # 输出训练结果
        logger.info(f"最佳模型保存在: runs/train/{project_name}/weights/best.pt")
        logger.info(f"最后一轮模型保存在: runs/train/{project_name}/weights/last.pt")
        
        return results
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        return None

def evaluate_model(model_path, dataset_path):
    """
    评估模型
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集配置文件路径
    """
    logger.info("开始评估模型...")
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 评估模型
        results = model.val(data=dataset_path, split='val')
        
        # 输出评估结果
        logger.info("模型评估完成！")
        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")
        logger.info(f"Precision: {results.box.mp:.4f}")
        logger.info(f"Recall: {results.box.mr:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        return None

def test_model(model_path, test_images_dir, output_dir='./test_results'):
    """
    测试模型
    
    Args:
        model_path: 模型路径
        test_images_dir: 测试图片目录
        output_dir: 输出目录
    """
    logger.info("开始测试模型...")
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 获取测试图片
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            test_images.extend(Path(test_images_dir).glob(ext))
        
        if not test_images:
            logger.warning(f"在 {test_images_dir} 中没有找到图片文件")
            return
            
        logger.info(f"找到 {len(test_images)} 张测试图片")
        
        # 进行预测
        results = model.predict(
            source=str(test_images_dir),
            save=True,
            project=str(output_path),
            name='predictions',
            conf=0.25,
            iou=0.45,
            show_labels=True,
            show_conf=True,
            save_txt=True,
            save_conf=True
        )
        
        logger.info(f"测试完成！结果保存在: {output_path}/predictions/")
        
        return results
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        return None

def main():
    """主函数"""
    # 设置参数
    dataset_path = r"d:\learning_files\embeding\Competition\ELF2\deep_learning\pcb_solder_detection\yolo_dataset\dataset.yaml"
    
    # 检查数据集配置文件是否存在
    if not Path(dataset_path).exists():
        logger.error(f"数据集配置文件不存在: {dataset_path}")
        logger.error("请先运行 convert_data.py 创建YOLO数据集")
        return
    
    # 训练参数
    train_config = {
        'model_size': 'n',    # 使用YOLOv8nano版本
        'epochs': 100,        # 训练轮数
        'batch_size': 16,     # 批次大小
        'img_size': 640,      # 图片大小
        'project_name': 'pcb_solder_detection'
    }
    
    print("开始训练YOLO模型...")
    print(f"训练配置: {train_config}")
    
    # 训练模型
    results = train_yolo_model(dataset_path, **train_config)
    
    if results:
        # 评估模型
        best_model_path = f"runs/train/{train_config['project_name']}/weights/best.pt"
        if Path(best_model_path).exists():
            evaluate_model(best_model_path, dataset_path)
            
            # 测试模型
            test_images_dir = r"d:\learning_files\embeding\Competition\ELF2\deep_learning\pcb_solder_detection\yolo_dataset\test\images"
            if Path(test_images_dir).exists():
                test_model(best_model_path, test_images_dir)
        
        print("训练和评估完成！")
    else:
        print("训练失败！")

if __name__ == "__main__":
    main()
