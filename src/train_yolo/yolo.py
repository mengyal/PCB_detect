"""
PCB焊点检测YOLO训练脚本
将语义分割标注转换为目标检测标注，然后训练YOLO模型
"""

import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCBSolderDetectionYOLO:
    def __init__(self, data_root, output_dir="data/dataset_0706/yolo"):
        """
        初始化PCB焊点检测YOLO训练器
        
        Args:
            data_root: 数据集根目录
            output_dir: 输出目录
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.jpg_dir = self.data_root / "jpg"
        self.json_dir = self.data_root / "json"
        
        # 类别映射
        self.class_mapping = {
            'background': 0,
            'good': 1,
            'insufficient': 2,
            'excess': 3,
            'shift': 4,
            'miss': 5
        }
        
        # 创建输出目录
        self.create_directories()
        
    def create_directories(self):
        """创建YOLO数据集目录结构"""
        dirs = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "labels",
            self.output_dir / "val" / "images", 
            self.output_dir / "val" / "labels",
            self.output_dir / "test" / "images",
            self.output_dir / "test" / "labels"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def polygon_to_bbox(self, polygon):
        """
        将多边形转换为边界框
        
        Args:
            polygon: 多边形坐标列表 [[x1, y1], [x2, y2], ...]
            
        Returns:
            bbox: [x_min, y_min, x_max, y_max]
        """
        if not polygon:
            return None
            
        polygon = np.array(polygon)
        x_min = np.min(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        x_max = np.max(polygon[:, 0])
        y_max = np.max(polygon[:, 1])
        
        return [x_min, y_min, x_max, y_max]
    
    def bbox_to_yolo_format(self, bbox, img_width, img_height):
        """
        将边界框转换为YOLO格式
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            yolo_bbox: [x_center, y_center, width, height] (归一化)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    def process_json_file(self, json_path):
        """
        处理单个JSON文件，提取标注信息
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            annotations: 标注信息列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        annotations = []
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        for shape in data['shapes']:
            label = shape['label']
            if label in self.class_mapping:
                class_id = self.class_mapping[label]
                points = shape['points']
                
                # 将多边形转换为边界框
                bbox = self.polygon_to_bbox(points)
                if bbox:
                    # 转换为YOLO格式
                    yolo_bbox = self.bbox_to_yolo_format(bbox, img_width, img_height)
                    annotations.append([class_id] + yolo_bbox)
                    
        return annotations
    
    def create_yolo_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        创建YOLO数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        logger.info("开始创建YOLO数据集...")
        
        # 获取所有图片文件
        image_files = list(self.jpg_dir.glob("*.jpg"))
        logger.info(f"找到 {len(image_files)} 张图片")
        
        # 过滤有对应JSON标注的图片
        valid_images = []
        for img_path in image_files:
            json_path = self.json_dir / f"{img_path.stem}.json"
            if json_path.exists():
                valid_images.append(img_path)
        
        logger.info(f"找到 {len(valid_images)} 张有标注的图片")
        
        # 划分数据集
        train_files, temp_files = train_test_split(
            valid_images, test_size=(1-train_ratio), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42
        )
        
        logger.info(f"训练集: {len(train_files)}, 验证集: {len(val_files)}, 测试集: {len(test_files)}")
        
        # 处理各个数据集
        self.process_dataset_split(train_files, "train")
        self.process_dataset_split(val_files, "val")
        self.process_dataset_split(test_files, "test")
        
        # 创建数据集配置文件
        self.create_dataset_config()
        
        logger.info("YOLO数据集创建完成！")
        
    def process_dataset_split(self, image_files, split_name):
        """
        处理数据集的一个分割
        
        Args:
            image_files: 图片文件列表
            split_name: 分割名称 (train/val/test)
        """
        logger.info(f"处理{split_name}数据集...")
        
        for img_path in tqdm(image_files):
            # 复制图片
            dst_img_path = self.output_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # 处理标注
            json_path = self.json_dir / f"{img_path.stem}.json"
            if json_path.exists():
                annotations = self.process_json_file(json_path)
                
                # 保存YOLO格式标注
                label_path = self.output_dir / split_name / "labels" / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                        
    def create_dataset_config(self):
        """创建YOLO数据集配置文件"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"数据集配置文件已保存到: {config_path}")
        
    def train_yolo_model(self, model_size='n', epochs=100, batch_size=16, img_size=640):
        """
        训练YOLO模型
        
        Args:
            model_size: 模型大小 (n, s, m, l, x)
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图片大小
        """
        logger.info("开始训练YOLO模型...")
        
        # 初始化模型
        model = YOLO(f'yolov8{model_size}.pt')
        
        # 检查是否有可用的GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 训练模型
        results = model.train(
            data=str(self.output_dir / "dataset.yaml"),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,  # 修正设备选择
            project=str(self.output_dir / "runs"),
            name='pcb_solder_detection',
            save_period=10,
            val=True,
            plots=True,
            verbose=True
        )
        
        logger.info("模型训练完成！")
        return results
    
    def evaluate_model(self, model_path):
        """
        评估训练好的模型
        
        Args:
            model_path: 模型路径
        """
        logger.info("开始评估模型...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 在验证集上评估
        results = model.val(data=str(self.output_dir / "dataset.yaml"))
        
        logger.info(f"模型评估完成！")
        logger.info(f"mAP50: {results.box.map50}")
        logger.info(f"mAP50-95: {results.box.map}")
        
        return results
    
    def visualize_predictions(self, model_path, num_images=5):
        """
        可视化模型预测结果
        
        Args:
            model_path: 模型路径
            num_images: 可视化图片数量
        """
        logger.info("开始可视化预测结果...")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 获取测试图片
        test_images = list((self.output_dir / "test" / "images").glob("*.jpg"))[:num_images]
        
        # 创建可视化目录
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        for img_path in test_images:
            # 进行预测
            results = model(str(img_path))
            
            # 保存预测结果
            for i, result in enumerate(results):
                # 绘制结果
                annotated_img = result.plot()
                
                # 保存图片
                output_path = vis_dir / f"{img_path.stem}_prediction.jpg"
                cv2.imwrite(str(output_path), annotated_img)
                
        logger.info(f"可视化结果已保存到: {vis_dir}")
        
    def run_complete_pipeline(self, model_size='n', epochs=100, batch_size=16, img_size=640):
        """
        运行完整的训练流程
        
        Args:
            model_size: 模型大小
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图片大小
        """
        logger.info("开始完整的训练流程...")
        
        # 1. 创建数据集
        self.create_yolo_dataset()
        
        # 2. 训练模型
        results = self.train_yolo_model(model_size, epochs, batch_size, img_size)
        
        # 3. 获取最佳模型路径
        best_model_path = self.output_dir / "runs" / "pcb_solder_detection" / "weights" / "best.pt"
        
        # 4. 评估模型
        if best_model_path.exists():
            self.evaluate_model(str(best_model_path))
            
            # 5. 可视化预测结果
            self.visualize_predictions(str(best_model_path))
        
        logger.info("完整训练流程完成！")
        
        return results


def main():
    """主函数"""
    # 设置数据路径
    data_root = "./data/dataset_0706"
    output_dir = "./data/dataset_0706/yolo"
    
    # 创建训练器实例
    trainer = PCBSolderDetectionYOLO(data_root, output_dir)
    
    # 运行完整流程
    # 可以根据需要调整参数
    results = trainer.run_complete_pipeline(
        model_size='n',  # 使用YOLOv8n模型 (nano版本，更轻量)
        epochs=100,      # 训练100轮
        batch_size=16,   # 批次大小16
        img_size=640     # 图片尺寸640x640
    )
    
    print("训练完成！")
    print(f"最佳模型保存在: {output_dir}/runs/pcb_solder_detection/weights/best.pt")


if __name__ == "__main__":
    main()
