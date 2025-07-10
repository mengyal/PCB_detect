#!/usr/bin/env python3
"""
数据转换脚本：将语义分割标注转换为YOLO目标检测格式
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
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationToYOLOConverter:
    def __init__(self, data_root, output_dir="./yolo_dataset"):
        """
        初始化数据转换器
        
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
            annotations: 标注信息列表, img_width, img_height
        """
        try:
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
                        
            return annotations, img_width, img_height
            
        except Exception as e:
            logger.error(f"处理JSON文件时出错 {json_path}: {e}")
            return [], 0, 0
    
    def analyze_dataset(self):
        """分析数据集"""
        logger.info("开始分析数据集...")
        
        # 获取所有图片文件
        image_files = list(self.jpg_dir.glob("*.jpg"))
        logger.info(f"找到 {len(image_files)} 张图片")
        
        # 统计有标注的图片
        valid_images = []
        class_counts = {label: 0 for label in self.class_mapping.keys()}
        
        for img_path in tqdm(image_files, desc="分析数据集"):
            json_path = self.json_dir / f"{img_path.stem}.json"
            if json_path.exists():
                valid_images.append(img_path)
                
                # 统计类别
                annotations, _, _ = self.process_json_file(json_path)
                for ann in annotations:
                    class_id = ann[0]
                    class_name = list(self.class_mapping.keys())[list(self.class_mapping.values()).index(class_id)]
                    class_counts[class_name] += 1
        
        logger.info(f"找到 {len(valid_images)} 张有标注的图片")
        logger.info("类别统计:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count}")
            
        return valid_images, class_counts
    
    def create_yolo_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        创建YOLO数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        logger.info("开始创建YOLO数据集...")
        
        # 分析数据集
        valid_images, class_counts = self.analyze_dataset()
        
        if len(valid_images) == 0:
            logger.error("没有找到有效的图片和标注文件对！")
            return
        
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
        return True
        
    def process_dataset_split(self, image_files, split_name):
        """
        处理数据集的一个分割
        
        Args:
            image_files: 图片文件列表
            split_name: 分割名称 (train/val/test)
        """
        logger.info(f"处理{split_name}数据集...")
        
        for img_path in tqdm(image_files, desc=f"处理{split_name}数据"):
            try:
                # 复制图片
                dst_img_path = self.output_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dst_img_path)
                
                # 处理标注
                json_path = self.json_dir / f"{img_path.stem}.json"
                if json_path.exists():
                    annotations, img_width, img_height = self.process_json_file(json_path)
                    
                    # 保存YOLO格式标注
                    label_path = self.output_dir / split_name / "labels" / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                            
            except Exception as e:
                logger.error(f"处理图片时出错 {img_path}: {e}")
                        
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
        
    def visualize_annotations(self, num_samples=5):
        """
        可视化标注结果
        
        Args:
            num_samples: 可视化样本数量
        """
        logger.info("开始可视化标注...")
        
        # 获取训练集图片
        train_images = list((self.output_dir / "train" / "images").glob("*.jpg"))[:num_samples]
        
        # 创建可视化目录
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        for img_path in train_images:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # 读取标注
            label_path = self.output_dir / "train" / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # 转换回像素坐标
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        # 计算边界框
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        # 绘制边界框
                        class_name = list(self.class_mapping.keys())[int(class_id)]
                        color = (0, 255, 0) if class_name == 'good' else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # 保存可视化结果
            output_path = vis_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(output_path), img)
            
        logger.info(f"可视化结果已保存到: {vis_dir}")


def main():
    """主函数"""
    # 设置数据路径
    data_root = r"d:\learning_files\embeding\Competition\ELF2\deep_learning\pcb_solder_detection\data\dataset 0706"
    output_dir = r"d:\learning_files\embeding\Competition\ELF2\deep_learning\pcb_solder_detection\yolo_dataset"
    
    # 创建转换器实例
    converter = SegmentationToYOLOConverter(data_root, output_dir)
    
    # 创建数据集
    success = converter.create_yolo_dataset()
    
    if success:
        # 可视化标注
        converter.visualize_annotations()
        
        print("数据转换完成！")
        print(f"YOLO数据集保存在: {output_dir}")
        print(f"数据集配置文件: {output_dir}/dataset.yaml")
        print(f"标注可视化结果: {output_dir}/visualizations/")
    else:
        print("数据转换失败！")


if __name__ == "__main__":
    main()
