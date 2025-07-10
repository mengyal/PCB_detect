#!/usr/bin/env python3
"""
PCB电子元件检测器 - 使用训练好的YOLO模型进行电子元件检测
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCBComponentDetector:
    """PCB电子元件检测器类"""
    
    def __init__(self, model_path: str = "models/detect/best.pt"):
        """
        初始化PCB电子元件检测器
        
        Args:
            model_path: 训练好的YOLO模型路径
        """
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.parent
        
        # 处理模型路径
        if not os.path.isabs(model_path):
            self.model_path = self.project_root / model_path
        else:
            self.model_path = Path(model_path)
            
        self.model = None
        
        # 电子元件类别名称（基于配置文件）
        self.class_names = {
            0: 'resistor',      # 电阻
            1: 'IC',            # 集成电路
            2: 'diode',         # 二极管
            3: 'capacitor',     # 电容
            4: 'transistor',    # 晶体管
            5: 'inductor',      # 电感
            6: 'LED',           # 发光二极管
            7: 'connector',     # 连接器
            8: 'clock',         # 时钟
            9: 'switch',        # 开关
            10: 'battery',      # 电池
            11: 'buzzer',       # 蜂鸣器
            12: 'display',      # 显示器
            13: 'fuse',         # 保险丝
            14: 'relay',        # 继电器
            15: 'potentiometer' # 电位器
        }
        
        # 类别颜色（用于可视化）
        self.colors = {
            0: (255, 0, 0),     # 红色 - resistor
            1: (0, 255, 0),     # 绿色 - IC
            2: (0, 0, 255),     # 蓝色 - diode
            3: (255, 255, 0),   # 黄色 - capacitor
            4: (255, 0, 255),   # 洋红 - transistor
            5: (0, 255, 255),   # 青色 - inductor
            6: (128, 0, 0),     # 深红 - LED
            7: (0, 128, 0),     # 深绿 - connector
            8: (0, 0, 128),     # 深蓝 - clock
            9: (128, 128, 0),   # 橄榄 - switch
            10: (128, 0, 128),  # 紫色 - battery
            11: (0, 128, 128),  # 深青 - buzzer
            12: (192, 192, 192), # 银色 - display
            13: (255, 165, 0),  # 橙色 - fuse
            14: (255, 20, 147), # 深粉 - relay
            15: (0, 191, 255)   # 深天蓝 - potentiometer
        }
        
        # 加载模型
        self.load_model()
    
    def load_model(self) -> bool:
        """
        加载YOLO模型
        
        Returns:
            bool: 加载是否成功
        """
        try:
            if not self.model_path.exists():
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
                
            self.model = YOLO(str(self.model_path))
            logger.info(f"模型加载成功: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def detect_components(self, 
                         image_input: Union[str, np.ndarray], 
                         conf_threshold: float = 0.25, 
                         iou_threshold: float = 0.45) -> Optional[Dict]:
        """
        检测图像中的电子元件
        
        Args:
            image_input: 图片路径或numpy数组
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            Dict: 检测结果，包含边界框、标签、置信度等信息
        """
        if self.model is None:
            logger.error("模型未加载，无法进行检测")
            return None
            
        try:
            # 进行预测
            results = self.model.predict(
                source=image_input,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            if not results:
                logger.warning("未检测到任何对象")
                return {
                    'detections': [],
                    'num_detections': 0,
                    'image_shape': None
                }
            
            result = results[0]
            
            # 解析检测结果
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
                scores = result.boxes.conf.cpu().numpy()  # 置信度
                classes = result.boxes.cls.cpu().numpy().astype(int)  # 类别
                
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    detection = {
                        'id': i,
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1), 
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        },
                        'label': self.class_names.get(cls, f'class_{cls}'),
                        'class_id': int(cls),
                        'confidence': float(score),
                        'center': {
                            'x': int((x1 + x2) / 2),
                            'y': int((y1 + y2) / 2)
                        }
                    }
                    detections.append(detection)
            
            # 获取图像尺寸
            image_shape = None
            if hasattr(result, 'orig_shape'):
                image_shape = result.orig_shape
            
            return {
                'detections': detections,
                'num_detections': len(detections),
                'image_shape': image_shape,
                'model_info': {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
            return None
    
    def get_detection_summary(self, detection_result: Dict) -> Dict:
        """
        获取检测结果摘要
        
        Args:
            detection_result: 检测结果
            
        Returns:
            Dict: 检测摘要信息
        """
        if not detection_result or not detection_result['detections']:
            return {
                'total_components': 0,
                'component_counts': {},
                'confidence_stats': {},
                'area_coverage': 0.0
            }
        
        detections = detection_result['detections']
        
        # 统计各类元件数量
        component_counts = {}
        confidences = []
        total_area = 0
        
        for det in detections:
            label = det['label']
            confidence = det['confidence']
            area = det['bbox']['width'] * det['bbox']['height']
            
            component_counts[label] = component_counts.get(label, 0) + 1
            confidences.append(confidence)
            total_area += area
        
        # 计算置信度统计
        confidence_stats = {
            'mean': float(np.mean(confidences)) if confidences else 0.0,
            'min': float(np.min(confidences)) if confidences else 0.0,
            'max': float(np.max(confidences)) if confidences else 0.0,
            'std': float(np.std(confidences)) if confidences else 0.0
        }
        
        # 计算面积覆盖率
        area_coverage = 0.0
        if detection_result['image_shape'] is not None:
            image_area = detection_result['image_shape'][0] * detection_result['image_shape'][1]
            area_coverage = total_area / image_area if image_area > 0 else 0.0
        
        return {
            'total_components': len(detections),
            'component_counts': component_counts,
            'confidence_stats': confidence_stats,
            'area_coverage': area_coverage
        }
    
    def visualize_detections(self, 
                           image_input: Union[str, np.ndarray],
                           detection_result: Dict,
                           save_path: Optional[str] = None,
                           show_confidence: bool = True,
                           show_center: bool = False) -> Optional[np.ndarray]:
        """
        可视化检测结果
        
        Args:
            image_input: 输入图像（路径或numpy数组）
            detection_result: 检测结果
            save_path: 保存路径
            show_confidence: 是否显示置信度
            show_center: 是否显示中心点
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        try:
            # 读取图像
            if isinstance(image_input, str):
                img = cv2.imread(str(image_input))
                if img is None:
                    logger.error(f"无法读取图片: {image_input}")
                    return None
            else:
                img = image_input.copy()
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 绘制检测结果
            if detection_result and detection_result['detections']:
                for det in detection_result['detections']:
                    bbox = det['bbox']
                    label = det['label']
                    confidence = det['confidence']
                    class_id = det['class_id']
                    center = det['center']
                    
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    
                    # 选择颜色
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # 绘制边界框
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    if show_confidence:
                        text = f"{label}: {confidence:.2f}"
                    else:
                        text = label
                    
                    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img_rgb, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(img_rgb, text, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 绘制中心点
                    if show_center:
                        cv2.circle(img_rgb, (center['x'], center['y']), 3, color, -1)
            
            # 保存结果
            if save_path:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), img_bgr)
                logger.info(f"可视化结果已保存到: {save_path}")
            
            return img_rgb
            
        except Exception as e:
            logger.error(f"可视化过程中出错: {e}")
            return None
    
    def detect_and_analyze(self, 
                          image_input: Union[str, np.ndarray],
                          conf_threshold: float = 0.25,
                          save_visualization: Optional[str] = None) -> Dict:
        """
        检测并分析图像中的电子元件
        
        Args:
            image_input: 输入图像
            conf_threshold: 置信度阈值
            save_visualization: 可视化保存路径
            
        Returns:
            Dict: 完整的检测和分析结果
        """
        # 进行检测
        detection_result = self.detect_components(image_input, conf_threshold)
        
        if detection_result is None:
            return {
                'success': False,
                'error': '检测失败',
                'detections': [],
                'summary': {},
                'visualization_saved': False
            }
        
        # 获取摘要
        summary = self.get_detection_summary(detection_result)
        
        # 可视化（如果需要）
        visualization_saved = False
        if save_visualization:
            vis_result = self.visualize_detections(image_input, detection_result, save_visualization)
            visualization_saved = vis_result is not None
        
        return {
            'success': True,
            'detections': detection_result['detections'],
            'summary': summary,
            'model_info': detection_result.get('model_info', {}),
            'visualization_saved': visualization_saved,
            'visualization_path': save_visualization if visualization_saved else None
        }


def detect_pcb_components(image_path: str, 
                         model_path: str = "models/detect/best.pt",
                         conf_threshold: float = 0.25,
                         save_visualization: Optional[str] = None) -> Dict:
    """
    便捷函数：检测PCB图像中的电子元件
    
    Args:
        image_path: 图像路径
        model_path: 模型路径
        conf_threshold: 置信度阈值
        save_visualization: 可视化保存路径
        
    Returns:
        Dict: 检测结果
    """
    detector = PCBComponentDetector(model_path)
    return detector.detect_and_analyze(image_path, conf_threshold, save_visualization)


if __name__ == "__main__":
    # 示例使用
    logger.info("PCB电子元件检测器示例")
    
    # 初始化检测器
    detector = PCBComponentDetector()
    
    # 测试图像路径（您可以修改为实际的测试图像）
    test_image = "data/users/72999c280768ee930728b5def97260e.jpg"
    
    if Path(detector.project_root / test_image).exists():
        logger.info(f"检测图像: {test_image}")
        
        # 进行检测和分析
        result = detector.detect_and_analyze(
            image_input=str(detector.project_root / test_image),
            conf_threshold=0.3,
            save_visualization="temp/component_detection_result.jpg"
        )
        
        if result['success']:
            print(f"\n=== 检测结果 ===")
            print(f"检测到 {result['summary']['total_components']} 个电子元件")
            print(f"元件分布: {result['summary']['component_counts']}")
            print(f"平均置信度: {result['summary']['confidence_stats']['mean']:.3f}")
            print(f"面积覆盖率: {result['summary']['area_coverage']:.3f}")
            
            print(f"\n=== 详细检测结果 ===")
            for i, det in enumerate(result['detections']):
                print(f"元件 {i+1}: {det['label']} (置信度: {det['confidence']:.3f}, "
                      f"位置: [{det['bbox']['x1']}, {det['bbox']['y1']}, "
                      f"{det['bbox']['x2']}, {det['bbox']['y2']}])")
        else:
            print(f"检测失败: {result.get('error', '未知错误')}")
    else:
        logger.warning(f"测试图像不存在: {test_image}")
        logger.info("请将测试图像放置在正确位置，或修改代码中的图像路径")
