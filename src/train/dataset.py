# dataset.py
import sys
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
# 添加2-train目录到sys.path，便于跨目录导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from augmentation import get_augmentation, CopyPaste

class SolderDataset(Dataset):
    def __init__(self, image_dir, json_dir, class_mapping, transform=None, use_augmentation=True, augmentation_strategy='custom', repeat_augment=1, use_copypaste=False, copypaste_prob=0.5):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.class_mapping = class_mapping
        self.use_augmentation = use_augmentation
        self.repeat_augment = max(1, int(repeat_augment))
        self.use_copypaste = use_copypaste
        self.copypaste_prob = copypaste_prob
        # 优先使用外部传入的transform，否则用augmentation.py配置
        if self.use_augmentation:
            if transform is not None:
                self.transform = transform
            else:
                self.transform = get_augmentation(augmentation_strategy)
        else:
            self.transform = None
        
        # 获取所有图片文件名（不含后缀），并检查对应的JSON文件是否存在
        self.ids = []
        for file in os.listdir(image_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(file)[0]
                json_path = os.path.join(json_dir, base_name + '.json')
                # 只有当对应的JSON文件存在时，才添加到数据集中
                if os.path.exists(json_path):
                    self.ids.append(base_name)
                else:
                    print(f"警告: 图片 {file} 对应的JSON文件 {base_name}.json 不存在，跳过该图片")
        
        print(f"数据集初始化完成，共找到 {len(self.ids)} 个有效的图片-标注对")

    def apply_simple_augmentation(self, img, mask):
        """使用OpenCV进行简单的数据增强"""
        # 随机水平翻转
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        
        # 随机垂直翻转
        if np.random.random() > 0.7:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        
        # 随机亮度调整
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        # 随机对比度调整
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        return img, mask

    def __len__(self):
        return len(self.ids) * self.repeat_augment

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_augment
        aug_idx = idx % self.repeat_augment
        base_id = self.ids[base_idx]
        img_path = os.path.join(self.image_dir, base_id + '.jpg')
        json_path = os.path.join(self.json_dir, base_id + '.json')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换为RGB
        
        with open(json_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
            
        # --- 2. 创建Mask (像素级标签) ---
        mask = np.zeros((ann_data['imageHeight'], ann_data['imageWidth']), dtype=np.uint8)
        
        # 遍历标注中的每个形状，为每个标注区域分配对应的类别ID
        for shape in ann_data['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            
            # 根据标签映射获取类别ID
            if label in self.class_mapping:
                class_id = self.class_mapping[label]
                
                # 检查形状类型并填充对应区域
                if shape['shape_type'] == 'polygon':
                    # 多边形标注：填充多边形内部区域
                    cv2.fillPoly(mask, [points], color=class_id)
                elif shape['shape_type'] == 'rectangle':
                    # 矩形标注：填充矩形区域
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    cv2.rectangle(mask, (x1, y1), (x2, y2), color=class_id, thickness=-1)
                elif shape['shape_type'] == 'circle':
                    # 圆形标注：填充圆形区域
                    center = tuple(points[0].astype(int))
                    # 计算半径（使用第二个点到圆心的距离）
                    radius = int(np.linalg.norm(points[1] - points[0]))
                    cv2.circle(mask, center, radius, color=class_id, thickness=-1)
                else:
                    print(f"警告: 未知的形状类型 '{shape['shape_type']}' 在标签 '{label}' 中，跳过该标注")
            else:
                print(f"警告: 标签 '{label}' 不在类别映射中，跳过该标注")
        
        # --- 3. 图像和mask尺寸处理（下采样） ---
        # 如果需要下采样，在这里进行尺寸调整
        # 确保图像和mask的尺寸完全一致
        target_height, target_width = 512, 512  # 你可以根据需要修改这个尺寸

        
        # 对图像进行下采样
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        # 对mask进行下采样（注意：mask使用最近邻插值避免标签值被改变）
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # --- 4. 数据增强 ---
        # --- 保证原图一定参与训练 ---
        if aug_idx == 0:
            # 不做任何增强，直接返回原图
            pass
        else:
            # 做增强
            if self.use_augmentation:
                if self.transform is not None:
                    try:
                        # 尝试使用albumentations增强
                        augmented = self.transform(image=img, mask=mask)
                        img = augmented['image']
                        mask = augmented['mask']
                    except Exception as e:
                        print(f"数据增强失败，使用原始数据: {e}")
                else:
                    # 使用简单的OpenCV增强
                    img, mask = self.apply_simple_augmentation(img, mask)
            # --- CopyPaste增强（仅训练集用，且在albumentations之后） ---
            if self.use_copypaste:
                copypaste = CopyPaste(self, p=self.copypaste_prob, max_paste=2)
                out = copypaste(img, mask)
                img, mask = out['image'], out['mask']
            
        # --- 5. 转换为Tensor ---
        # 将图像从 HWC (高, 宽, 通道) 转换为 PyTorch 需要的 CHW (通道, 高, 宽)
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0  # 归一化到[0,1]
        # Mask不需要通道维度，所以直接转换
        mask_tensor = torch.from_numpy(mask).long()
        
        # 释放numpy数组内存
        del img, mask
        
        return img_tensor, mask_tensor