# 图像预处理模块

负责PCB图像的预处理和增强，为后续的检测任务提供高质量的输入数据。

## 主要功能

### 基础预处理
- `image_loader.py` - 图像加载和格式转换
- `resize_normalize.py` - 尺寸调整和像素归一化
- `color_space.py` - 颜色空间转换(RGB, HSV, LAB等)

### 图像增强
- `noise_reduction.py` - 噪声去除和滤波
- `contrast_enhancement.py` - 对比度和亮度调整
- `edge_enhancement.py` - 边缘检测和增强
- `histogram_equalization.py` - 直方图均衡化

### 几何处理
- `geometric_correction.py` - 几何校正和畸变矫正
- `registration.py` - 图像配准和对齐
- `rotation_flip.py` - 旋转和翻转操作

### ROI提取
- `roi_detection.py` - 感兴趣区域检测
- `solder_segmentation.py` - 焊点区域分割
- `background_removal.py` - 背景去除

### 数据增强
- `augmentation.py` - 数据增强策略
- `transform_pipeline.py` - 变换流水线
- `batch_processor.py` - 批量处理工具

## 使用示例

```python
from preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor()
processed_image = preprocessor.process(raw_image)
```

## 配置参数

预处理参数可通过配置文件进行调整，包括：
- 图像尺寸设置
- 滤波器参数
- 增强强度设置
- ROI检测阈值
