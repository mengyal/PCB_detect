# 🚀 segmentation-models-pytorch 升级指南

本指南介绍如何将您的PCB焊点检测项目升级到使用 `segmentation-models-pytorch` 库。

## 📦 安装依赖

### 方法1: 自动安装
```bash
cd src/2-train
install_smp.bat
```

### 方法2: 手动安装
```bash
pip install segmentation-models-pytorch
pip install timm
```

### 方法3: 从requirements.txt安装
```bash
pip install -r requirements.txt
```

## 🎯 主要改进

### 1. **更多预训练模型**
- **UNet**: 经典语义分割模型
- **UNet++**: 改进的UNet，更好的跳跃连接
- **DeepLabV3/V3+**: 使用空洞卷积的先进模型
- **FPN**: 特征金字塔网络
- **PSPNet**: 金字塔场景解析网络
- **LinkNet**: 轻量级分割模型

### 2. **丰富的预训练编码器**
- **ResNet系列**: resnet18, resnet34, resnet50, resnet101
- **EfficientNet系列**: efficientnet-b0 到 b7
- **RegNet系列**: 新一代高效网络
- **MobileNet系列**: 移动端优化
- **Vision Transformer**: 最新的Transformer架构

### 3. **预训练权重**
- **ImageNet预训练**: 提供更好的初始化
- **更快收敛**: 减少训练时间
- **更好性能**: 通常比从头训练效果更好

## 📋 配置选择

### 快速选择配置
```bash
cd src/2-train
python model_selector.py
```

### 预定义配置

| 配置名称 | 描述 | 适用场景 |
|---------|------|---------|
| `basic` | UNet + ResNet18 | 快速测试，资源有限 |
| `recommended` | UNet + ResNet34 | 一般使用，平衡性能 |
| `pcb_optimized` | UNet++ + ResNet34 | PCB检测优化 |
| `high_performance` | UNet++ + ResNet50 | 追求最佳效果 |
| `efficient` | UNet + EfficientNet-B2 | 高效训练 |
| `lightweight` | LinkNet + MobileNetV2 | 轻量级部署 |
| `modern` | DeepLabV3+ + EfficientNet-B1 | 最新技术 |

## 🔧 使用方法

### 1. 选择配置
在 `train.py` 中设置：
```python
MODEL_CONFIG = 'pcb_optimized'  # 推荐用于PCB检测
```

### 2. 运行训练
```bash
cd src/2-train
python train.py
```

### 3. 模型会自动：
- 下载预训练权重（首次运行）
- 创建指定的分割模型
- 显示模型参数统计

## 📊 性能对比

| 配置 | 训练速度 | 准确度 | 内存使用 | 模型大小 |
|------|---------|-------|---------|----------|
| basic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 11MB |
| recommended | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 22MB |
| pcb_optimized | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 25MB |
| high_performance | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 47MB |
| efficient | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 15MB |
| lightweight | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8MB |

## 🎨 针对PCB检测的优化建议

### 1. **推荐配置**: `pcb_optimized`
- UNet++架构提供更好的细节保留
- ResNet34编码器平衡性能和速度
- 适合PCB焊点的精细分割

### 2. **训练策略**
- 使用预训练权重（ImageNet）
- 适当的数据增强
- 学习率调整策略

### 3. **硬件要求**
- **GPU内存**: 至少4GB（推荐8GB+）
- **系统内存**: 至少8GB
- **存储空间**: 预训练模型约50-200MB

## 🔄 向后兼容

项目仍然保持向后兼容：
- 如果没有安装 `segmentation-models-pytorch`，会自动使用原始UNet
- 现有的训练脚本和测试脚本无需修改
- 模型文件格式保持一致

## 🛠️ 故障排除

### 问题1: 安装失败
```bash
# 更新pip
pip install --upgrade pip

# 分别安装
pip install torch torchvision
pip install segmentation-models-pytorch
pip install timm
```

### 问题2: 模型下载缓慢
设置镜像源：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple segmentation-models-pytorch
```

### 问题3: GPU内存不足
- 使用更小的模型（如 `basic` 配置）
- 减少batch size
- 使用 `ENCODER_WEIGHTS = None` 禁用预训练

### 问题4: 预训练权重下载失败
```python
# 在train.py中设置
ENCODER_WEIGHTS = None  # 不使用预训练权重
```

## 📚 更多资源

- [segmentation-models-pytorch 文档](https://github.com/qubvel/segmentation_models.pytorch)
- [可用编码器列表](https://github.com/qubvel/segmentation_models.pytorch#encoders)
- [模型架构详解](https://github.com/qubvel/segmentation_models.pytorch#models)

---

**升级完成后，您将获得:**
- 🚀 更快的训练速度
- 🎯 更好的检测精度  
- 🔧 更灵活的模型选择
- 📊 更专业的工具链
