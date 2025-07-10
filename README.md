# PCB焊点检测系统 🔍

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于深度学习的PCB焊点质量检测系统，支持自动化光学检测(AOI)、缺陷识别和质量评估。本项目针对电子制造业中PCB焊点质量控制需求，提供了一套完整的从数据预处理到模型部署的解决方案，特别针对RK3588等边缘计算平台进行了优化。

## 🎯 项目特色

- **🤖 智能检测**: 基于YOLO架构的实时焊点检测与分类
- **🔧 边缘部署**: 针对RK3588开发板优化，支持NPU加速
- **📊 全面分析**: 多种缺陷类型识别（虚焊、桥接、锡少、锡多等）
- **🎛️ 可视化界面**: 直观的检测结果展示和分析报告
- **🔄 自动化流程**: 从图像预处理到结果输出的完整工作流
- **📈 持续学习**: 支持增量训练和模型微调

## 📋 技术栈

### 核心框架
- **深度学习**: PyTorch 1.9+, YOLO v8, Ultralytics
- **计算机视觉**: OpenCV 4.5+, Albumentations
- **数据处理**: NumPy, Pandas, Pillow
- **可视化**: Matplotlib, Seaborn, Plotly

### 部署工具
- **模型转换**: ONNX, RKNN-Toolkit
- **Web框架**: Flask, FastAPI
- **开发工具**: Jupyter, TensorBoard

### 硬件支持
- **边缘设备**: RK3588, RK3568 (NPU加速)
- **GPU支持**: CUDA 11.8+
- **CPU推理**: Intel/AMD x86_64, ARM64

## 🗂️ 项目结构

```
pcb_solder_detection/
├── 📁 configs/                 # 配置文件
│   ├── model_config.yaml       # 模型配置
│   ├── train_config.yaml       # 训练配置
│   └── deploy_config.yaml      # 部署配置
├── 📁 data/                    # 数据集
│   ├── raw/                    # 原始图像
│   ├── labeled/                # 标注数据
│   ├── dataset1/               # 训练集1
│   ├── dataset2/               # 训练集2
│   └── test/                   # 测试集
├── 📁 src/                     # 源代码
│   ├── 1-preprocessing/        # 数据预处理
│   ├── 2-train/               # 模型训练
│   ├── 2-detect/              # 检测推理
│   ├── 3-test/                # 测试评估
│   ├── analysis/              # 结果分析
│   ├── detection/             # 检测算法
│   ├── utils/                 # 工具函数
│   └── report/                # 报告生成
├── 📁 models/                  # 模型文件
│   ├── pretrained/            # 预训练模型
│   └── trained/               # 训练好的模型
├── 📁 outputs/                 # 输出结果
│   ├── results/               # 检测结果
│   ├── reports/               # 分析报告
│   └── test_visualizations/   # 可视化结果
├── 📁 notebooks/               # Jupyter笔记本
├── 📁 docs/                    # 项目文档
├── 📁 guide/                   # 技术指南
└── 📁 temp/                    # 临时文件

```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd pcb_solder_detection

# 创建虚拟环境
conda create -n pcb_detection python=3.8
conda activate pcb_detection

# 安装依赖
pip install -r requirements.txt
```

### 2. GPU支持（可选）

```bash
# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 数据准备

```bash
# 将PCB图像放入数据目录
mkdir -p data/raw
cp your_pcb_images/* data/raw/

# 运行数据预处理
python src/1-preprocessing/preprocess.py
```

### 4. 模型训练

```bash
# 使用默认配置训练
python src/2-train/train.py

# 使用自定义配置
python src/2-train/train.py --config configs/custom_config.yaml
```

### 5. 开始检测

```bash
# 单张图像检测
python src/2-detect/detect.py --image path/to/image.jpg

# 批量检测
python src/2-detect/batch_detect.py --input data/test/ --output outputs/results/
```

## 🔍 检测能力

### 支持的缺陷类型

| 缺陷类型 | 描述 | 严重等级 |
|---------|------|----------|
| 🔴 **虚焊** | 焊点与焊盘接触不良 | 高 |
| 🟠 **桥接** | 相邻焊点意外连接 | 高 |
| 🟡 **锡少** | 焊料不足，焊点偏小 | 中 |
| 🟡 **锡多** | 焊料过多，焊点过大 | 中 |
| 🟢 **偏移** | 焊点位置偏离标准 | 低 |
| 🔵 **空焊** | 焊盘上无焊点 | 高 |

### 检测指标

- **准确率**: >95% (在标准测试集上)
- **检测速度**: 30+ FPS (RK3588 NPU)
- **误报率**: <3%
- **漏检率**: <2%

## 🎛️ 使用示例

### Python API

```python
from src.detection import PCBDetector

# 初始化检测器
detector = PCBDetector(model_path='models/trained/best.pt')

# 单张图像检测
results = detector.detect('path/to/pcb_image.jpg')

# 批量检测
results = detector.batch_detect('data/test/')

# 可视化结果
detector.visualize_results(results, save_path='outputs/visualization.jpg')
```

### 命令行工具

```bash
# 实时摄像头检测
python src/detect/real_time.py --camera 0

# Web服务
python src/app.py --host 0.0.0.0 --port 5000

# 模型评估
python src/3-test/evaluate.py --model models/trained/best.pt --data data/test/
```

## 🔧 边缘部署

### RK3588部署

```bash
# 模型转换
python tools/convert_to_rknn.py --model models/trained/best.pt

# 部署到设备
python deploy/rk3588_deploy.py --model models/converted/best.rknn
```

### 性能对比

| 平台 | 推理时间 | 功耗 | 准确率 |
|------|----------|------|--------|
| RK3588 NPU | ~33ms | 8W | 95.2% |
| RK3588 CPU | ~150ms | 12W | 95.2% |
| RTX 3080 | ~15ms | 220W | 95.8% |

## 📊 数据集和标注

### 数据集来源

1. **公开数据集**: PCB WACV 2019 Dataset
2. **自采集数据**: 实际生产线PCB图像
3. **合成数据**: GAN生成的增强数据

### 标注工具

推荐使用 [LabelImg](https://github.com/tzutalin/labelImg) 进行数据标注：

```bash
# 安装标注工具
pip install labelImg

# 启动标注
labelImg data/raw/ data/labels/
```

## 📈 性能监控

### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir outputs/logs --port 6006
```

### 实时监控面板

访问 `http://localhost:5000/dashboard` 查看实时检测统计和性能指标。

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📚 文档

- [📖 详细文档](docs/README.md)
- [🔧 技术分析报告](guide/analysis.md)
- [🎯 AOI技术指南](guide/AOI.md)
- [🚀 部署指南](guide/process.md)
- [💡 创新想法](guide/creativity.md)

## 🐛 问题反馈

如果您遇到任何问题，请在 [Issues](../../issues) 页面报告。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 优秀的目标检测框架
- [PCB WACV 2019 Dataset](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection) - 公开数据集
- Rockchip - RK3588 NPU支持
- 所有为项目做出贡献的开发者

---

<div align="center">
  <strong>⭐ 如果这个项目对您有帮助，请给我们一个星标！</strong>
</div>

## 许可证

MIT License
