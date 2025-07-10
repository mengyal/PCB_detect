# PCB焊点检测YOLO训练系统

基于YOLOv8的PCB焊点检测模型训练系统。该系统可以将语义分割标注转换为目标检测标注，然后训练YOLO模型进行焊点缺陷检测。

## 功能特点

- **数据转换**: 将LabelMe语义分割标注转换为YOLO目标检测格式
- **模型训练**: 使用YOLOv8进行焊点检测模型训练
- **缺陷检测**: 支持5种焊点缺陷类型检测
- **可视化**: 提供标注可视化和检测结果可视化
- **批量处理**: 支持批量图片检测和分析

## 支持的缺陷类型

1. **background** (背景) - 类别ID: 0
2. **good** (良好) - 类别ID: 1
3. **insufficient** (不足) - 类别ID: 2
4. **excess** (过量) - 类别ID: 3
5. **shift** (偏移) - 类别ID: 4
6. **miss** (缺失) - 类别ID: 5

## 项目结构

```
src/2-train_YOLO/
├── main.py              # 主运行脚本
├── convert_data.py      # 数据转换脚本
├── train_yolo.py        # 模型训练脚本
├── inference.py         # 模型推理脚本
├── yolo.py             # 完整的训练管道
└── README.md           # 说明文档
```

## 环境要求

### Python版本
- Python 3.8+

### 依赖包
```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install pyyaml
pip install tqdm
pip install Pillow
```

## 快速开始

### 1. 数据准备

确保你的数据集结构如下：
```
data/dataset 0706/
├── jpg/                 # 图片文件
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── json/                # LabelMe标注文件
    ├── image1.json
    ├── image2.json
    └── ...
```

### 2. 一键运行

```bash
# 运行完整训练流程
python main.py
```

这将自动完成：
- 数据转换
- 模型训练
- 模型评估
- 推理测试

### 3. 分步运行

如果需要分步执行，可以使用单独的脚本：

#### 步骤1: 数据转换
```bash
python convert_data.py
```

#### 步骤2: 模型训练
```bash
python train_yolo.py
```

#### 步骤3: 模型推理
```bash
python inference.py
```

## 详细使用说明

### 数据转换

`convert_data.py`脚本功能：
- 将LabelMe多边形标注转换为YOLO边界框格式
- 自动划分训练集/验证集/测试集 (7:2:1)
- 生成数据集配置文件
- 提供标注可视化

关键参数：
- `data_root`: 原始数据集路径
- `output_dir`: YOLO数据集输出路径
- `train_ratio`: 训练集比例 (默认0.7)
- `val_ratio`: 验证集比例 (默认0.2)
- `test_ratio`: 测试集比例 (默认0.1)

### 模型训练

`train_yolo.py`脚本功能：
- 自动检测GPU可用性
- 根据GPU内存调整批次大小
- 使用数据增强提高模型泛化能力
- 自动保存最佳模型

训练参数：
- `model_size`: 模型大小 (n, s, m, l, x)
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `img_size`: 输入图片尺寸
- `conf_threshold`: 置信度阈值

### 模型推理

`inference.py`脚本功能：
- 单张图片检测
- 批量图片检测
- 检测结果可视化
- 质量分析和统计

使用示例：
```python
# 创建检测器
detector = PCBSolderDetector("path/to/best.pt")

# 检测单张图片
result = detector.detect_single_image("test.jpg")

# 批量检测
results = detector.detect_batch("test_images/", "output/")
```

## 输出文件说明

### 训练输出
- `runs/train/pcb_solder_detection/weights/best.pt`: 最佳模型权重
- `runs/train/pcb_solder_detection/weights/last.pt`: 最后一轮权重
- `runs/train/pcb_solder_detection/results.png`: 训练指标图表
- `runs/train/pcb_solder_detection/confusion_matrix.png`: 混淆矩阵

### 数据集输出
- `yolo_dataset/train/`: 训练集
- `yolo_dataset/val/`: 验证集
- `yolo_dataset/test/`: 测试集
- `yolo_dataset/dataset.yaml`: 数据集配置文件
- `yolo_dataset/visualizations/`: 标注可视化

### 检测输出
- `detection_results/`: 检测结果目录
- `*_detected.jpg`: 检测结果图片
- `*_analysis.txt`: 检测分析报告

## 性能优化建议

### 硬件要求
- **GPU**: 建议使用NVIDIA GPU (4GB+显存)
- **内存**: 建议16GB+系统内存
- **CPU**: 多核心CPU用于数据加载

### 训练优化
1. **批次大小**: 根据GPU内存调整
   - 4GB GPU: batch_size=8
   - 8GB GPU: batch_size=16
   - 12GB+ GPU: batch_size=32

2. **图片尺寸**: 根据数据集调整
   - 小目标: 640x640
   - 大目标: 1280x1280

3. **数据增强**: 适当调整增强强度
   - 数据量少: 增强强度高
   - 数据量多: 增强强度低

### 推理优化
1. **模型选择**:
   - YOLOv8n: 最快速度，适合实时应用
   - YOLOv8s: 平衡速度和精度
   - YOLOv8m/l/x: 最高精度，适合离线处理

2. **推理参数**:
   - `conf_threshold`: 置信度阈值，默认0.25
   - `iou_threshold`: NMS阈值，默认0.45

## 常见问题

### Q1: 数据转换失败
**A**: 检查JSON文件格式是否正确，确保包含`imageWidth`、`imageHeight`和`shapes`字段

### Q2: 训练过程中GPU内存不足
**A**: 减少批次大小或使用更小的模型 (如YOLOv8n)

### Q3: 模型精度不高
**A**: 
- 增加训练轮数
- 调整学习率
- 增加数据增强
- 使用更大的模型

### Q4: 检测结果误报多
**A**: 提高置信度阈值，或者增加负样本训练数据

## 进阶使用

### 自定义训练参数

修改`train_yolo.py`中的训练参数：
```python
train_args = {
    'epochs': 200,           # 增加训练轮数
    'batch': 32,             # 增加批次大小
    'lr0': 0.01,            # 初始学习率
    'weight_decay': 0.0005,  # 权重衰减
    'mosaic': 1.0,          # 马赛克增强
    'mixup': 0.1,           # 混合增强
    'copy_paste': 0.1,      # 复制粘贴增强
}
```

### 自定义检测阈值

修改`inference.py`中的检测参数：
```python
detector.detect_single_image(
    image_path,
    conf_threshold=0.5,     # 提高置信度阈值
    iou_threshold=0.4       # 调整NMS阈值
)
```

### 模型导出

将训练好的模型导出为其他格式：
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('best.pt')

# 导出为ONNX格式
model.export(format='onnx')

# 导出为TensorRT格式
model.export(format='engine')
```

## 技术支持

如有问题请检查：
1. 数据集路径是否正确
2. 依赖包是否完整安装
3. GPU驱动是否正常
4. 数据格式是否符合要求

## 版本更新

- v1.0: 初始版本，支持基本的检测功能
- v1.1: 增加批量检测和可视化功能
- v1.2: 优化训练参数和推理性能

## 新增功能 - 电子元件检测

### 文件说明

#### `component_detector.py`
完整的PCB电子元件检测器类，提供高级功能：

- **PCBComponentDetector类**: 完整的检测器实现
- **检测功能**: 检测16种不同类型的电子元件
- **可视化**: 生成带标注的检测结果图像
- **分析功能**: 提供详细的检测统计和分析
- **批量处理**: 支持批量图像检测

#### `simple_detector.py`
简化的电子元件检测接口，提供易用的API：

- **detect_pcb_components_simple()**: 简单的检测接口
- **get_component_locations()**: 获取特定元件位置
- **count_components()**: 统计元件数量
- **visualize_detections_simple()**: 简单的可视化功能

#### `demo_component_detection.py`
完整的使用示例和演示脚本

### 支持的电子元件类型

| 类别ID | 中文名称 | 英文名称 |
|--------|----------|----------|
| 0 | 电阻 | resistor |
| 1 | 集成电路 | IC |
| 2 | 二极管 | diode |
| 3 | 电容 | capacitor |
| 4 | 晶体管 | transistor |
| 5 | 电感 | inductor |
| 6 | 发光二极管 | LED |
| 7 | 连接器 | connector |
| 8 | 时钟 | clock |
| 9 | 开关 | switch |
| 10 | 电池 | battery |
| 11 | 蜂鸣器 | buzzer |
| 12 | 显示器 | display |
| 13 | 保险丝 | fuse |
| 14 | 继电器 | relay |
| 15 | 电位器 | potentiometer |

### 快速开始

#### 1. 简单检测
```python
from simple_detector import detect_pcb_components_simple

# 检测图像中的电子元件
result = detect_pcb_components_simple("path/to/image.jpg")

if result['success']:
    print(f"检测到 {result['total_count']} 个电子元件")
    print(f"元件分布: {result['component_types']}")
    
    # 查看详细结果
    for detection in result['detections']:
        print(f"{detection['label']}: 置信度 {detection['confidence']:.3f}")
        print(f"  位置: {detection['bbox']}")
```

#### 2. 高级检测
```python
from component_detector import PCBComponentDetector

# 创建检测器
detector = PCBComponentDetector(model_path="../../models/detect/best.pt")

# 检测并分析
result = detector.detect_and_analyze(
    image_input="path/to/image.jpg",
    conf_threshold=0.25,
    save_visualization="output/result.jpg"
)

if result['success']:
    print(f"检测摘要: {result['summary']}")
    print(f"检测详情: {result['detections']}")
```

#### 3. 获取特定元件位置
```python
from simple_detector import get_component_locations

# 获取所有电阻的位置
resistor_locations = get_component_locations("image.jpg", "resistor")
print(f"找到 {len(resistor_locations)} 个电阻")

# 获取所有元件的位置
all_locations = get_component_locations("image.jpg")
print(f"总共 {len(all_locations)} 个元件")
```

#### 4. 统计元件数量
```python
from simple_detector import count_components

# 统计各类型元件数量
counts = count_components("image.jpg")
for component, count in counts.items():
    print(f"{component}: {count} 个")
```

#### 5. 生成可视化结果
```python
from simple_detector import visualize_detections_simple

# 生成带标注的检测结果图像
success = visualize_detections_simple(
    image_path="input.jpg",
    output_path="output.jpg",
    confidence_threshold=0.25,
    show_labels=True
)
```

### 函数返回格式

#### 检测结果格式
```python
{
    'success': True,                    # 是否成功
    'detections': [                     # 检测结果列表
        {
            'label': 'resistor',        # 元件类型
            'confidence': 0.85,         # 置信度
            'bbox': {                   # 边界框
                'x1': 100, 'y1': 150,
                'x2': 200, 'y2': 250
            },
            'center': {'x': 150, 'y': 200}  # 中心点
        },
        ...
    ],
    'total_count': 5,                   # 总检测数量
    'component_types': {                # 各类型统计
        'resistor': 2,
        'capacitor': 1,
        'IC': 2
    }
}
```

### 使用示例

#### 批量检测示例
```python
import os
from pathlib import Path
from simple_detector import detect_pcb_components_simple

# 批量检测目录中的所有图像
image_dir = Path("path/to/images")
results = {}

for image_file in image_dir.glob("*.jpg"):
    result = detect_pcb_components_simple(str(image_file))
    if result['success']:
        results[image_file.name] = result['component_types']
        print(f"{image_file.name}: {result['total_count']} 个元件")

# 汇总统计
total_components = {}
for image_result in results.values():
    for component, count in image_result.items():
        total_components[component] = total_components.get(component, 0) + count

print(f"总计统计: {total_components}")
```

### 性能和参数调节

#### 置信度阈值
- **0.1-0.2**: 检测更多对象，但可能包含误检
- **0.25-0.4**: 平衡的检测结果（推荐）
- **0.5-0.8**: 更严格的检测，减少误检但可能遗漏

#### 模型路径
- 默认使用: `../../models/detect/best.pt`
- 可以指定其他训练好的模型路径

#### 输出控制
- 可以选择是否保存可视化结果
- 可以控制是否显示置信度和标签

### 注意事项

1. **模型文件**: 确保 `models/detect/best.pt` 文件存在
2. **图像格式**: 支持 jpg, png 等常见格式
3. **依赖包**: 需要安装 ultralytics, opencv-python
4. **内存使用**: 大图像可能需要较多内存
5. **GPU加速**: 如果有GPU，会自动使用GPU加速

### 故障排除

1. **模型文件不存在**: 检查模型路径是否正确
2. **图像无法读取**: 检查图像文件是否损坏
3. **检测结果为空**: 尝试降低置信度阈值
4. **内存不足**: 尝试处理较小的图像或使用CPU模式
