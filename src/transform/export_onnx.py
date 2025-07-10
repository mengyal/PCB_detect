"""
模型转换脚本：将 models 文件夹下的 PyTorch 模型（.pth/.pt）批量转换为 ONNX 格式。
支持常见的分割和检测模型（如 UNet、YOLO 等）。

用法：
    python export_onnx.py --model_path ../../models/trained/unet_solder_model.pth --output_path ./unet_solder_model.onnx --model_type unet
    # 或批量转换
    python export_onnx.py --auto_search

注意：
- 需根据实际模型结构补充/修改模型定义部分。
- 需安装 torch、onnx。
"""
import os
import sys
import torch
import onnx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../2-train')))
from unet import create_model_from_checkpoint

# 示例：UNet 模型结构（如有自定义结构请替换）
class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        # ...此处应补充你的UNet结构...
        # 这里只是一个占位示例
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)

# 示例：UNet++ 模型结构（简化版，如有自定义结构请替换）
class UNetPP(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        # 这里只是一个极简结构示例，实际请替换为你的Unet++实现
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)

# YOLOv8s ONNX导出直接用ultralytics包
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

MODEL_FACTORY = {
    'unet': lambda: UNet(in_channels=3, out_channels=2),
    'unetpp': lambda: UNetPP(in_channels=3, out_channels=2),
    # 'yolov8s' 不需要手动结构
}

def export_onnx(model_path, output_path, model_type, input_shape=(1, 3, 512, 512)):
    # 自动推断类别数
    checkpoint = torch.load(model_path, map_location='cpu')
    # 兼容不同保存格式
    state_dict = None
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # 推断类别数（取 segmentation_head.0.weight 的第0维）
    n_classes = None
    for k in state_dict.keys():
        if 'segmentation_head.0.weight' in k:
            n_classes = state_dict[k].shape[0]
            break
    if n_classes is None:
        n_classes = 2  # fallback
    model = create_model_from_checkpoint(model_path, n_channels=input_shape[1], n_classes=n_classes, model_name=None, encoder_name=None)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input'], output_names=['output'],
        opset_version=11, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"导出成功: {output_path}")

def auto_search_and_export(models_dir, output_dir, model_type_hint=None):
    for root, _, files in os.walk(models_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt'):
                model_path = os.path.join(root, f)
                fname = f.lower()
                if 'unetpp' in fname or 'unet++' in fname:
                    model_type = 'unetpp'
                elif 'unet' in fname:
                    model_type = 'unet'
                elif 'yolov8' in fname or 'yolov8s' in fname:
                    model_type = 'yolov8s'
                else:
                    if model_type_hint:
                        model_type = model_type_hint
                    else:
                        print(f"跳过未知类型: {f}")
                        continue
                output_path = os.path.join(output_dir, os.path.splitext(f)[0] + '.onnx')
                export_onnx(model_path, output_path, model_type)

# ====== 直接指定输入输出路径和模型类型 ======
# 单模型导出
MODEL_PATH = './models/trained/unetpp_solder_model.pth'  # 修改为你的模型路径
OUTPUT_PATH = './models/onnx/unetpp_solder_model.onnx'                   # 修改为你的ONNX输出路径
MODEL_TYPE = 'unetpp'                                        # 'unet'、'unetpp'、'yolov8s'等
INPUT_SHAPE = (1, 3, 512, 512)                             # 根据实际输入调整

# 批量导出（如不需要可忽略）
AUTO_SEARCH = False
MODELS_DIR = '../../models'    # 批量模式下模型目录
OUTPUT_DIR = './onnx_out'      # 批量模式下ONNX输出目录
# =========================================

def main():
    if AUTO_SEARCH:
        auto_search_and_export(MODELS_DIR, OUTPUT_DIR)
    else:
        export_onnx(MODEL_PATH, OUTPUT_PATH, MODEL_TYPE, INPUT_SHAPE)

if __name__ == '__main__':
    main()
