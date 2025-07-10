"""
YOLOv8s PyTorch模型导出ONNX脚本
需先安装ultralytics: pip install ultralytics

直接运行即可，按需修改YOLO_PTH_PATH和ONNX_OUTPUT_PATH。
"""
from ultralytics import YOLO

# ====== 直接指定输入输出路径 ======
YOLO_PTH_PATH = './models/detect/analysis.pt'   # 修改为你的YOLOv8s模型路径
ONNX_OUTPUT_PATH = './models/onnx/yolov8s_analysis.onnx'    # 修改为导出的ONNX路径
IMG_SIZE = (640, 640)                       # 输入图片尺寸
# ====================================

def main():
    model = YOLO(YOLO_PTH_PATH)
    model.export(format='onnx', imgsz=IMG_SIZE, dynamic=True, simplify=True, opset=12, output=ONNX_OUTPUT_PATH)
    print(f'YOLOv8s 导出成功: {ONNX_OUTPUT_PATH}')

if __name__ == '__main__':
    main()
