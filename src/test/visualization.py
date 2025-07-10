import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import importlib.util
import sys

# 配置参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_COLORS = {
    0: (0, 0, 0),        # background - 黑色
    1: (0, 255, 0),      # good - 绿色
    2: (0, 0, 255),      # insufficient - 红色
    3: (0, 165, 255),    # excess - 橙色
    4: (255, 0, 255),    # shift - 紫色
    5: (255, 255, 0),    # miss - 黄色
}
OVERLAY_ALPHA = 0.6
MASK_ALPHA = 0.4

def preprocess_image(image, target_size):
    """预处理图片"""
    original_size = (image.shape[1], image.shape[0])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, original_size

def postprocess_prediction(prediction, original_size):
    """后处理预测结果"""
    prediction_resized = cv2.resize(
        prediction.astype(np.uint8), 
        original_size, 
        interpolation=cv2.INTER_NEAREST
    )
    return prediction_resized

def create_colored_mask(prediction, class_colors):
    """创建彩色分割掩码"""
    h, w = prediction.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        mask = prediction == class_id
        colored_mask[mask] = color
    return colored_mask

def create_overlay_visualization(original_image, colored_mask):
    """创建叠加可视化图像"""
    overlay = cv2.addWeighted(original_image, OVERLAY_ALPHA, colored_mask, MASK_ALPHA, 0)
    return overlay


def save_visualizations(overlay, output_dir, base_name):
    """保存可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)
    
    # 同时保存到报告系统的可视化目录
    report_viz_dir = os.path.join(os.path.dirname(__file__), '..', 'temp', 'report_visualizations')
    os.makedirs(report_viz_dir, exist_ok=True)
    report_overlay_path = os.path.join(report_viz_dir, f"{base_name}_overlay.jpg")
    cv2.imwrite(report_overlay_path, overlay)
    
    print(f"可视化结果保存到: {overlay_path}")
    print(f"报告可视化保存到: {report_overlay_path}")
    return overlay_path, report_overlay_path

def load_model_with_config():
    """加载模型配置和权重"""
    print("🔧 加载模型配置和架构...")

    try:
        # 动态导入需要的模块
        unet_path = os.path.join(os.path.dirname(__file__), '..', '2-train', 'unet.py')
        config_path = os.path.join(os.path.dirname(__file__), '..', '2-train', 'model_configs.py')

        # 加载unet模块
        spec = importlib.util.spec_from_file_location("unet", unet_path)
        unet_module = importlib.util.module_from_spec(spec)
        sys.modules["unet"] = unet_module
        spec.loader.exec_module(unet_module)

        # 加载配置模块
        spec = importlib.util.spec_from_file_location("model_configs", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["model_configs"] = config_module
        spec.loader.exec_module(config_module)

        print("✓ 模块加载成功")

        # 获取用户选择的配置
        config = config_module.get_config(MODEL_CONFIG)
        model_name = config['model_name']
        encoder_name = config['encoder_name']
        encoder_weights = config['encoder_weights']

        print(f"📋 使用配置: {MODEL_CONFIG}")
        print(f"   架构: {model_name}")
        print(f"   编码器: {encoder_name}")
        print(f"   预训练权重: {encoder_weights}")
        print(f"   描述: {config['description']}")

        # 手动创建模型
        print(f"🏗️ 手动创建模型: {model_name}")
        model = unet_module.create_model(
            n_channels=3,
            n_classes=len(CLASS_COLORS),
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )

        # 加载模型权重
        print(f"📦 加载模型权重: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model = model.to(DEVICE)

        return model, config

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

def main():
    """主程序入口"""
    print("🔬 PCB焊点检测可视化工具")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # 检查路径
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取测试图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    test_images = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(image_extensions)]

    if not test_images:
        print(f"❌ 在 {INPUT_DIR} 中没有找到图片文件")
        return

    print(f"✓ 找到 {len(test_images)} 张图片")

    # 处理图片
    for img_name in tqdm(test_images, desc="处理进度"):
        try:
            img_path = os.path.join(INPUT_DIR, img_name)
            base_name = os.path.splitext(img_name)[0]

            # 读取图片
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"⚠️ 无法读取图片: {img_name}")
                continue

            # 预处理
            input_tensor, original_size = preprocess_image(original_image, TARGET_SIZE)

            # 模型推理
            model, config = load_model_with_config()
            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor.to(DEVICE))
                
                # 自动检测输出通道数并处理
                if len(prediction.shape) == 4 and prediction.shape[1] > 1:
                    # 多通道输出，使用argmax
                    prediction = torch.argmax(prediction, dim=1).cpu().numpy()
                else:
                    # 单通道输出，直接转换
                    prediction = prediction.squeeze().cpu().numpy()

            # 调试模型输出
            print(f"模型输出形状: {prediction.shape}")
            print(f"模型输出类型: {type(prediction)}")
            print(f"输出值范围: {prediction.min()} - {prediction.max()}")

            # 确保prediction是二维数组
            if len(prediction.shape) > 2:
                prediction = prediction[0]  # 取第一个批次
            
            # 后处理
            prediction = postprocess_prediction(prediction, original_size)

            # 创建可视化
            colored_mask = create_colored_mask(prediction, CLASS_COLORS)
            overlay = create_overlay_visualization(original_image, colored_mask)

            # 保存结果
            overlay_path, report_overlay_path = save_visualizations(overlay, OUTPUT_DIR, base_name)

        except Exception as e:
            print(f"⚠️ 处理图片 {img_name} 时出错: {e}")
            continue

    print(f"\n✅ 可视化完成!")
    print(f"📂 结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    INPUT_DIR = "../../data/users"  # 替换为实际输入目录路径
    OUTPUT_DIR = "../../temp/visualization"  # 替换为实际输出目录路径
    TARGET_SIZE = (512, 512)  # 模型训练时的输入尺寸
    MODEL_CONFIG = "pcb_optimized"  # 默认模型配置
    MODEL_PATH = "../../models/trained/unetpp_solder_model.pth"
    main()
