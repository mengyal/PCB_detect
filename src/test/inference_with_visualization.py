import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
from datetime import datetime
import importlib.util

# 添加父目录到路径，以便导入训练模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
train_dir = os.path.join(parent_dir, '2-train')
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, train_dir)

# --- 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(project_root, 'models', 'trained', 'unetpp_solder_model.pth')
INPUT_DIR = os.path.join(project_root, 'data', 'users')
OUTPUT_DIR = os.path.join(project_root, 'temp1','plus')
TARGET_SIZE = (512, 512)  # 模型训练时的输入尺寸

# 使用pcb_optimized配置
MODEL_CONFIG = 'pcb_optimized' # 这里可以替换为其他配置，如 'recommended', 'basic', 等

# 类别映射（与训练时保持一致）
CLASS_MAPPING = {
    'background': 0,
    'good': 1,
    'insufficient': 2,
    'excess': 3,
    'shift': 4,
    'miss': 5,
}

ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

# 类别颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 0),        # background - 黑色
    1: (0, 255, 0),      # good - 绿色
    2: (0, 0, 255),      # insufficient - 红色
    3: (0, 165, 255),    # excess - 橙色
    4: (255, 0, 255),    # shift - 紫色
    5: (255, 255, 0),    # miss - 黄色
}

# 透明度设置
OVERLAY_ALPHA = 0.6
MASK_ALPHA = 0.4

def load_model_with_config():
    """手动加载模型"""
    print(f"🔧 加载模型配置和架构...")

    try:
        # 动态导入需要的模块
        unet_path = os.path.join(train_dir, 'unet.py')
        config_path = os.path.join(train_dir, 'model_configs.py')

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
            n_classes=NUM_CLASSES,
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )

        return model, config

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

def preprocess_image(image, target_size):
    """预处理图片"""
    # 保存原始尺寸
    original_size = (image.shape[1], image.shape[0])
    
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整尺寸
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size

def postprocess_prediction(prediction, original_size):
    """后处理预测结果"""
    # 调整回原始尺寸
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
    # 创建叠加图像（原图+掩码）
    overlay = cv2.addWeighted(original_image, OVERLAY_ALPHA, colored_mask, MASK_ALPHA, 0)
    return overlay

def create_side_by_side_visualization(original_image, colored_mask, overlay):
    """创建并排对比图像"""
    h, w = original_image.shape[:2]
    
    # 创建三联图：原图 | 掩码 | 叠加
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, 0:w] = original_image
    combined[:, w:2*w] = colored_mask
    combined[:, 2*w:3*w] = overlay
    
    # 添加分割线
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    cv2.line(combined, (2*w, 0), (2*w, h), (255, 255, 255), 2)
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(combined, 'Original', (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined, 'Segmentation', (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(combined, 'Overlay', (2*w + 10, 30), font, font_scale, color, thickness)
    
    return combined

def analyze_segmentation(prediction):
    """分析分割结果"""
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    analysis = {}
    for class_id, count in zip(unique, counts):
        class_name = ID_TO_CLASS.get(class_id, f"unknown_{class_id}")
        percentage = (count / total_pixels) * 100
        analysis[class_name] = {
            'pixels': int(count),
            'percentage': percentage
        }
    
    return analysis

def save_analysis_report(image_name, analysis, output_dir):
    """保存单张图片的分析报告"""
    report_path = os.path.join(output_dir, f"{image_name}_analysis.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"PCB焊点检测分析报告\n")
        f.write(f"图片: {image_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # 按重要性排序显示
        important_classes = ['good', 'insufficient', 'excess', 'shift', 'background']
        
        for class_name in important_classes:
            if class_name in analysis:
                stats = analysis[class_name]
                f.write(f"{class_name:12s}: {stats['pixels']:8d} 像素 ({stats['percentage']:6.2f}%)\n")
        
        # 显示其他类别
        for class_name, stats in analysis.items():
            if class_name not in important_classes:
                f.write(f"{class_name:12s}: {stats['pixels']:8d} 像素 ({stats['percentage']:6.2f}%)\n")
        
        f.write(f"\n总像素数: {sum(stats['pixels'] for stats in analysis.values())}\n")

def create_legend_image():
    """创建类别颜色图例"""
    legend_height = 200
    legend_width = 300
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    y_start = 30
    y_step = 35
    
    cv2.putText(legend, 'Class Legend:', (10, 20), font, 0.8, (0, 0, 0), 2)
    
    for i, (class_id, color) in enumerate(CLASS_COLORS.items()):
        if class_id == 0:  # 跳过背景
            continue
            
        class_name = ID_TO_CLASS[class_id]
        y_pos = y_start + i * y_step
        
        # 绘制颜色块
        cv2.rectangle(legend, (10, y_pos - 15), (40, y_pos + 5), color, -1)
        cv2.rectangle(legend, (10, y_pos - 15), (40, y_pos + 5), (0, 0, 0), 1)
        
        # 绘制类别名称
        cv2.putText(legend, class_name, (50, y_pos), font, font_scale, (0, 0, 0), thickness)
    
    return legend

def run_inference():
    """运行推理主程序"""
    print("🔬 PCB焊点检测推理工具")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 检查路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    try:
        model, config = load_model_with_config()
        
        # 加载权重
        print(f"📦 加载模型权重...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model = model.to(DEVICE)
        model.eval()
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 获取测试图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    test_images = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(image_extensions)]
    
    if not test_images:
        print(f"❌ 在 {INPUT_DIR} 中没有找到图片文件")
        return
    
    print(f"✓ 找到 {len(test_images)} 张图片")
    
    # 创建图例
    legend = create_legend_image()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "class_legend.jpg"), legend)
    
    # 处理图片
    successful_count = 0
    failed_count = 0
    
    for img_name in tqdm(test_images, desc="处理进度"):
        try:
            img_path = os.path.join(INPUT_DIR, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            # 读取图片
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"⚠️ 无法读取图片: {img_name}")
                failed_count += 1
                continue
            
            # 预处理
            input_tensor, original_size = preprocess_image(original_image, TARGET_SIZE)
            input_tensor = input_tensor.to(DEVICE)
            
            # 模型推理
            with torch.no_grad():
                outputs = model(input_tensor)
                predictions = torch.argmax(outputs, dim=1)
                prediction = predictions.cpu().numpy()[0]
            
            # 后处理
            prediction = postprocess_prediction(prediction, original_size)
            
            # 创建可视化
            colored_mask = create_colored_mask(prediction, CLASS_COLORS)
            overlay = create_overlay_visualization(original_image, colored_mask)
            side_by_side = create_side_by_side_visualization(original_image, colored_mask, overlay)
            
            # 保存结果
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg"), original_image)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask.jpg"), colored_mask)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.jpg"), overlay)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_comparison.jpg"), side_by_side)
            
            # 保存原始预测掩码（用于进一步分析）
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_raw_mask.png"), prediction)
            
            # 分析结果
            analysis = analyze_segmentation(prediction)
            save_analysis_report(base_name, analysis, OUTPUT_DIR)
            
            successful_count += 1
            
        except Exception as e:
            print(f"⚠️ 处理图片 {img_name} 时出错: {e}")
            failed_count += 1
            continue
    
    # 生成总结报告
    generate_summary_report(OUTPUT_DIR, successful_count, failed_count, len(test_images), config)
    
    print(f"\n✅ 推理完成!")
    print(f"✓ 成功处理: {successful_count} 张")
    print(f"✗ 失败: {failed_count} 张")
    print(f"📂 结果保存在: {OUTPUT_DIR}")

def generate_summary_report(output_dir, successful, failed, total, config):
    """生成总结报告"""
    report_path = os.path.join(output_dir, "inference_summary.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PCB焊点检测推理总结报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"推理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型配置: {MODEL_CONFIG}\n")
        f.write(f"模型架构: {config['model_name']}\n")
        f.write(f"编码器: {config['encoder_name']}\n")
        f.write(f"描述: {config['description']}\n\n")
        
        f.write(f"处理统计:\n")
        f.write(f"  总图片数: {total}\n")
        f.write(f"  成功处理: {successful}\n")
        f.write(f"  处理失败: {failed}\n")
        f.write(f"  成功率: {successful/total*100:.1f}%\n\n")
        
        f.write(f"输入目录: {INPUT_DIR}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"模型路径: {MODEL_PATH}\n\n")
        
        f.write("类别说明:\n")
        for class_name, class_id in CLASS_MAPPING.items():
            color = CLASS_COLORS[class_id]
            f.write(f"  {class_name:12s} (ID: {class_id}) - BGR{color}\n")
        
        f.write("\n生成的文件类型:\n")
        f.write("  *_original.jpg     - 原始图片\n")
        f.write("  *_mask.jpg         - 彩色分割掩码\n")
        f.write("  *_overlay.jpg      - 叠加可视化\n")
        f.write("  *_comparison.jpg   - 三联对比图\n")
        f.write("  *_raw_mask.png     - 原始预测掩码\n")
        f.write("  *_analysis.txt     - 像素统计分析\n")
        f.write("  class_legend.jpg   - 类别颜色图例\n")
        f.write("  inference_summary.txt - 本总结报告\n")

if __name__ == "__main__":
    run_inference()
