import os
import glob
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def run_prediction(project_root, image_source, output_dir_name="prediction_results"):
    """
    使用训练好的YOLO模型对用户指定的图片或文件夹进行预测，
    并将绘制了检测框的结果图保存到指定目录。

    Args:
        project_root (str): 项目的根目录绝对路径。
        image_source (str): 要预测的单个图片文件路径或图片文件夹路径。
        output_dir_name (str): 在 temp/ 目录下用于保存结果的文件夹名称。
    """
    # 1. 定义可能的模型路径
    model_paths_to_check = [
        os.path.join(project_root, 'models', 'detect', 'yolov8s_pcb_solder', 'weights', 'best.pt'),
        os.path.join(project_root, 'models', 'detect', 'yolov8n_pcb_solder', 'weights', 'best.pt'),
        os.path.join(project_root, 'models', 'detect', 'best.pt')
    ]

    model_path = None
    for path in model_paths_to_check:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("❌ 错误: 在以下路径均未找到模型文件:")
        for path in model_paths_to_check:
            print(f"  - {path}")
        print("请确认模型训练是否成功。")
        return

    # 2. 定义输出目录
    output_dir = os.path.join(project_root, 'temp', output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"🖼️  结果图片将保存至: {output_dir}")

    # 3. 加载YOLO模型
    print(f"🧠 正在从 '{model_path}' 加载模型...")
    try:
        model = YOLO(model_path)
        print("✅ 模型加载成功。")
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return

    # 4. 获取要预测的图片列表
    image_paths = []
    if os.path.isfile(image_source):
        image_paths.append(image_source)
    elif os.path.isdir(image_source):
        # 支持多种常见图片格式
        supported_formats = ['*.jpg', '*.jpeg', '*.png']
        for fmt in supported_formats:
            image_paths.extend(glob.glob(os.path.join(image_source, fmt)))
    
    if not image_paths:
        print(f"❌ 错误: 在路径 '{image_source}' 中找不到任何有效的图片文件。")
        return

    print(f"\n🚀 将对 {len(image_paths)} 张图片进行预测...")

    # 5. 遍历所有图片进行预测和可视化
    for image_path in tqdm(image_paths, desc="预测进度"):
        try:
            # 执行预测, verbose=False 让输出更简洁
            results = model.predict(image_path, verbose=False)
            result = results[0]

            # 绘制结果并保存
            plotted_image = result.plot()
            
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, plotted_image)

        except Exception as e:
            print(f"\n处理图片 '{os.path.basename(image_path)}' 时发生错误: {e}")

    print(f"\n🎉 处理完成！ {len(image_paths)} 张图片的检测结果已保存。")

if __name__ == '__main__':
    # 动态计算项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    print(f"📍 项目根目录检测为: {project_root}")

    # --- 用户自定义区域 ---
    # 请在这里修改为您想预测的图片文件路径或文件夹路径
    # 示例1: 单个文件
    # CUSTOM_IMAGE_PATH = os.path.join(project_root, 'data', '2700', 'JPEGImages', '50-03_jpg.rf.24768e75ec99bd1bf6a2f6c2094117a4.jpg')
    # 示例2: 整个文件夹
    CUSTOM_IMAGE_PATH = os.path.join(project_root, 'data', 'users') 
    
    # 为这次预测的结果指定一个输出文件夹名称
    OUTPUT_FOLDER_NAME = "my_test_results"
    # --- 用户自定义区域结束 ---

    if not os.path.exists(CUSTOM_IMAGE_PATH):
        print(f"❌ 错误: 您指定的路径不存在: {CUSTOM_IMAGE_PATH}")
        print("请在脚本中修改 CUSTOM_IMAGE_PATH 变量。")
    else:
        run_prediction(
            project_root=project_root, 
            image_source=CUSTOM_IMAGE_PATH,
            output_dir_name=OUTPUT_FOLDER_NAME
        )
