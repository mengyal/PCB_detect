import cv2
import numpy as np
import os
import glob

'''待优化：图像增强方面的参数'''

# 全局路径变量
INPUT_DIR = "./data/dataset_0706/jpg"
OUTPUT_DIR = "./data/dataset_0706/processed"

class ImagePreprocessor:
    """图像预处理器类，提供快速的图像预处理功能"""
    
    def __init__(self):
        pass
    
    def to_grayscale(self, image):
        """
        将图像转换为灰度图像
        
        Args:
            image: 输入图像
            
        Returns:
            gray_image: 灰度图像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def downsample(self, image, factor=2):
        """
        下采样图像（降低分辨率）
        
        Args:
            image: 输入图像
            factor: 下采样因子
            
        Returns:
            downsampled_image: 下采样后的图像
        """
        height, width = image.shape[:2]
        new_width = width // factor
        new_height = height // factor
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    
    def resize(self, image, new_size=(512, 512)):
        """
        调整图像大小到512x512
        
        Args:
            image: 输入图像
            
        Returns:
            resized_image: 调整后的图像
        """
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    def preprocess_pipeline(self, image, filename=None, resizer=True, grayscale=False, clip_limit=2.0, tile_grid_size=(8, 8), enhance_contrast=False):
        """
        图像预处理流水线
        
        Args:
            image: 输入图像
            grayscale: 是否转换为灰度图像
            hsv: 是否转换为HSV图像

        Returns:
            processed_image: 处理后的图像
        """
        processed = image.copy()
        
        # 灰度转换
        if grayscale:
            processed = self.to_grayscale(processed)
            
        # 图像清晰度调整
        if resizer:
            processed = processed.copy()
            processed = self.resize_to_512(processed)
        
        # 对比度增强
        if enhance_contrast:
            processed = self.enhance_contrast(processed, alpha=1.5, beta=0)

        return processed

def main():
    preprocessor = ImagePreprocessor()

    # 创建输出目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    # 获取所有图像文件
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, extension)))
    
    if not image_files:
        print(f"在 {INPUT_DIR} 目录中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")

    for img_path in image_files:
        # 读取图像
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 获取文件名（不带路径）
        filename = os.path.basename(img_path)
        print(f"处理图像: {filename}")
        
        # 快速预处理
        processed_image = preprocessor.preprocess_pipeline(image)
        
        # 保存结果
        output_path = os.path.join(OUTPUT_DIR, f"{filename}")
        cv2.imwrite(output_path, processed_image)
        print(f"保存处理后的图像: {output_path}")
    
    cv2.destroyAllWindows()
    print("所有图像处理完成")

if __name__ == "__main__":
    main()
