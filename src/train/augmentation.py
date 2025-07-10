# augmentation.py
"""
数据增强配置文件
提供多种数据增强策略供选择
"""
import cv2
import numpy as np
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("警告: albumentations库未安装，某些数据增强功能将不可用")

def get_light_augmentation():
    """轻度数据增强 - 适合数据质量较好的情况"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
    ])

def get_medium_augmentation():
    """中度数据增强 - 平衡增强效果和数据质量"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.4
        ),
        
        # 颜色调整
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3
        ),
        
        # 质量增强
        A.CLAHE(p=0.2),
    ])

def get_heavy_augmentation():
    """强度数据增强 - 适合数据量较少需要大量增强的情况"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),
        
        # 颜色和亮度变换
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),
        
        # 噪声和模糊
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.Blur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),
        
        # 质量增强
        A.CLAHE(p=0.3),
        A.Sharpen(p=0.2),
        
        # 高级变换
        A.OneOf([
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.1),
        ], p=0.1),
    ])

def get_custom_augmentation(height=512, width=512):
    """
    一个自定义的数据增强流水线。
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
        
    return A.Compose([
        # --- 1. 几何变换 ---
        # 保持较大范围的随机裁剪和缩放，这是增加多样性的核心
        A.RandomResizedCrop(size=(height, width), scale=(0.5, 1.0), p=0.5),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 适度的旋转和仿射变换
        A.ShiftScaleRotate(
            shift_limit=0.6,      # 平移范围
            scale_limit=0.3,       # 缩放范围
            rotate_limit=55,       # 旋转角度从45度降到15度
            border_mode=cv2.BORDER_CONSTANT,
            p=0.6
        ),
        
        # 大幅降低透视变换的概率，因为它很容易产生严重失真
        A.Perspective(scale=(0.05, 0.1), p=0.1),

        # --- 2. 色彩与光照变换 ---
        
        # 降低亮度和对比度的变化范围
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25), 
            contrast_limit=(-0.25, 0.25), 
            p=0.7
        ),
        
        # 降低色调和饱和度的变化范围，以保持PCB颜色基本可辨认
        A.HueSaturationValue(
            hue_shift_limit=20, # 色调变化范围
            sat_shift_limit=30, # 饱和度变化范围
            val_shift_limit=25, # 明度变化范围
            p=0.5
        ),

        # --- 3. 清晰度、噪声与伪影 ---
        
        # 保持原有强度，这些是常见的真实世界噪声
        A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),

            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
            ], p=0.2)
        ])

# 增强策略字典
AUGMENTATION_STRATEGIES = {
    'none': None,
    'light': get_light_augmentation,
    'medium': get_medium_augmentation,
    'heavy': get_heavy_augmentation,
    'custom': get_custom_augmentation,  # 只存函数，不要提前调用
}

def get_augmentation(strategy='medium', height=640, width=640, dataset=None):
    """
    获取指定的数据增强策略
    
    Args:
        strategy (str): 增强策略名称 ['none', 'light', 'medium', 'heavy', 'custom']
        dataset: 若需CopyPaste增强，需传入SolderDataset对象
    
    Returns:
        albumentations.Compose 或 None
    """
    if strategy not in AUGMENTATION_STRATEGIES:
        print(f"警告: 未知的增强策略 '{strategy}'，使用默认的 'medium' 策略")
        strategy = 'medium'
    
    if strategy == 'none':
        return None
    
    aug_func = AUGMENTATION_STRATEGIES[strategy]
    if aug_func is None:
        return None
    
    if strategy == 'custom':
        return aug_func(height=height, width=width)  # 不再传dataset参数
    
    return aug_func(height=height, width=width) if callable(aug_func) else aug_func

class CopyPaste(object):
    """
    简单的分割任务“复制-粘贴”增强：
    随机从其他图片复制前景区域（mask>0），粘贴到当前图片随机位置。
    适用于小目标分割任务。
    用法：在albumentations.Compose中加入 CopyPaste(dataset=...) 变换。
    参数：
        dataset: SolderDataset对象，用于提供源图像和掩码
        p: 变换的概率
        max_paste: 最大粘贴次数
    """
    def __init__(self, dataset, p=0.5, max_paste=2):
        self.dataset = dataset  # 需传入SolderDataset对象或类似接口
        self.p = p
        self.max_paste = max_paste

    def __call__(self, image, mask):
        import random
        if random.random() > self.p:
            return {'image': image, 'mask': mask}
        h, w = image.shape[:2]
        for _ in range(random.randint(1, self.max_paste)):
            # 随机选一张源图
            idx = random.randint(0, len(self.dataset)-1)
            src_img, src_mask = self.dataset[idx]
            src_img = (src_img.numpy().transpose(1,2,0)*255).astype('uint8') if hasattr(src_img, 'numpy') else src_img
            src_mask = src_mask.numpy() if hasattr(src_mask, 'numpy') else src_mask
            # 随机选一个前景类别
            fg_classes = [c for c in np.unique(src_mask) if c>0]
            if not fg_classes:
                continue
            fg_cls = random.choice(fg_classes)
            fg_mask = (src_mask==fg_cls).astype('uint8')
            # 找到所有前景连通区域
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = random.choice(contours)
            x,y,wc,hc = cv2.boundingRect(cnt)
            # 裁剪区域
            patch_img = src_img[y:y+hc, x:x+wc].copy()
            patch_mask = fg_mask[y:y+hc, x:x+wc].copy()*fg_cls
            # 随机粘贴位置
            px = random.randint(0, w-wc) if w-wc>0 else 0
            py = random.randint(0, h-hc) if h-hc>0 else 0
            # 粘贴（mask为0处才粘贴，避免覆盖原有目标）
            region_mask = mask[py:py+hc, px:px+wc]
            region_img = image[py:py+hc, px:px+wc]
            paste_area = (patch_mask>0) & (region_mask==0)
            region_img[paste_area] = patch_img[paste_area]
            region_mask[paste_area] = patch_mask[paste_area]
            image[py:py+hc, px:px+wc] = region_img
            mask[py:py+hc, px:px+wc] = region_mask
        return {'image': image, 'mask': mask}
