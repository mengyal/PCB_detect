import os
import cv2
import numpy as np
import random
from augmentation import get_augmentation

# 输入图片目录和输出文件夹
IMG_DIR = './data/dataset_0706/jpg'  # 你的图片目录
OUT_DIR = 'temp/custom'
os.makedirs(OUT_DIR, exist_ok=True)

# 获取所有图片文件名
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if len(img_files) == 0:
    raise RuntimeError('图片目录为空！')

# 随机采样100张（如不足100张则全用）
sample_files = random.sample(img_files, min(100, len(img_files)))

for i, fname in enumerate(sample_files):
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8)
    aug = get_augmentation('custom', height=height, width=width)
    augmented = aug(image=img, mask=mask)
    aug_img = augmented['image']
    out_path = os.path.join(OUT_DIR, f'aug_{i:03d}.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

print(f'已随机增强{len(sample_files)}张图片，保存在 {OUT_DIR} ，请用图片浏览器快速翻看。')
