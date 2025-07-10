import os
import cv2
import json
import numpy as np
import sys

def extract_foreground_patches(image, mask, min_area=10):
    """
    从mask中提取所有前景目标patch（小图+小mask+类别），返回列表
    Args:
        image: 原图 (H, W, 3)
        mask: 掩码 (H, W)，前景区域为类别id，背景为0
        min_area: 最小目标面积，过滤噪声
    Returns:
        [(patch_img, patch_mask, class_id), ...]
    """
    patches = []
    for cls in np.unique(mask):
        if cls == 0:
            continue
        fg_mask = (mask == cls).astype('uint8')
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_area:
                continue
            patch_img = image[y:y+h, x:x+w].copy()
            patch_mask = fg_mask[y:y+h, x:x+w].copy() * cls
            patches.append((patch_img, patch_mask, cls))
    return patches

IMG_DIR = './data/dataset1/jpg'
JSON_DIR = './data/dataset1/json'
PATCH_IMG_DIR = './data/dataset1/patch_jpg'
PATCH_JSON_DIR = './data/dataset1/patch_json'
os.makedirs(PATCH_IMG_DIR, exist_ok=True)
os.makedirs(PATCH_JSON_DIR, exist_ok=True)

img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]

def print_progress(cur, total, bar_len=30):
    filled = int(bar_len * cur / total)
    bar = '█' * filled + '-' * (bar_len - filled)
    print(f'\r进度: |{bar}| {cur}/{total}', end='')

# 标签映射（可根据实际情况修改）
CLASS_ID_TO_NAME = {
    1: 'good',
    2: 'insufficient',
    3: 'excess',
    4: 'shift',
    5: 'miss',
    # 如有更多类别请补充
}

for idx, img_file in enumerate(img_files, 1):
    print_progress(idx, len(img_files))
    base = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMG_DIR, img_file)
    json_path = os.path.join(JSON_DIR, base + '.json')
    if not os.path.exists(json_path):
        continue
    img = cv2.imread(img_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    mask = np.zeros((ann['imageHeight'], ann['imageWidth']), dtype=np.uint8)
    for shape in ann['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        class_id = 1  # 可根据实际类别映射调整
        if shape['shape_type'] == 'polygon':
            cv2.fillPoly(mask, [points], color=class_id)
        elif shape['shape_type'] == 'rectangle':
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=class_id, thickness=-1)
        elif shape['shape_type'] == 'circle':
            center = tuple(points[0].astype(int))
            radius = int(np.linalg.norm(points[1] - points[0]))
            cv2.circle(mask, center, radius, color=class_id, thickness=-1)
    patches = extract_foreground_patches(img, mask)
    for i, (patch_img, patch_mask, cls) in enumerate(patches):
        patch_img_name = f'{base}_patch_{i}.jpg'
        patch_json_name = f'{base}_patch_{i}.json'
        patch_img_path = os.path.join(PATCH_IMG_DIR, patch_img_name)
        patch_json_path = os.path.join(PATCH_JSON_DIR, patch_json_name)
        cv2.imwrite(patch_img_path, patch_img)
        # 生成json标注
        h, w = patch_mask.shape
        shapes = []
        fg_mask = (patch_mask == cls).astype('uint8')
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            points = cnt.squeeze().tolist()
            if len(points) < 3:
                continue
            shapes.append({
                'label': CLASS_ID_TO_NAME.get(cls, str(cls)),
                'points': points,
                'shape_type': 'polygon',
            })
        patch_ann = {
            'imagePath': patch_img_name,
            'imageHeight': h,
            'imageWidth': w,
            'shapes': shapes
        }
        with open(patch_json_path, 'w', encoding='utf-8') as f:
            json.dump(patch_ann, f, ensure_ascii=False, indent=2)
print('\nPatch导出完成！')
