import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc 
import psutil  
import time
from contextlib import nullcontext
import torch.amp

# 从我们自己写的文件中导入模型和数据集类
from unet import create_model, get_model_info
from dataset import SolderDataset
from augmentation import get_augmentation
from model_configs import get_config, list_all_configs
from PIL import Image
import glob

# --- 1. 配置训练参数 ---
torch.backends.cudnn.benchmark = True  # 卷积加速

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # 如果你的显存不够，可以调小这个值，比如2或1
EPOCHS = 20 # 训练轮次，先用一个较小的值跑通
IMG_DIR = './data/dataset_0706/processed' # 你的图片文件夹路径
JSON_DIR = './data/dataset_0706/json' # 你的JSON文件夹路径
MODEL_SAVE_PATH = './models/trained/unet_solder_model.pth' # 训练好的模型保存路径

# 数据增强配置
AUGMENTATION_STRATEGY = 'custom'  # 可选: 'none', 'light', 'medium', 'heavy', 'custom'

# 模型配置方式1: 使用预定义配置（推荐）
MODEL_CONFIG = 'pcb_optimized'  # 可选: 'basic', 'recommended', 'pcb_optimized', 'high_performance', 'efficient', 'lightweight', 'modern'

# 模型配置方式2: 手动指定（如果不使用预定义配置，可以取消注释下面的行）
# MODEL_NAME = 'unet'              # 可选: 'unet', 'unetplusplus', 'deeplabv3', 'deeplabv3plus', 'fpn', 'pspnet', 'linknet'
# ENCODER_NAME = 'resnet34'        # 可选: 'resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', etc.
# ENCODER_WEIGHTS = 'imagenet'     # 可选: 'imagenet', None (None表示随机初始化)

# 定义类别和映射。这个必须和dataset.py以及你的数据保持一致
# 注意：类别0通常留给背景，所以我们的物体从1开始
CLASS_MAPPING = {
    'background': 0,
    'good': 1,
    'insufficient': 2,
    'excess': 3,
    'shift': 4,
    'miss': 5,
    # --- 添加你所有的类别 ---
}
NUM_CLASSES = len(CLASS_MAPPING)


def train_fn(loader, model, optimizer, loss_fn, scaler=None):
    """一轮训练的逻辑，支持AMP混合精度"""
    model.train() 
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE, non_blocking=True)
        targets = targets.to(device=DEVICE, non_blocking=True)
        optimizer.zero_grad()
        # 支持AMP
        context = torch.amp.autocast('cuda') if scaler is not None else nullcontext()
        with context:
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        del data, targets, predictions, loss
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total_loss / len(loader)

def validate_fn(loader, model, loss_fn, scaler=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(loader, desc="Validation")
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE, non_blocking=True)
            targets = targets.to(device=DEVICE, non_blocking=True)
            context = torch.amp.autocast('cuda') if scaler is not None else nullcontext()
            with context:
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            del data, targets, predictions, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total_loss / len(loader)

# --- 添加内存监控功能 ---
def print_memory_usage():
    """打印当前内存使用情况"""
    if torch.cuda.is_available():
        print(f"GPU内存已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU内存已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"CPU内存使用: {mem_info.rss / 1024**3:.2f} GB")

def print_gpu_utilization():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} | 已分配: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB | 已缓存: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")

def main():
    print(f"正在使用设备: {DEVICE}")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
        print(f"检测到 {torch.cuda.device_count()} 块GPU: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    # 获取模型配置
    if 'MODEL_CONFIG' in globals(): 
        # 使用预定义配置
        config = get_config(MODEL_CONFIG)
        model_name = config['model_name']
        encoder_name = config['encoder_name']
        encoder_weights = config['encoder_weights']
        print(f"\n📋 使用预定义配置: {MODEL_CONFIG}")
        print(f"   描述: {config['description']}")
    
    # 显示模型信息
    model_info = get_model_info()
    print(f"\n🏗️ 模型配置:")
    print(f"  架构: {model_name}")
    print(f"  编码器: {encoder_name}")
    print(f"  预训练权重: {encoder_weights}")
    print(f"  可用架构: {', '.join(model_info['models'][:5])}...")
    print(f"  推荐编码器: {', '.join(model_info['encoders'][:5])}...")

    # 1. 创建数据集和数据加载器
    # 获取训练集中第一张图片的尺寸
    first_image_path = glob.glob(os.path.join(IMG_DIR, "*.jpg"))[0]  # 假设图片为jpg格式
    with Image.open(first_image_path) as img:
        width, height = img.size
    print(f"样本图片尺寸: {width}x{height}")

    # 先初始化不带transform的数据集（仅custom增强需要dataset）
    train_dataset_plain = SolderDataset(
        image_dir=IMG_DIR,
        json_dir=JSON_DIR,
        class_mapping={k: v for k, v in CLASS_MAPPING.items() if k != 'background'},
        transform=None,
        use_augmentation=False
    )
    if AUGMENTATION_STRATEGY == 'custom':
        augmentation_transform = get_augmentation(AUGMENTATION_STRATEGY, height=height, width=width, dataset=train_dataset_plain)
    else:
        augmentation_transform = get_augmentation(AUGMENTATION_STRATEGY, height=height, width=width, dataset=None)
    print(f"使用数据增强策略: {AUGMENTATION_STRATEGY}")

    # 用带transform的train_dataset覆盖
    train_dataset = SolderDataset(
        image_dir=IMG_DIR,
        json_dir=JSON_DIR,
        class_mapping={k: v for k, v in CLASS_MAPPING.items() if k != 'background'},
        transform=augmentation_transform,
        use_augmentation=True if augmentation_transform is not None else False
    )
    
    # 自动调整num_workers和pin_memory
    num_cpu_cores = os.cpu_count()
    if torch.cuda.is_available():
        workers = 10 if num_cpu_cores is not None else 2
        pin_memory = True
    else:
        workers = max(2, num_cpu_cores // 2) if num_cpu_cores else 2
        pin_memory = False
    print(f"CPU核心数: {num_cpu_cores}, DataLoader将使用 {workers} 个工作进程。")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True
    )
    # 2. 初始化模型、损失函数和优化器
    print(f"\n🏗️ 创建模型...")
    model = create_model(
        n_channels=3, 
        n_classes=NUM_CLASSES,
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 块GPU进行DataParallel训练")
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    loss_fn = nn.CrossEntropyLoss() # 多分类交叉熵损失，是分割任务的标准选择
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # AMP混合精度
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    # 3. 开始训练循环
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        start_time = time.time()
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        print_memory_usage()
        print_gpu_utilization()
        print(f"Epoch耗时: {time.time()-start_time:.1f}秒")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    # 4. 保存训练好的模型
    print("\n训练完成，正在保存模型...")
    # 确保保存目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")
    
    # 最终内存清理
    del model, optimizer, loss_fn, train_loader, train_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
