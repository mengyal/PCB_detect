import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

if __name__ == "__main__" or (hasattr(os, 'getppid') and os.getppid() == 1):
    if SMP_AVAILABLE:
        print("✓ segmentation_models_pytorch 可用")
    else:
        print("⚠️ segmentation_models_pytorch 未安装，使用自定义UNet")

def detect_model_architecture(checkpoint_path):
    """
    智能检测checkpoint中保存的模型架构
    
    Args:
        checkpoint_path (str): 模型权重文件路径
        
    Returns:
        tuple: (model_name, encoder_name, confidence)
            - model_name: 检测到的模型架构名称
            - encoder_name: 检测到的编码器名称（如果能识别）
            - confidence: 检测置信度 (0.0-1.0)
    """
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 获取state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 分析键名模式来判断模型架构
        keys = list(state_dict.keys())
        
        # 检测UNet++特有的模式 (x_i_j格式)
        unetpp_patterns = [
            'decoder.blocks.x_0_0',
            'decoder.blocks.x_1_0',
            'decoder.blocks.x_0_1',
            'decoder.blocks.x_2_0',
            'decoder.blocks.x_1_1',
            'decoder.blocks.x_0_2',
        ]
        
        unetpp_matches = sum(1 for pattern in unetpp_patterns 
                            if any(pattern in key for key in keys))
        
        # 检测DeepLabV3特有的模式
        deeplabv3_patterns = [
            'segmentation_head.aspp',
            'decoder.aspp',
            'classification_head'
        ]
        
        deeplabv3_matches = sum(1 for pattern in deeplabv3_patterns 
                               if any(pattern in key for key in keys))
        
        # 检测FPN特有的模式
        fpn_patterns = [
            'decoder.p5',
            'decoder.p4',
            'decoder.p3',
            'decoder.p2',
            'segmentation_head.conv'
        ]
        
        fpn_matches = sum(1 for pattern in fpn_patterns 
                         if any(pattern in key for key in keys))
        
        # 检测PSPNet特有的模式
        pspnet_patterns = [
            'decoder.psp',
            'decoder.conv1',
            'decoder.conv2'
        ]
        
        pspnet_matches = sum(1 for pattern in pspnet_patterns 
                            if any(pattern in key for key in keys))
        
        # 检测LinkNet特有的模式
        linknet_patterns = [
            'decoder.center',
            'decoder.block',
            'decoder.final_conv'
        ]
        
        linknet_matches = sum(1 for pattern in linknet_patterns 
                             if any(pattern in key for key in keys))
        
        # 检测编码器类型
        encoder_name = 'resnet34'  # 默认值
        if any('resnet' in key for key in keys):
            # 检查ResNet的具体类型
            if any('layer4' in key for key in keys):
                # 通过检查某些层的通道数来判断ResNet类型
                for key in keys:
                    if 'encoder.layer4' in key and 'conv1.weight' in key:
                        try:
                            channels = state_dict[key].shape[0]
                            if channels == 512:
                                encoder_name = 'resnet50'
                                break
                            elif channels == 256:
                                encoder_name = 'resnet34'
                                break
                            elif channels == 128:
                                encoder_name = 'resnet18'
                                break
                        except:
                            continue
                if encoder_name == 'resnet34':  # 如果没找到layer4，可能是resnet18
                    if not any('encoder.layer4' in key for key in keys):
                        encoder_name = 'resnet18'
            else:
                encoder_name = 'resnet18'
        elif any('efficientnet' in key for key in keys):
            # 检测EfficientNet类型
            if any('_blocks.0' in key for key in keys):
                encoder_name = 'efficientnet-b0'
            else:
                encoder_name = 'efficientnet-b0'
        elif any('mobilenet' in key for key in keys):
            encoder_name = 'mobilenet_v2'
        
        # 根据匹配度判断模型架构
        if unetpp_matches >= 3:
            return 'unetplusplus', encoder_name, 0.9
        elif deeplabv3_matches >= 2:
            if any('decoder.aspp' in key for key in keys):
                return 'deeplabv3plus', encoder_name, 0.85
            else:
                return 'deeplabv3', encoder_name, 0.85
        elif fpn_matches >= 3:
            return 'fpn', encoder_name, 0.8
        elif pspnet_matches >= 2:
            return 'pspnet', encoder_name, 0.8
        elif linknet_matches >= 2:
            return 'linknet', encoder_name, 0.8
        else:
            # 默认认为是标准UNet
            return 'unet', encoder_name, 0.7
            
    except Exception as e:
        print(f"⚠️ 检测模型架构时出错: {e}")
        return 'unet', 'resnet34', 0.5

def create_model_from_checkpoint(checkpoint_path, n_channels=3, n_classes=5, 
                                model_name=None, encoder_name=None, encoder_weights='imagenet'):
    """
    从checkpoint自动检测并创建对应的模型
    
    Args:
        checkpoint_path (str): 模型权重文件路径
        n_channels (int): 输入通道数
        n_classes (int): 输出类别数
        model_name (str, optional): 手动指定模型架构，None则自动检测
        encoder_name (str, optional): 手动指定编码器，None则自动检测
        encoder_weights (str): 预训练权重
        
    Returns:
        torch.nn.Module: 创建的模型
    """
    
    if model_name is None or encoder_name is None:
        print(f"🔍 正在检测模型架构: {checkpoint_path}")
        detected_model, detected_encoder, confidence = detect_model_architecture(checkpoint_path)
        
        if model_name is None:
            model_name = detected_model
        if encoder_name is None:
            encoder_name = detected_encoder
            
        print(f"✓ 检测结果: {model_name} + {encoder_name} (置信度: {confidence:.2f})")
    
    # 创建模型
    model = create_model(
        n_channels=n_channels,
        n_classes=n_classes,
        model_name=model_name,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )
    
    return model

def create_model(n_channels=3, n_classes=5, model_name='unet', encoder_name='resnet34', encoder_weights='imagenet'):
    """
    创建分割模型，优先使用segmentation_models_pytorch
    
    Args:
        n_channels: 输入通道数
        n_classes: 输出类别数
        model_name: 模型架构 ('unet', 'unetplusplus', 'deeplabv3', 'fpn', 'pspnet', 'linknet')
        encoder_name: 编码器名称 ('resnet34', 'resnet50', 'efficientnet-b0', 'timm-efficientnet-b0', etc.)
        encoder_weights: 预训练权重 ('imagenet', None)
    
    Returns:
        PyTorch模型
    """
    
    if SMP_AVAILABLE:
        try:
            print(f"使用 segmentation_models_pytorch 创建 {model_name} 模型")
            print(f"编码器: {encoder_name}, 预训练权重: {encoder_weights}")
            
            # 根据模型名称创建不同的分割模型
            if model_name.lower() == 'unet':
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None  # 在训练时使用CrossEntropyLoss，不需要激活函数
                )
            elif model_name.lower() == 'unetplusplus':
                model = smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            elif model_name.lower() == 'deeplabv3':
                model = smp.DeepLabV3(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            elif model_name.lower() == 'deeplabv3plus':
                model = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            elif model_name.lower() == 'fpn':
                model = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            elif model_name.lower() == 'pspnet':
                model = smp.PSPNet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            elif model_name.lower() == 'linknet':
                model = smp.Linknet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            else:
                print(f"未知模型名称 {model_name}，使用默认UNet")
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=n_channels,
                    classes=n_classes,
                    activation=None
                )
            
            print(f"✓ 成功创建 {model_name} 模型")
            return model
            
        except Exception as e:
            print(f"⚠️ 使用segmentation_models_pytorch创建模型失败: {e}")
            print("回退到自定义UNet模型")
            return UNet(n_channels=n_channels, n_classes=n_classes)
    
    else:
        print("使用自定义UNet模型")
        return UNet(n_channels=n_channels, n_classes=n_classes)

def get_model_info():
    """获取可用的模型和编码器信息"""
    if not SMP_AVAILABLE:
        return {
            'models': ['unet'],
            'encoders': ['custom'],
            'weights': ['random']
        }
    
    # 常用的模型架构
    available_models = [
        'unet',           # 经典UNet
        'unetplusplus',   # UNet++，更好的跳跃连接
        'deeplabv3',      # DeepLabV3，使用空洞卷积
        'deeplabv3plus',  # DeepLabV3+，改进版
        'fpn',            # Feature Pyramid Network
        'pspnet',         # Pyramid Scene Parsing Network
        'linknet',        # LinkNet，轻量级
    ]
    
    # 常用的编码器（预训练骨干网络）
    recommended_encoders = [
        # ResNet系列（平衡性能和速度）
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        
        # EfficientNet系列（高效）
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
        'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
        
        # RegNet系列（新一代高效网络）
        'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_008',
        
        # MobileNet系列（轻量级）
        'mobilenet_v2',
        
        # Vision Transformer（最新技术）
        'mit_b0', 'mit_b1', 'mit_b2',
    ]
    
    return {
        'models': available_models,
        'encoders': recommended_encoders,
        'weights': ['imagenet', None]
    }

# 为了向后兼容，保留UNet类的别名
def UNet(n_channels, n_classes, bilinear=True):
    """向后兼容的UNet创建函数"""
    return create_model(n_channels=n_channels, n_classes=n_classes, model_name='unet')