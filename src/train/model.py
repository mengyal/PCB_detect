# model.py

import segmentation_models_pytorch as smp

def create_model(arch='unetplusplus', encoder='resnet34', in_channels=3, num_classes=4):
    """
    使用segmentation_models_pytorch库创建模型。

    Args:
        arch (str): 模型架构, 如 'unet', 'unetplusplus', 'linknet'.
        encoder (str): 编码器骨干网络, 如 'resnet34', 'efficientnet-b0'.
        in_channels (int): 输入通道数.
        num_classes (int): 输出类别数.
    
    Returns:
        torch.nn.Module: 一个PyTorch模型.
    """
    print(f"正在创建模型: {arch}，编码器: {encoder}")
    
    # 从众多架构中选择一个
    model_class = getattr(smp, arch.capitalize())
    
    model = model_class(
        encoder_name=encoder,
        encoder_weights="imagenet",  # 使用在ImageNet上预训练的权重进行迁移学习
        in_channels=in_channels,
        classes=num_classes,
    )
    return model