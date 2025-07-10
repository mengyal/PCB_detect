# model_configs.py
"""
模型配置文件 - 提供不同的预定义模型配置供选择
"""

# 基础配置（快速测试）
BASIC_CONFIG = {
    'model_name': 'unet',
    'encoder_name': 'resnet18',
    'encoder_weights': 'imagenet',
    'description': '基础UNet + ResNet18编码器，训练速度快，适合快速测试'
}

# 推荐配置（平衡性能和速度）
RECOMMENDED_CONFIG = {
    'model_name': 'unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'description': '推荐配置：UNet + ResNet34编码器，性能和速度的良好平衡'
}

# 高性能配置（追求最佳效果）
HIGH_PERFORMANCE_CONFIG = {
    'model_name': 'unetplusplus',
    'encoder_name': 'resnet50',
    'encoder_weights': 'imagenet',
    'description': '高性能配置：UNet++ + ResNet50编码器，更好的性能但训练较慢'
}

# 高效配置（EfficientNet系列）
EFFICIENT_CONFIG = {
    'model_name': 'unet',
    'encoder_name': 'efficientnet-b2',
    'encoder_weights': 'imagenet',
    'description': '高效配置：UNet + EfficientNet-B2编码器，参数少但性能优秀'
}

# 轻量级配置（资源受限环境）
LIGHTWEIGHT_CONFIG = {
    'model_name': 'linknet',
    'encoder_name': 'mobilenet_v2',
    'encoder_weights': 'imagenet',
    'description': '轻量级配置：LinkNet + MobileNetV2编码器，适合资源受限环境'
}

# 现代配置（最新技术）
MODERN_CONFIG = {
    'model_name': 'deeplabv3plus',
    'encoder_name': 'timm-efficientnet-b1',
    'encoder_weights': 'imagenet',
    'description': '现代配置：DeepLabV3+ + EfficientNet-B1编码器，使用最新技术'
}

# 精细分割配置（适合PCB检测）
PCB_OPTIMIZED_CONFIG = {
    'model_name': 'unetplusplus',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'description': 'PCB优化配置：UNet++ + ResNet34编码器，针对PCB焊点检测优化'
}

# 所有预定义配置
PREDEFINED_CONFIGS = {
    'basic': BASIC_CONFIG,
    'recommended': RECOMMENDED_CONFIG,
    'high_performance': HIGH_PERFORMANCE_CONFIG,
    'efficient': EFFICIENT_CONFIG,
    'lightweight': LIGHTWEIGHT_CONFIG,
    'modern': MODERN_CONFIG,
    'pcb_optimized': PCB_OPTIMIZED_CONFIG,
}

def get_config(config_name='recommended'):
    """
    获取指定的模型配置
    
    Args:
        config_name (str): 配置名称
    
    Returns:
        dict: 模型配置字典
    """
    if config_name not in PREDEFINED_CONFIGS:
        print(f"警告: 未知配置 '{config_name}'，使用默认的 'recommended' 配置")
        config_name = 'recommended'
    
    config = PREDEFINED_CONFIGS[config_name].copy()
    print(f"使用配置: {config_name}")
    print(f"描述: {config['description']}")
    
    return config

def list_all_configs():
    """列出所有可用的配置"""
    print("📋 可用的模型配置:")
    print("=" * 60)
    
    for name, config in PREDEFINED_CONFIGS.items():
        print(f"\n🏷️  {name}:")
        print(f"   模型: {config['model_name']}")
        print(f"   编码器: {config['encoder_name']}")
        print(f"   权重: {config['encoder_weights']}")
        print(f"   描述: {config['description']}")
    
    print("\n" + "=" * 60)
    print("使用方法: 在train.py中设置 MODEL_CONFIG = 'config_name'")

def get_model_recommendations():
    """根据不同需求推荐模型配置"""
    recommendations = {
        "🚀 快速测试": "basic",
        "⭐ 一般使用": "recommended", 
        "🎯 PCB检测": "pcb_optimized",
        "💪 追求性能": "high_performance",
        "⚡ 高效训练": "efficient",
        "📱 资源受限": "lightweight",
        "🔬 最新技术": "modern"
    }
    
    print("🎯 根据需求选择配置:")
    print("=" * 50)
    for need, config in recommendations.items():
        print(f"{need}: '{config}'")
    
    return recommendations

if __name__ == "__main__":
    # 显示所有配置
    list_all_configs()
    print()
    get_model_recommendations()
