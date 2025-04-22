"""
模型模块初始化和模型工厂
"""
from typing import Dict, Any, Optional, Union, Type

# 导入模型基类
from mini_models.models.base import BaseModel

# 这里将导入所有具体模型实现
# from mini_models.models.vision.resnet import ResNet18Mini, ResNet34Mini
# from mini_models.models.vision.mobilenet import MobileNetV2Mini
# from mini_models.models.nlp.bert import BertMini
# 等等...

# 模型注册表
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    # 将在实际导入后添加
    # 'resnet18_mini': ResNet18Mini,
    # 'mobilenet_v2_mini': MobileNetV2Mini,
    # 'bert_mini': BertMini,
}

def register_model(model_name: str, model_class: Type[BaseModel]):
    """注册模型"""
    MODEL_REGISTRY[model_name] = model_class
    
def get_model(
    model_name: str, 
    pretrained: bool = True, 
    **kwargs
) -> BaseModel:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称
        pretrained: 是否加载预训练权重
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        BaseModel: 模型实例
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，可用模型: {list(MODEL_REGISTRY.keys())}")
    
    # 创建模型实例
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(model_name=model_name, pretrained=pretrained, **kwargs)
    
    return model

def list_available_models():
    """列出所有可用模型"""
    return list(MODEL_REGISTRY.keys())

# 注：此时注册表为空，我们将在各模型模块中注册具体模型
