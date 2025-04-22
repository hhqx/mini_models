"""
模型模块初始化和模型工厂
"""
from typing import Dict, Any, Optional, Union, Type

# 导入模型基类
from mini_models.models.base import BaseModel
from mini_models.weights.manager import weight_manager

# 模型注册表 - 先定义空表，稍后通过注册函数填充
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(model_name: str, model_class: Type[BaseModel]):
    """注册模型"""
    MODEL_REGISTRY[model_name] = model_class
    
def get_model(
    model_name: str, 
    pretrained: bool = True,
    weight_version: str = "latest",
    prefer_user_weights: bool = True,
    ignore_download_error: bool = True,
    **kwargs
) -> BaseModel:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称
        pretrained: 是否加载预训练权重
        weight_version: 指定权重版本，默认为"latest"
        prefer_user_weights: 是否优先使用用户自定义权重
        ignore_download_error: 是否忽略权重下载错误
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        BaseModel: 模型实例
    """
    # 确保所有模型已导入并注册
    _ensure_models_imported()
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，可用模型: {list(MODEL_REGISTRY.keys())}")
    
    # 创建模型实例
    model_class = MODEL_REGISTRY[model_name]
    
    try:
        model = model_class(
            model_name=model_name, 
            pretrained=pretrained,
            weight_version=weight_version,
            prefer_user_weights=prefer_user_weights,
            **kwargs
        )
    except RuntimeError as e:
        if not ignore_download_error or "下载模型" not in str(e):
            raise
        print(f"警告: 预训练权重下载失败，使用随机初始化。原因: {e}")
        # 重试，但不加载预训练权重
        model = model_class(model_name=model_name, pretrained=False, **kwargs)
    
    return model

def list_available_models():
    """列出所有可用模型"""
    _ensure_models_imported()
    return list(MODEL_REGISTRY.keys())

def _ensure_models_imported():
    """确保所有模型已导入并注册"""
    # 导入所需的模型
    # 如果MODULE_REGISTRY为空，注册我们的模型
    # if not MODEL_REGISTRY:
    #     try:
    #         # 从vision模块导入
    #         from mini_models.models.vision import dit
    #         # 手动注册模型
    #         from mini_models.models.vision.dit import DiTMNIST
    #         register_model("dit_mnist", DiTMNIST)
    #         # print(f"已注册模型: {list(MODEL_REGISTRY.keys())}")
    #     except ImportError as e:
    #         print(f"导入模型时出错: {e}")

def import_user_weight(model_name: str, weight_path: str) -> bool:
    """
    导入用户自定义权重
    
    Args:
        model_name: 模型名称
        weight_path: 权重文件路径
        
    Returns:
        bool: 导入是否成功
    """
    return weight_manager.import_user_weight(model_name, weight_path)

def check_model_updates() -> Dict[str, bool]:
    """
    检查所有模型是否有更新
    
    Returns:
        Dict[str, bool]: 模型名称到是否有更新的映射
    """
    return weight_manager.check_for_updates()

def weight_info(model_name: str) -> Dict[str, Any]:
    """
    获取模型权重的详细信息
    
    Args:
        model_name: 模型名称
        
    Returns:
        Dict[str, Any]: 权重信息
    """
    return weight_manager.get_weight_info(model_name)
