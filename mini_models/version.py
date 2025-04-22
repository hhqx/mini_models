"""
版本信息模块
"""

__version__ = "0.1.1"

def get_version():
    """返回当前版本"""
    return __version__

def get_model_version(model_name: str) -> str:
    """
    获取特定模型的版本
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型版本号，若不存在则返回"未知"
    """
    try:
        import os
        import yaml
        
        # 模型注册表路径
        registry_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "release", 
            "model_registry.yaml"
        )
        
        if not os.path.exists(registry_file):
            return "unknown"
            
        with open(registry_file, 'r') as f:
            registry = yaml.safe_load(f) or {}
            
        if model_name in registry:
            return registry[model_name].get('version', 'unknown')
        return "unknown"
    except Exception:
        return "unknown"
