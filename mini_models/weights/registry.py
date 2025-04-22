"""
模型权重注册表管理
"""
import os
import json
import yaml
from typing import Dict, Any, Optional, List

from mini_models.config import config

# 模型注册表
MODEL_REGISTRY = {}

def init_weight_registry():
    """初始化权重注册表"""
    global MODEL_REGISTRY
    
    # 注册表路径
    registry_path = os.path.join(config["weights_dir"], "registry.json")
    registry_yaml_path = os.path.join(config["weights_dir"], "registry.yaml")
    
    # 尝试从JSON加载
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                MODEL_REGISTRY = json.load(f)
            return
        except Exception as e:
            print(f"Warning: Failed to load model registry from JSON: {e}")
    
    # 尝试从YAML加载
    if os.path.exists(registry_yaml_path):
        try:
            with open(registry_yaml_path, 'r') as f:
                MODEL_REGISTRY = yaml.safe_load(f)
            return
        except Exception as e:
            print(f"Warning: Failed to load model registry from YAML: {e}")
    
    # 尝试从其他位置加载
    MODEL_REGISTRY = {}
    yaml_paths = [
        os.path.join(os.path.dirname(__file__), "../../release/model_info.yaml"),
        os.path.join(os.path.dirname(__file__), "../../release/model_info.yml")
    ]
    json_path = os.path.join(os.path.dirname(__file__), "../../release/model_info.json")
    
    # 尝试从YAML加载
    for path in yaml_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    MODEL_REGISTRY = yaml.safe_load(f)
                break
            except Exception as e:
                print(f"Warning: Failed to load model info from {path}: {e}")
    
    # 如果YAML加载失败，尝试从JSON加载
    if not MODEL_REGISTRY and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                MODEL_REGISTRY = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load model info from {json_path}: {e}")
    
    # 如果仍然为空，使用默认值
    if not MODEL_REGISTRY:
        # 初始化一些默认模型（回退方案）
        MODEL_REGISTRY = {
            # 视觉模型
            "resnet18_mini": {
                "url": "v0.1/vision/resnet18_mini.pth",
                "size": 44000000,
                "sha256": "abcdef...",
                "task": "image_classification",
                "description": "轻量级ResNet18，在ImageNet上训练"
            },
            "mobilenet_v2_mini": {
                "url": "v0.1/vision/mobilenet_v2_mini.pth",
                "size": 13000000,
                "sha256": "123456...",
                "task": "image_classification",
                "description": "轻量级MobileNetV2，适合移动设备"
            },
            
            # NLP模型
            "bert_mini": {
                "url": "v0.1/nlp/bert_mini.pth",
                "size": 55000000,
                "sha256": "789abc...",
                "task": "text_classification",
                "description": "轻量级BERT模型，6层transformer"
            },
            
            # 扩散模型
            "dit_mnist": {
                "url": "v0.1/diffusion/dit_mnist.pth",
                "size": 2500000,
                "sha256": "def789...",
                "task": "image_generation",
                "description": "MNIST图像生成的DiT模型"
            }
        }
    
    # 保存注册表
    save_registry()

def register_model(
    model_name: str, 
    url: str, 
    size: int, 
    sha256: str, 
    task: str, 
    description: str
):
    """注册新模型"""
    global MODEL_REGISTRY
    
    MODEL_REGISTRY[model_name] = {
        "url": url,
        "size": size,
        "sha256": sha256,
        "task": task,
        "description": description
    }
    
    # 保存更新的注册表
    save_registry()

def save_registry():
    """保存模型注册表到本地"""
    registry_path = os.path.join(config["weights_dir"], "registry.json")
    registry_yaml_path = os.path.join(config["weights_dir"], "registry.yaml")
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    # 保存JSON格式
    try:
        with open(registry_path, 'w') as f:
            json.dump(MODEL_REGISTRY, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save model registry to JSON: {e}")
    
    # 保存YAML格式
    try:
        with open(registry_yaml_path, 'w') as f:
            yaml.dump(MODEL_REGISTRY, f, default_flow_style=False)
    except Exception as e:
        print(f"Warning: Failed to save model registry to YAML: {e}")

def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """获取模型信息"""
    return MODEL_REGISTRY.get(model_name)

def list_models(task: Optional[str] = None) -> List[str]:
    """列出所有注册的模型，可按任务过滤"""
    if task is None:
        return list(MODEL_REGISTRY.keys())
    
    return [
        model_name for model_name, info in MODEL_REGISTRY.items()
        if info.get("task") == task
    ]
