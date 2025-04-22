"""
模型基类，定义所有模型的通用接口
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

from mini_models.weights.downloader import download_if_needed
from mini_models.weights.registry import get_model_info

class BaseModel(nn.Module, ABC):
    """
    所有mini_models模型的基类
    """
    def __init__(self, model_name: str, pretrained: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.config = kwargs
        self._is_trained = False
        
        # 初始化模型架构
        self._build_model(**kwargs)
        
        # 加载预训练权重（如果需要）
        if pretrained:
            self.load_pretrained_weights()
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """构建模型架构，需要子类实现"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向传播，需要子类实现"""
        pass
    
    def load_pretrained_weights(self):
        """加载预训练权重"""
        model_info = get_model_info(self.model_name)
        if not model_info:
            print(f"Warning: No registered pretrained weights found for {self.model_name}")
            return False
        
        # 下载权重（如果本地没有）
        weight_path = download_if_needed(self.model_name)
        
        if not os.path.exists(weight_path):
            print(f"Warning: Failed to download weights for {self.model_name}")
            return False
            
        # 加载权重
        try:
            state_dict = torch.load(weight_path, map_location='cpu')
            # 处理state_dict格式不一致的情况
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # 尝试加载权重
            self.load_state_dict(state_dict, strict=False)
            self._is_trained = True
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def save_weights(self, path: str):
        """保存模型权重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.state_dict(),
            'config': self.config,
        }, path)
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.copy()
    
    @property
    def is_pretrained(self) -> bool:
        """是否已加载预训练权重"""
        return self._is_trained
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
