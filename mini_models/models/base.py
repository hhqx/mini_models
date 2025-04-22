"""
Base model class implementation
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

from mini_models.weights.manager import weight_manager
from mini_models.weights.registry import get_model_info

class BaseModel(nn.Module):
    """模型基类"""
    
    def __init__(self, model_name: str, pretrained: bool = True, 
                 weight_version: str = "latest", prefer_user_weights: bool = True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.config = kwargs
        self._is_trained = False
        self._weight_version = weight_version
        self._prefer_user_weights = prefer_user_weights
        
        # 初始化模型架构
        self._build_model(**kwargs)
        
        # 加载预训练权重（如果需要）
        if pretrained:
            self.load_pretrained_weights()
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """构建模型架构，需要子类实现"""
        raise NotImplementedError("子类必须实现_build_model方法")
    
    @abstractmethod
    def forward(self, x):
        """前向传播"""
        raise NotImplementedError("子类必须实现forward方法")
    
    def load_pretrained_weights(self) -> bool:
        """
        加载预训练权重
        
        Returns:
            bool: 是否成功加载权重
        """
        # 首先检查是否有可用权重
        weight_path = weight_manager.get_weight_path(
            self.model_name, 
            version=self._weight_version,
            prefer_user=self._prefer_user_weights
        )
        
        # 如果没有找到权重，尝试下载
        if not weight_path:
            print(f"未找到本地权重，尝试下载 {self.model_name}...")
            weight_path = weight_manager.download_weight(
                self.model_name, 
                version=self._weight_version
            )
        
        # 如果仍然没有权重，返回失败
        if not weight_path:
            print(f"警告：无法加载 {self.model_name} 的预训练权重")
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
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"警告：模型加载时缺少键：{missing_keys}")
            if unexpected_keys:
                print(f"警告：模型加载时发现意外键：{unexpected_keys}")
                
            self._is_trained = True
            print(f"成功加载预训练权重：{weight_path}")
            return True
        except Exception as e:
            print(f"加载权重时出错: {e}")
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
