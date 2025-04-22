"""
MNIST数字识别的轻量级CNN模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from mini_models.models.base import BaseModel
from mini_models.models.register import register_model

class MNISTModel(BaseModel):
    """
    用于MNIST数字识别的简单CNN模型
    """
    def __init__(self, model_name: str = "mnist_cnn", pretrained: bool = True, **kwargs):
        """
        初始化MNIST模型
        
        Args:
            model_name: 模型名称
            pretrained: 是否加载预训练权重
        """
        self.num_classes = kwargs.get('num_classes', 10)
        super().__init__(model_name=model_name, pretrained=pretrained, **kwargs)
    
    def _build_model(self, **kwargs):
        """构建模型架构"""
        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, 1, 28, 28]
            
        Returns:
            输出张量，形状为 [batch_size, num_classes]
        """
        # 第一层卷积 + 批归一化 + ReLU + 最大池化
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        # 第二层卷积 + 批归一化 + ReLU + 最大池化
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.25)
        x = self.fc2(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测函数，返回类别和概率
        
        Args:
            x: 输入张量
            
        Returns:
            预测的类别和概率
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities

# 注册模型
register_model("mnist_cnn", MNISTModel)
