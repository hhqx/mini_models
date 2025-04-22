"""
轻量级ResNet模型实现
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from mini_models.models.base import BaseModel
from mini_models.models.register import register_model

class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出维度不匹配，使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNet18Mini(BaseModel):
    """
    轻量级ResNet18模型实现
    """
    def __init__(
        self,
        model_name: str = "resnet18_mini",
        pretrained: bool = True,
        **kwargs
    ):
        """
        初始化ResNet18Mini
        
        Args:
            model_name: 模型名称
            pretrained: 是否加载预训练权重
        """
        self.num_classes = kwargs.get('num_classes', 1000)
        self.in_channels = 64
        super().__init__(model_name=model_name, pretrained=pretrained, **kwargs)
    
    def _build_model(self, **kwargs):
        """构建模型架构"""
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """创建残差层"""
        layers = []
        # 第一个残差块可能需要调整维度
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        # 更新输入通道数
        self.in_channels = out_channels
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 注册模型
register_model("resnet18_mini", ResNet18Mini)
