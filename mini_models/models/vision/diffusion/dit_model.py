"""
DiT (Diffusion Transformer) 模型实现
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from mini_models.models.base import BaseModel
from mini_models.models.vision.diffusion.time_embedding import TimeEmbedding
from mini_models.models.vision.diffusion.dit_block import DiTBlock
from mini_models.models import register_model

class DiTModel(BaseModel):
    """
    DiT (Diffusion Transformer) 模型，用于条件图像生成
    """
    def __init__(
        self,
        model_name: str = "dit_mnist",
        pretrained: bool = True,
        **kwargs
    ):
        """
        初始化DiT模型
        
        Args:
            model_name: 模型名称
            pretrained: 是否加载预训练权重
        """
        self.img_size = kwargs.get('img_size', 28)
        self.patch_size = kwargs.get('patch_size', 4)
        self.in_channels = kwargs.get('channel', 1)
        self.emb_size = kwargs.get('emb_size', 64)
        self.label_num = kwargs.get('label_num', 10)
        self.dit_num = kwargs.get('dit_num', 3)
        self.head = kwargs.get('head', 4)
        
        super().__init__(model_name=model_name, pretrained=pretrained, **kwargs)
    
    def _build_model(self, **kwargs):
        """构建模型架构"""
        # 计算补丁数量
        self.patch_count = self.img_size // self.patch_size
        
        # patchify 图像
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels * self.patch_size**2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        
        # 补丁嵌入
        self.patch_emb = nn.Linear(
            in_features=self.in_channels * self.patch_size**2,
            out_features=self.emb_size
        )
        
        # 补丁位置嵌入
        self.patch_pos_emb = nn.Parameter(
            torch.rand(1, self.patch_count**2, self.emb_size)
        )
        
        # 时间嵌入
        self.time_emb = nn.Sequential(
            TimeEmbedding(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size)
        )
        
        # 标签嵌入
        self.label_emb = nn.Embedding(
            num_embeddings=self.label_num,
            embedding_dim=self.emb_size
        )
        
        # DiT块
        self.dits = nn.ModuleList()
        for _ in range(self.dit_num):
            self.dits.append(DiTBlock(self.emb_size, self.head))
        
        # 层归一化
        self.ln = nn.LayerNorm(self.emb_size)
        
        # 线性投影回补丁
        self.linear = nn.Linear(
            self.emb_size,
            self.in_channels * self.patch_size**2
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入噪声图像，形状为 [batch_size, channel, height, width]
            t: 时间步，形状为 [batch_size]
            y: 标签，形状为 [batch_size]
            
        Returns:
            预测的噪声，形状与输入x相同
        """
        # 标签嵌入
        y_emb = self.label_emb(y)  # [batch, emb_size]
        
        # 时间步嵌入
        t_emb = self.time_emb(t)   # [batch, emb_size]
        
        # 合并条件嵌入
        cond = y_emb + t_emb       # [batch, emb_size]
        
        # 图像补丁化
        x = self.conv(x)  # [batch, patch_size^2*channel, patch_count, patch_count]
        x = x.permute(0, 2, 3, 1)  # [batch, patch_count, patch_count, patch_size^2*channel]
        x = x.reshape(x.size(0), self.patch_count*self.patch_count, -1)  # [batch, patch_count^2, patch_size^2*channel]
        
        # 补丁嵌入
        x = self.patch_emb(x)  # [batch, patch_count^2, emb_size]
        
        # 添加位置嵌入
        x = x + self.patch_pos_emb  # [batch, patch_count^2, emb_size]
        
        # 通过DiT块
        for dit in self.dits:
            x = dit(x, cond)  # [batch, patch_count^2, emb_size]
        
        # 层归一化
        x = self.ln(x)  # [batch, patch_count^2, emb_size]
        
        # 线性投影回原始补丁维度
        x = self.linear(x)  # [batch, patch_count^2, patch_size^2*channel]
        
        # 重塑回图像格式
        x = x.reshape(
            x.size(0), 
            self.patch_count, 
            self.patch_count, 
            self.in_channels, 
            self.patch_size, 
            self.patch_size
        )  # [batch, patch_count(H), patch_count(W), channel, patch_size(H), patch_size(W)]
        
        # 重排轴以获得正确的输出形状
        x = x.permute(0, 3, 1, 4, 2, 5)  # [batch, channel, patch_count(H), patch_size(H), patch_count(W), patch_size(W)]
        x = x.reshape(
            x.size(0), 
            self.in_channels, 
            self.img_size, 
            self.img_size
        )  # [batch, channel, img_size, img_size]
        
        return x
    
    def generate(
        self, 
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_steps: int = 1000,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        生成图像
        
        Args:
            batch_size: 生成图像的批次大小
            labels: 条件标签，如果为None，则随机生成
            device: 设备
            num_steps: 去噪步骤数量
            show_progress: 是否显示进度
            
        Returns:
            生成的最终图像和每一步的图像列表
        """
        from mini_models.models.vision.diffusion.diffusion_process import DiffusionProcess
        
        # 创建扩散过程
        diffusion = DiffusionProcess(num_steps=num_steps)
        
        # 如果未提供设备，则使用模型的设备
        if device is None:
            device = next(self.parameters()).device
            
        # 如果未提供标签，则随机生成
        if labels is None:
            labels = torch.randint(0, self.label_num, (batch_size,), device=device)
        else:
            labels = labels.to(device)
            
        # 生成初始噪声
        x = torch.randn(
            (batch_size, self.in_channels, self.img_size, self.img_size),
            device=device
        )
        
        # 执行去噪过程
        return diffusion.backward_denoise(self, x, labels, show_progress)

# 注册模型
register_model("dit_mnist", DiTModel)
