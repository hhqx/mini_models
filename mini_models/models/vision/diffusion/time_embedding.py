"""
扩散模型的时间嵌入模块
"""
import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块，使用正弦余弦位置编码
    """
    def __init__(self, embedding_dim: int):
        """
        初始化时间嵌入层
        
        Args:
            embedding_dim: 嵌入维度
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 使用半个维度，因为我们为每个维度使用sin和cos
        assert embedding_dim % 2 == 0, "嵌入维度必须是偶数"
        half_dim = embedding_dim // 2
        
        # 定义常量序列用于位置编码
        self.emb_scale = 10000
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        时间步编码
        
        Args:
            timesteps: 形状为 [batch_size] 的时间步张量
            
        Returns:
            形状为 [batch_size, embedding_dim] 的嵌入向量
        """
        # 确保输入是正确形状
        if len(timesteps.shape) == 0:
            timesteps = timesteps.unsqueeze(0)  # [1]
        
        # 创建半个维度的位置索引
        half_dim = self.embedding_dim // 2
        
        # 计算不同频率的 log 空间
        emb = math.log(self.emb_scale) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        
        # 与时间步相乘以获取不同频率
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        
        # 应用正弦和余弦
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # 如果维度不匹配，使用零填充
        if self.embedding_dim > emb.shape[1]:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :self.embedding_dim - emb.shape[1]])], dim=1)
            
        return emb  # [batch_size, embedding_dim]
