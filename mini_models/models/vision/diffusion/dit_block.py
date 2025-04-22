"""
DiT (Diffusion Transformer) 块实现
"""
import torch
import torch.nn as nn
import math
from typing import Optional

class DiTBlock(nn.Module):
    """
    DiT基本构建块，包含自注意力和前馈网络，以及条件缩放和偏移
    """
    def __init__(self, emb_size: int, nhead: int):
        """
        初始化DiT块
        
        Args:
            emb_size: 嵌入维度
            nhead: 注意力头数量
        """
        super().__init__()
        
        self.emb_size = emb_size
        self.nhead = nhead
        
        # 条件层
        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)        
        self.alpha1 = nn.Linear(emb_size, emb_size)
        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        
        # 多头自注意力
        self.wq = nn.Linear(emb_size, nhead * emb_size)  # 查询权重
        self.wk = nn.Linear(emb_size, nhead * emb_size)  # 键权重
        self.wv = nn.Linear(emb_size, nhead * emb_size)  # 值权重
        self.lv = nn.Linear(nhead * emb_size, emb_size)  # 线性输出层
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, emb_size]
            cond: 条件嵌入，形状为 [batch_size, emb_size]
            
        Returns:
            形状为 [batch_size, seq_len, emb_size] 的输出张量
        """
        # 计算条件缩放和偏移
        gamma1_val = self.gamma1(cond)
        beta1_val = self.beta1(cond)
        alpha1_val = self.alpha1(cond)
        gamma2_val = self.gamma2(cond)
        beta2_val = self.beta2(cond)
        alpha2_val = self.alpha2(cond)
        
        # 第一层归一化
        y = self.ln1(x)  # [batch, seq_len, emb_size]
        
        # 条件缩放和偏移
        y = y * (1 + gamma1_val.unsqueeze(1)) + beta1_val.unsqueeze(1) 

        # 多头自注意力
        q = self.wq(y)    # [batch, seq_len, nhead*emb_size]
        k = self.wk(y)    # [batch, seq_len, nhead*emb_size]    
        v = self.wv(y)    # [batch, seq_len, nhead*emb_size]
        
        # 重塑张量以进行多头注意力
        q = q.view(q.size(0), q.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # [batch, nhead, seq_len, emb_size]
        k = k.view(k.size(0), k.size(1), self.nhead, self.emb_size).permute(0, 2, 3, 1)  # [batch, nhead, emb_size, seq_len]
        v = v.view(v.size(0), v.size(1), self.nhead, self.emb_size).permute(0, 2, 1, 3)  # [batch, nhead, seq_len, emb_size]
        
        # 计算注意力分数
        attn = q @ k / math.sqrt(q.size(2))   # [batch, nhead, seq_len, seq_len]
        attn = torch.softmax(attn, dim=-1)    # [batch, nhead, seq_len, seq_len]
        
        # 应用注意力权重
        y = attn @ v    # [batch, nhead, seq_len, emb_size]
        
        # 重塑回原始形状
        y = y.permute(0, 2, 1, 3)  # [batch, seq_len, nhead, emb_size]
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))  # [batch, seq_len, nhead*emb_size]
        y = self.lv(y)  # [batch, seq_len, emb_size]
        
        # 条件缩放
        y = y * alpha1_val.unsqueeze(1)
        
        # 残差连接
        y = x + y  
        
        # 第二层归一化
        z = self.ln2(y)
        
        # 条件缩放和偏移
        z = z * (1 + gamma2_val.unsqueeze(1)) + beta2_val.unsqueeze(1)
        
        # 前馈网络
        z = self.ff(z)
        
        # 条件缩放
        z = z * alpha2_val.unsqueeze(1)
        
        # 残差连接
        return y + z
