"""
扩散过程实现
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

class DiffusionProcess:
    """
    扩散过程，管理噪声添加和去除
    """
    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        """
        初始化扩散过程
        
        Args:
            num_steps: 扩散步骤数量
            beta_start: 初始β值
            beta_end: 最终β值
        """
        self.num_steps = num_steps
        
        # 创建噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 计算扩散过程中使用的常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 计算后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从x_0采样获取x_t，即正向扩散过程
        
        Args:
            x_0: 初始干净图像
            t: 时间步
            noise: 可选的预定义噪声
            
        Returns:
            t时刻的噪声图像
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_0.device)
        
        # 为了进行广播，调整形状
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        使用模型预测x_0
        
        Args:
            model: DiT模型
            x_t: t时刻的噪声图像
            t: 时间步
            y: 条件标签
            
        Returns:
            预测的x_0
        """
        # 预测噪声
        pred_noise = model(x_t, t, y)
        
        # 使用预测的噪声重构x_0
        sqrt_recip_alphas = self.sqrt_recip_alphas_cumprod[t].to(x_t.device)
        sqrt_recipm1_alphas = self.sqrt_recipm1_alphas_cumprod[t].to(x_t.device)
        
        # 为了进行广播，调整形状
        sqrt_recip_alphas = sqrt_recip_alphas.view(-1, 1, 1, 1)
        sqrt_recipm1_alphas = sqrt_recipm1_alphas.view(-1, 1, 1, 1)
        
        pred_x0 = sqrt_recip_alphas * x_t - sqrt_recipm1_alphas * pred_noise
        return pred_x0
    
    def p_mean_variance(
        self, 
        model: torch.nn.Module, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算p(x_{t-1} | x_t)的均值和方差
        
        Args:
            model: DiT模型
            x_t: t时刻的噪声图像
            t: 时间步
            y: 条件标签
            
        Returns:
            均值和方差的元组
        """
        # 获取设备
        device = x_t.device
        
        # 预测噪声
        pred_noise = model(x_t, t, y)
        
        # 计算均值
        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        betas = self.betas.to(device)
        
        # 为了广播，需要调整形状
        shape = (x_t.size(0), 1, 1, 1)
        
        coef1 = betas[t].view(shape) / sqrt_one_minus_alphas_cumprod[t].view(shape)
        coef2 = sqrt_one_minus_alphas_cumprod[t].view(shape) / (1.0 - alphas_cumprod[t]).view(shape)
        
        pred_mean = (1.0 / torch.sqrt(alphas[t])).view(shape) * (
            x_t - coef1 * pred_noise
        )
        
        # 计算方差
        if t[0] == 0:
            variance = torch.zeros_like(pred_mean)
        else:
            variance = self.posterior_variance[t].to(device).view(shape)
            
        return pred_mean, variance
    
    def backward_denoise(
        self, 
        model: torch.nn.Module, 
        x_t: torch.Tensor,
        y: torch.Tensor,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        执行逆向扩散过程（去噪）
        
        Args:
            model: DiT模型
            x_t: 初始噪声图像
            y: 条件标签
            show_progress: 是否显示进度条
            
        Returns:
            生成的最终图像和每一步的图像列表
        """
        device = x_t.device
        model.eval()
        
        # 存储生成过程中的图像
        steps = [x_t.clone()]
        
        # 设置进度条
        time_range = range(self.num_steps - 1, -1, -1)
        if show_progress:
            time_range = tqdm(time_range, desc="生成图像")
            
        with torch.no_grad():
            for time_step in time_range:
                # 创建批量时间步
                batch_time = torch.full((x_t.size(0),), time_step, device=device, dtype=torch.long)
                
                # 获取p(x_{t-1} | x_t)的均值和方差
                pred_mean, variance = self.p_mean_variance(model, x_t, batch_time, y)
                
                # 如果不是最后一步，添加噪声
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                    x_t = pred_mean + torch.sqrt(variance) * noise
                else:
                    # 最后一步不添加噪声
                    x_t = pred_mean
                
                # 限制图像范围至[-1, 1]
                x_t = torch.clamp(x_t, -1.0, 1.0)
                
                # 保存当前步骤的图像
                steps.append(x_t.clone())
        
        return x_t, steps
