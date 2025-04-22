"""
DiT (Diffusion Transformer) MNIST训练示例
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import sys
import time
from tqdm import tqdm

# 添加项目根目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_models.models import get_model
from mini_models.datasets import get_mnist_dataloaders
from mini_models.models.vision.diffusion.diffusion_process import DiffusionProcess
from mini_models.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DiT MNIST训练示例')
    parser.add_argument('--batch-size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-dir', type=str, default='./trained_models', help='模型保存路径')
    parser.add_argument('--diffusion-steps', type=int, default=1000, help='扩散步骤数量')
    return parser.parse_args()

def train(
    model, 
    diffusion, 
    train_loader, 
    optimizer, 
    device, 
    epoch, 
    log_interval=10
):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch} [Train]") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # 规范化数据到[-1, 1]
            data = (data - 0.5) * 2
            
            # 随机时间步
            t = torch.randint(0, diffusion.num_steps, (data.shape[0],), device=device).long()
            
            # 添加噪声
            noise = torch.randn_like(data)
            noisy_data = diffusion.q_sample(data, t, noise=noise)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播 - 预测噪声
            pred_noise = model(noisy_data, t, target)
            
            # 计算损失
            loss = nn.MSELoss()(pred_noise, noise)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 参数更新
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            
            # 更新进度条
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = train_loss / len(train_loader)
    logger.info(f"Train Epoch: {epoch} Loss: {avg_loss:.6f}")
    return avg_loss

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # 更新配置
    config["use_cuda"] = use_cuda
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    logger.info("Loading MNIST dataset...")
    dataloaders = get_mnist_dataloaders(batch_size=args.batch_size)
    train_loader = dataloaders["train"]
    
    # 创建模型
    logger.info("Creating DiT model...")
    model = get_model(
        "dit_mnist", 
        pretrained=False, 
        img_size=28,
        patch_size=4,
        channel=1,
        emb_size=64,
        label_num=10,
        dit_num=3,
        head=4
    )
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 创建扩散过程
    diffusion = DiffusionProcess(num_steps=args.diffusion_steps)
    
    # 训练模型
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model, 
            diffusion, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            log_interval=10
        )
        
        # 每10个epoch保存一次模型
        if epoch % 1 == 0:
            model_path = os.path.join(args.save_dir, f"dit_mnist_epoch_{epoch}.pth")
            model.save_weights(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # 训练完成
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, "dit_mnist_final.pth")
    model.save_weights(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
