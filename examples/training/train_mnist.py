"""
MNIST模型训练示例
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import sys
import time

# 添加项目根目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_models.models import get_model
from mini_models.datasets import get_mnist_dataloaders
from mini_models.train import Trainer
from mini_models.evaluation import Evaluator
from mini_models.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST模型训练示例')
    parser.add_argument('--batch-size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-dir', type=str, default='./trained_models', help='模型保存路径')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 更新配置
    config["use_cuda"] = use_cuda
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    logger.info("Loading MNIST dataset...")
    dataloaders = get_mnist_dataloaders(batch_size=args.batch_size)
    train_loader = dataloaders["train"]
    test_loader = dataloaders["test"]
    
    # 创建模型
    logger.info("Creating model...")
    model = get_model("mnist_cnn", pretrained=False)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # 配置训练器
    trainer_config = {
        "epochs": args.epochs,
        "save_dir": args.save_dir,
        "save_best_only": True,
        "early_stopping_patience": 5,
        "verbose": 1,
        "log_interval": 10,
    }
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        config_dict=trainer_config
    )
    
    # 训练模型
    logger.info("Starting training...")
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # 评估模型
    logger.info("Evaluating model...")
    evaluator = Evaluator(model, test_loader, criterion, device)
    evaluation_results = evaluator.evaluate()
    
    # 打印结果
    logger.info(f"Final test accuracy: {evaluation_results['accuracy']:.2f}%")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, "mnist_final_model.pth")
    model.save_weights(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
