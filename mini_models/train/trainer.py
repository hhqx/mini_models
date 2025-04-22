"""
模型训练器
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from tqdm import tqdm
import logging

from mini_models.models.base import BaseModel
from mini_models.config import config

logger = logging.getLogger(__name__)

class Trainer:
    """
    模型训练器，支持常见训练策略和技巧
    """
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            device: 训练设备
            config_dict: 训练配置
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and config["use_cuda"] else 'cpu')
        else:
            self.device = device
            
        # 设置模型
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设置默认损失函数
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # 设置默认优化器
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.lr_scheduler = lr_scheduler
        
        # 默认训练配置
        self.config = {
            "epochs": 10,
            "save_dir": os.path.join(config["cache_dir"], "trained_models"),
            "save_best_only": True,
            "early_stopping_patience": 0,  # 0表示不使用早停
            "verbose": 1,
            "log_interval": 10,
        }
        
        # 更新配置
        if config_dict:
            self.config.update(config_dict)
        
        # 创建保存目录
        os.makedirs(self.config["save_dir"], exist_ok=True)
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        训练模型
        
        Returns:
            训练历史记录
        """
        total_start_time = time.time()
        
        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 如果有验证集，进行验证
            if self.val_loader:
                val_loss, val_acc = self._validate_epoch(epoch)
                
                # 更新最佳模型
                if self.config["save_best_only"]:
                    improved = False
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        improved = True
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        improved = True
                        
                    if improved:
                        self._save_model(f"best_model.pth")
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                    
                    # 早停检查
                    if (self.config["early_stopping_patience"] > 0 and 
                        self.epochs_without_improvement >= self.config["early_stopping_patience"]):
                        logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
            else:
                # 没有验证集时，直接保存模型
                if epoch % 5 == 0:  # 每5个epoch保存一次
                    self._save_model(f"model_epoch_{epoch}.pth")
            
            # 更新学习率
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.lr_scheduler.step()
            
            # 记录训练历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
            if self.val_loader:
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
            
            # 打印epoch信息
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch}/{self.config['epochs']} completed in {epoch_time:.2f}s")
        
        # 训练完成
        total_time = time.time() - total_start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # 返回训练历史
        return self.history
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", disable=not self.config["verbose"]) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 参数更新
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 更新进度条
                if batch_idx % self.config["log_interval"] == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100. * correct / total
                    })
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        logger.info(f"Train Epoch: {epoch} Loss: {avg_loss:.6f} Acc: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            平均损失和准确率
        """
        self.model.eval()
        
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", disable=not self.config["verbose"]) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 前向传播
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # 统计
                    val_loss += loss.item()
                    
                    # 计算准确率
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100. * correct / total
                    })
        
        # 计算平均损失和准确率
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        logger.info(f"Validation Epoch: {epoch} Loss: {avg_loss:.6f} Acc: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def _save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.config["save_dir"], filename)
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.model.config,
            'history': self.history,
        }
        
        if self.lr_scheduler:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        torch.save(state_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        从检查点加载模型
        
        Args:
            filepath: 检查点路径
        """
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint {filepath} does not exist.")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'lr_scheduler' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {filepath}")
