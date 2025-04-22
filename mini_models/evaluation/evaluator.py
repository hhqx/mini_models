"""
模型评估器
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from mini_models.models.base import BaseModel
from mini_models.config import config

logger = logging.getLogger(__name__)

class Evaluator:
    """
    模型评估器
    """
    def __init__(
        self,
        model: BaseModel,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            test_loader: 测试数据加载器
            criterion: 损失函数
            device: 评估设备
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and config["use_cuda"] else 'cpu')
        else:
            self.device = device
            
        # 设置模型
        self.model = model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 数据加载器
        self.test_loader = test_loader
        
        # 设置默认损失函数
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
    
    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            verbose: 是否显示进度条
            
        Returns:
            包含评估结果的字典
        """
        self.model.eval()
        
        test_loss = 0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc="Evaluating", disable=not verbose) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 前向传播
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # 统计
                    test_loss += loss.item()
                    
                    # 获取预测结果
                    pred = output.argmax(dim=1, keepdim=True).squeeze()
                    
                    # 收集预测和目标
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': loss.item()})
        
        # 计算评估指标
        avg_loss = test_loss / len(self.test_loader)
        
        # 转为numpy数组便于计算
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算各种指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # 计算评估时间
        eval_time = time.time() - start_time
        
        # 创建结果字典
        results = {
            'loss': avg_loss,
            'accuracy': accuracy * 100,  # 转为百分比
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'confusion_matrix': conf_matrix,
            'eval_time': eval_time
        }
        
        # 打印评估结果
        if verbose:
            logger.info(f"Evaluation completed in {eval_time:.2f}s")
            logger.info(f"Loss: {avg_loss:.4f}")
            logger.info(f"Accuracy: {accuracy * 100:.2f}%")
            logger.info(f"Precision: {precision * 100:.2f}%")
            logger.info(f"Recall: {recall * 100:.2f}%")
            logger.info(f"F1 Score: {f1 * 100:.2f}%")
        
        return results
    
    def predict(self, data_loader: DataLoader, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        Args:
            data_loader: 数据加载器
            verbose: 是否显示进度条
            
        Returns:
            预测结果、目标值和预测概率的元组
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            with tqdm(data_loader, desc="Predicting", disable=not verbose) as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 前向传播
                    output = self.model(data)
                    
                    # 计算概率
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    
                    # 获取预测结果
                    pred = output.argmax(dim=1)
                    
                    # 收集预测、目标和概率
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_probabilities)
        )
