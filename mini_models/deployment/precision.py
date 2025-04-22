"""
模型精度测试工具
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from tabulate import tabulate
import logging

from mini_models.models.base import BaseModel
from mini_models.config import config

logger = logging.getLogger(__name__)

def evaluate_precision(
    original_model: BaseModel,
    quantized_model: nn.Module,
    test_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    num_batches: int = -1  # -1表示使用所有批次
) -> Dict[str, Any]:
    """
    评估原始模型和量化模型的精度差异
    
    Args:
        original_model: 原始模型
        quantized_model: 量化后的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 评估设备
        num_batches: 用于评估的批次数（-1表示全部）
        
    Returns:
        包含评估结果的字典
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and config["use_cuda"] else 'cpu')
    
    # 设置损失函数
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # 将模型设置为评估模式
    original_model.to(device).eval()
    
    # 注意：量化模型通常在CPU上运行最快
    quantized_model.eval()
    
    # 初始化指标
    metrics = {
        "original": {
            "accuracy": 0,
            "loss": 0,
            "inference_time": 0,
        },
        "quantized": {
            "accuracy": 0,
            "loss": 0,
            "inference_time": 0,
        },
    }
    
    batch_count = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if num_batches > 0 and batch_idx >= num_batches:
            break
        
        batch_count += 1
        batch_size = data.size(0)
        total_samples += batch_size
        
        # 评估原始模型
        original_data = data.to(device)
        original_target = target.to(device)
        
        start_time = time.time()
        with torch.no_grad():
            original_output = original_model(original_data)
            original_loss = criterion(original_output, original_target)
        original_time = time.time() - start_time
        
        original_pred = original_output.argmax(dim=1)
        original_correct = original_pred.eq(original_target).sum().item()
        
        # 累加指标
        metrics["original"]["accuracy"] += original_correct
        metrics["original"]["loss"] += original_loss.item() * batch_size
        metrics["original"]["inference_time"] += original_time
        
        # 评估量化模型（在CPU上）
        start_time = time.time()
        with torch.no_grad():
            quantized_output = quantized_model(data)  # 量化模型通常在CPU上
            quantized_loss = criterion(quantized_output, target)  # 目标也应在CPU上
        quantized_time = time.time() - start_time
        
        quantized_pred = quantized_output.argmax(dim=1)
        quantized_correct = quantized_pred.eq(target).sum().item()
        
        # 累加指标
        metrics["quantized"]["accuracy"] += quantized_correct
        metrics["quantized"]["loss"] += quantized_loss.item() * batch_size
        metrics["quantized"]["inference_time"] += quantized_time
    
    # 计算平均指标
    for model_type in metrics:
        metrics[model_type]["accuracy"] = 100 * metrics[model_type]["accuracy"] / total_samples
        metrics[model_type]["loss"] /= total_samples
        metrics[model_type]["inference_time"] /= batch_count
    
    # 计算比较指标
    metrics["comparison"] = {
        "accuracy_diff": metrics["original"]["accuracy"] - metrics["quantized"]["accuracy"],
        "loss_diff": metrics["original"]["loss"] - metrics["quantized"]["loss"],
        "speedup": metrics["original"]["inference_time"] / metrics["quantized"]["inference_time"]
                   if metrics["quantized"]["inference_time"] > 0 else float('inf')
    }
    
    # 打印结果表格
    table_data = [
        ["Metric", "Original Model", "Quantized Model", "Difference/Ratio"],
        ["Accuracy (%)", f"{metrics['original']['accuracy']:.2f}",
         f"{metrics['quantized']['accuracy']:.2f}", f"{metrics['comparison']['accuracy_diff']:.2f}"],
        ["Loss", f"{metrics['original']['loss']:.4f}",
         f"{metrics['quantized']['loss']:.4f}", f"{metrics['comparison']['loss_diff']:.4f}"],
        ["Inference Time (ms)", f"{metrics['original']['inference_time'] * 1000:.2f}",
         f"{metrics['quantized']['inference_time'] * 1000:.2f}", f"{metrics['comparison']['speedup']:.2f}x"]
    ]
    
    logger.info("\nPrecision Evaluation Results:")
    logger.info(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    return metrics
