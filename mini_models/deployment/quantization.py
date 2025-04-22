"""
模型量化工具
"""
import torch
import torch.nn as nn
import torch.quantization
from typing import Optional, Dict, Any, List
import logging
import copy
import time

from mini_models.models.base import BaseModel
from mini_models.config import config

logger = logging.getLogger(__name__)

def prepare_model_for_quantization(model: BaseModel) -> torch.nn.Module:
    """
    准备模型进行量化
    
    Args:
        model: 原始模型
        
    Returns:
        准备好的模型
    """
    # 创建模型副本
    model_fp32 = copy.deepcopy(model)
    
    # 设置为评估模式
    model_fp32.eval()
    
    # 增加量化/反量化存根
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 融合模块 (如Conv+BN+ReLU)
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu']], inplace=False)
    
    # 准备量化
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    
    return model_fp32_prepared

def quantize_model(
    model: BaseModel,
    calibration_data: Optional[List[torch.Tensor]] = None,
    quantization_type: str = 'static',
    backend: str = 'fbgemm'  # 'fbgemm'用于x86, 'qnnpack'用于ARM
) -> torch.nn.Module:
    """
    量化模型
    
    Args:
        model: 原始模型
        calibration_data: 用于校准的数据
        quantization_type: 量化类型 ('static' 或 'dynamic')
        backend: 量化后端
        
    Returns:
        量化后的模型
    """
    # 记录开始时间
    start_time = time.time()
    logger.info(f"Starting {quantization_type} quantization with {backend} backend")
    
    if quantization_type == 'dynamic':
        # 动态量化 (无需校准数据)
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
    else:  # static
        # 准备模型进行静态量化
        prepared_model = prepare_model_for_quantization(model)
        
        # 如果提供了校准数据，则进行校准
        if calibration_data:
            logger.info("Calibrating with provided data")
            with torch.no_grad():
                for data in calibration_data:
                    prepared_model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(prepared_model)
    
    # 记录完成时间
    elapsed_time = time.time() - start_time
    logger.info(f"Quantization completed in {elapsed_time:.2f}s")
    
    # 记录模型大小变化
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    logger.info(f"Original model size: {original_size / 1e6:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size / 1e6:.2f} MB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    return quantized_model

def get_model_size(model: torch.nn.Module) -> int:
    """
    估算模型大小
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型大小（字节）
    """
    param_size = 0
    buffer_size = 0
    
    # 计算参数大小
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # 计算缓冲区大小
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return param_size + buffer_size
