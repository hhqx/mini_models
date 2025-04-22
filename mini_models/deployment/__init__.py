"""
模型部署模块
"""

from mini_models.deployment.quantization import quantize_model
from mini_models.deployment.precision import evaluate_precision

__all__ = ['quantize_model', 'evaluate_precision']
