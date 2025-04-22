"""
视觉模型模块
"""

from mini_models.models.vision.mnist_model import MNISTModel
from mini_models.models.vision.resnet import ResNet18Mini
from mini_models.models.vision.diffusion.dit_model import DiTModel

__all__ = ['MNISTModel', 'ResNet18Mini', 'DiTModel']
