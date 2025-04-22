"""
Vision models package
"""
# This file makes the vision directory a Python package

from mini_models.models.vision.mnist_model import MNISTModel  # Keep existing imports
from mini_models.models.vision.resnet import ResNet18Mini


from mini_models.models.vision.diffusion.dit_model import DiTModel