"""
MNIST数据集处理模块
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict

from mini_models.config import config

class MNISTDataset(Dataset):
    """
    MNIST数据集封装
    """
    def __init__(self, train: bool = True, transform=None):
        """
        初始化MNIST数据集
        
        Args:
            train: 是否加载训练集
            transform: 数据变换
        """
        self.train = train
        
        # 设置默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transform
        
        # 获取数据集存储路径
        data_dir = config["datasets_dir"]
        
        # 加载数据集
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            图像和标签的元组
        """
        return self.dataset[idx]

def get_mnist_dataloaders(
    batch_size: int = 64,
    train_transform=None,
    test_transform=None,
    num_workers: int = 4,
    shuffle: bool = True
) -> Dict[str, DataLoader]:
    """
    获取MNIST数据加载器
    
    Args:
        batch_size: 批次大小
        train_transform: 训练集变换
        test_transform: 测试集变换
        num_workers: 加载数据的线程数
        shuffle: 是否打乱数据
        
    Returns:
        包含训练集和测试集数据加载器的字典
    """
    # 创建数据集
    train_dataset = MNISTDataset(train=True, transform=train_transform)
    test_dataset = MNISTDataset(train=False, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        "train": train_loader,
        "test": test_loader
    }
