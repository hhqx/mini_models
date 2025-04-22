"""
DiT模型单元测试
"""
import unittest
import torch
import os
import sys

# 添加项目根目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_models.models import get_model
from mini_models.models.vision.diffusion.diffusion_process import DiffusionProcess

class TestDiTModel(unittest.TestCase):
    """DiT模型测试类"""
    
    def setUp(self):
        """测试准备"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(
            "dit_mnist",
            pretrained=False,
            img_size=28,
            patch_size=4,
            channel=1,
            emb_size=64,
            label_num=10,
            dit_num=3,
            head=4
        ).to(self.device)
        self.batch_size = 4
        self.diffusion = DiffusionProcess(num_steps=10)  # 使用较少的步骤进行测试
    
    def test_model_forward(self):
        """测试模型前向传播"""
        # 创建输入
        x = torch.randn(self.batch_size, 1, 28, 28, device=self.device)
        t = torch.randint(0, 10, (self.batch_size,), device=self.device)
        y = torch.randint(0, 10, (self.batch_size,), device=self.device)
        
        # 前向传播
        output = self.model(x, t, y)
        
        # 检查输出尺寸
        self.assertEqual(output.shape, x.shape)
    
    def test_diffusion_process(self):
        """测试扩散过程"""
        # 创建干净图像
        x0 = torch.randn(self.batch_size, 1, 28, 28, device=self.device)
        
        # 添加噪声到t=5
        t = torch.full((self.batch_size,), 5, device=self.device)
        noise = torch.randn_like(x0)
        xt = self.diffusion.q_sample(x0, t, noise)
        
        # 检查xt形状
        self.assertEqual(xt.shape, x0.shape)
        
        # 确保xt不等于x0（添加了噪声）
        self.assertFalse(torch.allclose(xt, x0))
    
    def test_generation(self):
        """测试图像生成"""
        # 生成图像（使用较少的步骤进行测试）
        with torch.no_grad():
            labels = torch.arange(0, 4, device=self.device)
            images, steps = self.model.generate(
                batch_size=4,
                labels=labels,
                device=self.device,
                num_steps=10,
                show_progress=False
            )
        
        # 检查生成的图像尺寸
        self.assertEqual(images.shape, (4, 1, 28, 28))
        
        # 检查步骤数量（应该是num_steps+1，包括初始噪声）
        self.assertEqual(len(steps), 11)
        
        # 检查每个步骤的形状
        for step in steps:
            self.assertEqual(step.shape, (4, 1, 28, 28))

if __name__ == "__main__":
    unittest.main()
