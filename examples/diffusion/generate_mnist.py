"""
使用DiT模型生成MNIST图像示例
"""
import torch
import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_models.models import get_model
from mini_models.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DiT MNIST图像生成示例')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model-path', type=str, help='模型路径，如果不指定则使用预训练模型')
    parser.add_argument('--batch-size', type=int, default=10, help='生成图像的批次大小')
    parser.add_argument('--diffusion-steps', type=int, default=1000, help='扩散步骤数量')
    parser.add_argument('--num-display-steps', type=int, default=20, help='展示的去噪步骤数量')
    parser.add_argument('--output-dir', type=str, default='./generated_images', help='生成图像的保存路径')
    return parser.parse_args()

def plot_generation_process(steps, labels, num_steps_to_show, output_path=None):
    """绘制生成过程"""
    batch_size = len(labels)
    
    # 选择要显示的步骤
    indices = []
    step_size = len(steps) // num_steps_to_show
    for i in range(num_steps_to_show):
        indices.append(min((i + 1) * step_size, len(steps) - 1))
    
    plt.figure(figsize=(num_steps_to_show * 1.5, batch_size * 1.5))
    
    for b in range(batch_size):
        for i, idx in enumerate(indices):
            # 像素值从[-1,1]转换为[0,1]
            img = (steps[idx][b].cpu() + 1) / 2
            # 转换为可视化格式
            img = img.permute(1, 2, 0).squeeze()
            
            plt.subplot(batch_size, num_steps_to_show, b * num_steps_to_show + i + 1)
            plt.imshow(img, cmap='gray')
            
            # 第一列显示标签
            if i == 0:
                plt.ylabel(f"Label: {labels[b].item()}")
                
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    
    # 如果指定了输出路径，则保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    plt.show()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # 更新配置
    config["use_cuda"] = use_cuda
    
    # 加载模型
    logger.info("Loading DiT model...")
    model = get_model(
        "dit_mnist", 
        pretrained=True if args.model_path is None else False,
        img_size=28,
        patch_size=4,
        channel=1,
        emb_size=64,
        label_num=10,
        dit_num=3,
        head=4
    )
    
    # 如果指定了模型路径，加载权重
    if args.model_path:
        logger.info(f"Loading weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    # 创建数字0-9的标签序列
    labels = torch.arange(0, 10, device=device)
    
    # 生成图像
    logger.info("Generating images...")
    with torch.no_grad():
        images, steps = model.generate(
            batch_size=10,
            labels=labels,
            device=device,
            num_steps=args.diffusion_steps,
            show_progress=True
        )
    
    # 显示生成过程
    logger.info("Plotting generation process...")
    output_path = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"generated_mnist_{args.seed}.png")
    
    plot_generation_process(steps, labels, args.num_display_steps, output_path)
    
    if output_path:
        logger.info(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    main()
