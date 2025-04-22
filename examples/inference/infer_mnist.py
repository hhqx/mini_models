"""
MNIST推理示例
"""
import torch
import torch.nn as nn
import argparse
import logging
import os
import sys
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 添加项目根目录到路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mini_models.models import get_model
from mini_models.datasets import get_mnist_dataloaders
from mini_models.evaluation import Evaluator
from mini_models.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MNIST模型推理示例')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA')
    parser.add_argument('--model-path', type=str, help='模型路径，如果不指定则使用预训练模型')
    parser.add_argument('--image-path', type=str, help='要推理的图像路径，如果不指定则使用测试集')
    parser.add_argument('--show-samples', action='store_true', help='显示测试集样本和预测结果')
    parser.add_argument('--num-samples', type=int, default=10, help='显示的样本数量')
    return parser.parse_args()

def preprocess_image(image_path):
    """预处理图像"""
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 打开并转换图像
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    
    return img_tensor, img

def predict_single_image(model, image_path, device):
    """预测单张图像"""
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    # 打印结果
    logger.info(f"Prediction: {prediction}, Confidence: {confidence * 100:.2f}%")
    
    # 显示图像和预测结果
    plt.figure(figsize=(5, 5))
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Prediction: {prediction} (Confidence: {confidence * 100:.2f}%)")
    plt.axis('off')
    plt.show()

def show_samples_with_predictions(model, test_loader, device, num_samples=10):
    """显示测试样本和预测结果"""
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # 显示样本
    fig = plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        correct = predictions[i] == labels[i]
        color = 'green' if correct else 'red'
        plt.title(f"Pred: {predictions[i]} (True: {labels[i]})", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 加载模型
    logger.info("Loading model...")
    model = get_model("mnist_cnn", pretrained=True if args.model_path is None else False)
    
    # 如果指定了模型路径，加载权重
    if args.model_path:
        logger.info(f"Loading weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    
    # 将模型移到设备上
    model = model.to(device)
    
    # 处理单张图像
    if args.image_path:
        predict_single_image(model, args.image_path, device)
    else:
        # 加载测试数据
        logger.info("Loading test dataset...")
        dataloaders = get_mnist_dataloaders(batch_size=args.batch_size)
        test_loader = dataloaders["test"]
        
        # 评估模型
        logger.info("Evaluating model...")
        evaluator = Evaluator(model, test_loader, device=device)
        evaluation_results = evaluator.evaluate()
        
        # 打印结果
        logger.info(f"Test accuracy: {evaluation_results['accuracy']:.2f}%")
        
        # 如果需要，显示样本
        if args.show_samples:
            show_samples_with_predictions(model, test_loader, device, args.num_samples)

if __name__ == "__main__":
    main()
