"""
演示模型权重管理的高级用法
"""
import argparse
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_models.weights.manager import weight_manager
from mini_models.models import get_model, import_user_weight, check_model_updates, weight_info, list_available_models

def print_separator():
    print("\n" + "-" * 60 + "\n")

def demo_weight_info():
    """展示权重信息查询"""
    print("展示模型权重信息:")
    
    for model_name in list_available_models():
        info = weight_info(model_name)
        print(f"\n模型: {model_name}")
        print(f"  已缓存版本: {len(info['cached_versions'])}")
        print(f"  最新版本: {info['latest_version'] or '未知'}")
        print(f"  用户自定义权重: {'是' if info['has_user_weight'] else '否'}")

def demo_weight_management():
    """展示权重管理功能"""
    print("模型权重管理演示:")
    
    # 1. 列出所有可用模型
    models = list_available_models()
    print(f"可用模型: {models}")
    
    if not models:
        print("无可用模型，退出演示")
        return
    
    # 2. 选择第一个模型进行演示
    model_name = models[0]
    print(f"\n选择模型 {model_name} 进行演示")
    
    # 3. 检查模型权重更新
    print("\n检查权重更新...")
    updates = check_model_updates(model_name)
    print(f"模型 {model_name} {'有' if updates.get(model_name, False) else '没有'}可用更新")
    
    # 4. 下载权重
    print("\n下载最新权重...")
    weight_path = weight_manager.download_weight(model_name, force=False)
    if weight_path:
        print(f"权重已下载到: {weight_path}")
    
    # 5. 列出可用版本
    print("\n列出可用版本...")
    versions = weight_manager.list_weight_versions(model_name)
    for version in versions:
        print(f"  版本: {version['version']}")
        print(f"  下载时间: {version['downloaded_at']}")
        print(f"  最后使用: {version['last_used']}")
        print()
    
    # 6. 加载模型（使用不同的权重选项）
    print("\n使用最新权重加载模型...")
    model = get_model(model_name, pretrained=True)
    print(f"模型加载成功，是否使用预训练: {model.is_pretrained}")

def main():
    parser = argparse.ArgumentParser(description="模型权重管理演示")
    parser.add_argument("--info", action="store_true", help="展示权重信息")
    parser.add_argument("--demo", action="store_true", help="运行权重管理演示")
    
    args = parser.parse_args()
    
    if args.info:
        demo_weight_info()
    elif args.demo:
        demo_weight_management()
    else:
        parser.print_help()
        print("\n请选择一个演示选项")

if __name__ == "__main__":
    main()
