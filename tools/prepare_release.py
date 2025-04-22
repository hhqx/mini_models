"""
准备模型发布的通用工具脚本
"""
import os
import sys
import hashlib
import shutil
import argparse
import json
import yaml
from typing import Dict, Any, Optional, List

def calculate_sha256(filepath: str) -> str:
    """计算文件的SHA256哈希值"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def prepare_model_release(
    model_path: str,
    model_name: str,
    task_type: str,
    description: str,
    output_dir: str,
    version: str = "v0.1",
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    准备单个模型的发布文件
    
    Args:
        model_path: 模型文件路径
        model_name: 模型名称标识
        task_type: 任务类型（如'image_classification', 'image_generation'等）
        description: 模型描述
        output_dir: 输出目录
        version: 发布版本
        category: 模型类别（如'vision', 'nlp', 'diffusion'等）
        
    Returns:
        模型信息字典
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return {}
    
    # 确定模型类别
    if category is None:
        # 根据任务类型推断类别
        if task_type in ['image_classification', 'image_generation', 'object_detection']:
            category = 'vision'
        elif task_type in ['text_classification', 'translation', 'text_generation']:
            category = 'nlp'
        elif task_type in ['audio_classification', 'speech_recognition']:
            category = 'audio'
        else:
            category = 'misc'
    
    # 创建类别目录
    release_dir = os.path.join(output_dir, version, category)
    os.makedirs(release_dir, exist_ok=True)
    
    # 准备发布文件名（使用模型名称）
    release_filename = f"{model_name}.pth"
    release_path = os.path.join(release_dir, release_filename)
    
    # 复制模型文件
    shutil.copy2(model_path, release_path)
    
    # 计算文件大小和哈希值
    file_size = os.path.getsize(release_path)
    sha256 = calculate_sha256(release_path)
    
    # 准备模型信息
    model_info = {
        model_name: {
            "url": f"{version}/{category}/{release_filename}",
            "size": file_size,
            "sha256": sha256,
            "task": task_type,
            "description": description
        }
    }
    
    print(f"Model prepared for release:")
    print(f"- Name: {model_name}")
    print(f"- Path: {release_path}")
    print(f"- Size: {file_size / 1024 / 1024:.2f}MB")
    print(f"- SHA256: {sha256}")
    
    return model_info

def update_model_info(output_dir: str, model_info: Dict[str, Any]) -> str:
    """
    更新模型信息文件
    
    Args:
        output_dir: 输出目录
        model_info: 新的模型信息
        
    Returns:
        信息文件路径
    """
    # 信息文件路径
    json_path = os.path.join(output_dir, "model_info.json")
    yaml_path = os.path.join(output_dir, "model_info.yaml")
    
    # 如果文件存在，加载并更新
    existing_info = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_info = json.load(f)
    elif os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            existing_info = yaml.safe_load(f) or {}
    
    # 更新模型信息
    existing_info.update(model_info)
    
    # 保存JSON格式
    with open(json_path, 'w') as f:
        json.dump(existing_info, f, indent=2)
    
    # 保存YAML格式
    with open(yaml_path, 'w') as f:
        yaml.dump(existing_info, f, default_flow_style=False, sort_keys=False)
    
    print(f"Model info saved to: {json_path} and {yaml_path}")
    return json_path

def generate_release_notes(model_infos: Dict[str, Any], version: str) -> str:
    """
    生成发布说明
    
    Args:
        model_infos: 模型信息字典
        version: 发布版本
        
    Returns:
        发布说明文本
    """
    release_notes = f"# Mini Models Release {version}\n\n"
    
    # 按任务类型分组模型
    task_groups = {}
    for model_name, info in model_infos.items():
        task = info.get("task", "other")
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append((model_name, info))
    
    # 为每个任务组生成说明
    for task, models in task_groups.items():
        release_notes += f"## {task.replace('_', ' ').title()} Models\n\n"
        for model_name, info in models:
            release_notes += f"### {model_name}\n\n"
            release_notes += f"- **Description**: {info['description']}\n"
            release_notes += f"- **Size**: {info['size'] / 1024 / 1024:.2f}MB\n"
            release_notes += f"- **SHA256**: {info['sha256']}\n\n"
            release_notes += "```python\n"
            release_notes += f"from mini_models.models import get_model\n\n"
            release_notes += f"# Load the model\n"
            release_notes += f"model = get_model(\"{model_name}\", pretrained=True)\n"
            release_notes += "```\n\n"
    
    return release_notes

def main():
    parser = argparse.ArgumentParser(description="准备模型发布文件")
    parser.add_argument("--model-path", type=str, help="训练好的模型文件路径")
    parser.add_argument("--model-name", type=str, help="模型名称标识")
    parser.add_argument("--task-type", type=str, help="任务类型")
    parser.add_argument("--description", type=str, help="模型描述")
    parser.add_argument("--output-dir", type=str, default="./release", help="发布文件输出目录")
    parser.add_argument("--version", type=str, default="v0.1", help="发布版本")
    parser.add_argument("--category", type=str, help="模型类别")
    parser.add_argument("--batch", type=str, help="批量处理配置文件路径")
    parser.add_argument("--generate-notes", action="store_true", help="生成发布说明")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        if not os.path.exists(args.batch):
            print(f"Error: Batch config file {args.batch} not found!")
            return
        
        # 根据文件扩展名决定加载方式
        if args.batch.endswith('.yaml') or args.batch.endswith('.yml'):
            with open(args.batch, 'r') as f:
                batch_config = yaml.safe_load(f)
        elif args.batch.endswith('.json'):
            with open(args.batch, 'r') as f:
                batch_config = json.load(f)
        else:
            print(f"Error: Unsupported config file format: {args.batch}")
            print("Supported formats: .yaml, .yml, .json")
            return
        
        all_model_infos = {}
        for model_config in batch_config.get("models", []):
            # 跳过被注释掉的模型配置
            if not model_config or not isinstance(model_config, dict):
                continue
                
            model_info = prepare_model_release(
                model_path=model_config["path"],
                model_name=model_config["name"],
                task_type=model_config["task"],
                description=model_config["description"],
                output_dir=args.output_dir,
                version=model_config.get("version", args.version),
                category=model_config.get("category")
            )
            all_model_infos.update(model_info)
        
        # 更新模型信息文件
        if all_model_infos:
            info_path = update_model_info(args.output_dir, all_model_infos)
            
            # 生成发布说明
            if args.generate_notes:
                release_notes = generate_release_notes(all_model_infos, args.version)
                notes_path = os.path.join(args.output_dir, f"release_notes_{args.version}.md")
                with open(notes_path, 'w') as f:
                    f.write(release_notes)
                print(f"Release notes saved to: {notes_path}")
        else:
            print("Warning: No valid model configurations found in batch file")
    
    elif args.model_path and args.model_name and args.task_type and args.description:
        # 单个模型处理模式
        model_info = prepare_model_release(
            model_path=args.model_path,
            model_name=args.model_name,
            task_type=args.task_type,
            description=args.description,
            output_dir=args.output_dir,
            version=args.version,
            category=args.category
        )
        
        # 更新模型信息文件
        info_path = update_model_info(args.output_dir, model_info)
        
        # 生成发布说明
        if args.generate_notes:
            release_notes = generate_release_notes(model_info, args.version)
            notes_path = os.path.join(args.output_dir, f"release_notes_{args.version}.md")
            with open(notes_path, 'w') as f:
                f.write(release_notes)
            print(f"Release notes saved to: {notes_path}")
    
    else:
        parser.print_help()
        print("\nError: 必须提供--batch参数或者提供所有必要的单模型参数（model-path, model-name, task-type, description）")
        print("\nExample:")
        print("  # 使用YAML配置批量处理:")
        print(f"  python {sys.argv[0]} --batch models_config.yaml --output-dir ./release --generate-notes")
        print("\n  # 单模型处理:")
        print(f"  python {sys.argv[0]} --model-path ./trained_models/model.pth --model-name mymodel --task-type image_classification --description \"My Model\"")

if __name__ == "__main__":
    main()
