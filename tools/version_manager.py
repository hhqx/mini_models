"""
mini_models 版本管理工具

用于管理项目的版本号、模型版本和发布流程
"""
import os
import re
import sys
import yaml
import json
import argparse
import subprocess
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 版本信息文件
VERSION_FILE = os.path.join(PROJECT_ROOT, "mini_models", "version.py")
# 模型注册文件
MODELS_REGISTRY_FILE = os.path.join(PROJECT_ROOT, "release", "model_registry.yaml")

@dataclass
class Version:
    """版本号，遵循语义化版本控制"""
    major: int
    minor: int
    patch: int
    pre: Optional[str] = None
    
    def __str__(self) -> str:
        """版本号字符串表示"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            version += f"-{self.pre}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> 'Version':
        """从字符串解析版本号"""
        version_pattern = r"(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\.]+))?"
        match = re.match(version_pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        pre = match.group(4)
        
        return cls(major, minor, patch, pre)

def get_current_version() -> Version:
    """获取当前项目版本号"""
    version_pattern = r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]'
    
    with open(VERSION_FILE, "r") as f:
        content = f.read()
    
    match = re.search(version_pattern, content)
    if not match:
        raise ValueError(f"Version not found in {VERSION_FILE}")
    
    return Version.parse(match.group(1))

def set_project_version(version: Version) -> None:
    """设置项目版本号"""
    with open(VERSION_FILE, "r") as f:
        content = f.read()
    
    # 替换版本号
    new_content = re.sub(
        r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
        f'__version__ = "{str(version)}"',
        content
    )
    
    with open(VERSION_FILE, "w") as f:
        f.write(new_content)
    
    print(f"Project version updated to: {version}")

def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """加载模型注册表"""
    if not os.path.exists(MODELS_REGISTRY_FILE):
        # 如果不存在，创建空注册表
        registry = {}
        os.makedirs(os.path.dirname(MODELS_REGISTRY_FILE), exist_ok=True)
        return registry
    
    with open(MODELS_REGISTRY_FILE, "r") as f:
        return yaml.safe_load(f) or {}

def save_model_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    """保存模型注册表"""
    os.makedirs(os.path.dirname(MODELS_REGISTRY_FILE), exist_ok=True)
    with open(MODELS_REGISTRY_FILE, "w") as f:
        yaml.dump(registry, f, default_flow_style=False)

def update_model_version(model_name: str, new_version: str, update_type: str = "patch") -> Dict[str, Any]:
    """
    更新模型版本
    
    Args:
        model_name: 模型名称
        new_version: 新版本号（如果为None则自动生成）
        update_type: 更新类型（major/minor/patch）
        
    Returns:
        更新后的模型信息
    """
    registry = load_model_registry()
    
    # 检查模型是否存在
    if model_name not in registry:
        # 如果是新模型，创建初始版本
        model_info = {
            "version": "1.0.0",
            "release_date": datetime.now().strftime("%Y-%m-%d"),
            "history": []
        }
        registry[model_name] = model_info
    else:
        model_info = registry[model_name]
        current_version = Version.parse(model_info["version"])
        
        # 保存历史版本
        model_info["history"].append({
            "version": str(current_version),
            "release_date": model_info["release_date"]
        })
        
        # 如果未指定新版本，则根据更新类型生成
        if not new_version:
            if update_type == "major":
                new_version = Version(current_version.major + 1, 0, 0)
            elif update_type == "minor":
                new_version = Version(current_version.major, current_version.minor + 1, 0)
            else:  # patch
                new_version = Version(current_version.major, current_version.minor, current_version.patch + 1)
            new_version = str(new_version)
        
        # 更新版本和日期
        model_info["version"] = new_version
        model_info["release_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # 保存注册表
    save_model_registry(registry)
    print(f"Model {model_name} updated to version {model_info['version']}")
    return model_info

def update_batch_config(model_name: str, config_file: str, new_version: str) -> None:
    """更新批量配置文件中的模型版本"""
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found")
        return
    
    # 根据文件类型读取
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Unsupported config file format: {config_file}")
        return
    
    # 更新版本
    updated = False
    for model in config.get("models", []):
        if isinstance(model, dict) and model.get("name") == model_name:
            model["version"] = new_version
            updated = True
    
    if updated:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Updated model version in config file: {config_file}")
    else:
        print(f"Model {model_name} not found in config file")

def bump_project_version(update_type: str = "patch", pre: Optional[str] = None) -> Version:
    """
    升级项目版本号
    
    Args:
        update_type: 更新类型（major/minor/patch）
        pre: 预发布标识（如alpha.1/beta.2）
        
    Returns:
        新版本号
    """
    current_version = get_current_version()
    
    if update_type == "major":
        new_version = Version(current_version.major + 1, 0, 0, pre)
    elif update_type == "minor":
        new_version = Version(current_version.major, current_version.minor + 1, 0, pre)
    elif update_type == "patch":
        new_version = Version(current_version.major, current_version.minor, current_version.patch + 1, pre)
    elif update_type == "pre":
        # 只更新预发布标识
        new_version = Version(current_version.major, current_version.minor, current_version.patch, pre)
    else:
        raise ValueError(f"Invalid update type: {update_type}")
    
    # 更新版本文件
    set_project_version(new_version)
    return new_version

def create_release(version: str, models: List[str] = None) -> None:
    """
    创建项目发布
    
    Args:
        version: 版本号
        models: 包含的模型列表（如果为None，则包含所有模型）
    """
    registry = load_model_registry()
    
    if models is None:
        # 使用所有模型
        models = list(registry.keys())
    
    # 准备发布说明
    release_notes = f"# Release v{version}\n\n"
    release_notes += f"Release Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    
    # 添加模型信息
    release_notes += "## Included Models\n\n"
    for model_name in models:
        if model_name in registry:
            model_info = registry[model_name]
            release_notes += f"### {model_name} (v{model_info['version']})\n\n"
    
    # 保存发布说明
    notes_path = os.path.join(PROJECT_ROOT, "release", f"release_notes_{version}.md")
    os.makedirs(os.path.dirname(notes_path), exist_ok=True)
    with open(notes_path, "w") as f:
        f.write(release_notes)
    
    print(f"Release notes created at: {notes_path}")
    
    # 使用release_to_github.py创建发布
    release_script = os.path.join(PROJECT_ROOT, "tools", "release_to_github.py")
    if os.path.exists(release_script):
        command = [
            sys.executable, release_script,
            "--version", f"v{version}",
            "--notes", notes_path,
            "--dry-run"  # 先以干运行模式显示，确认后再执行
        ]
        print("Execute the following command to create GitHub release:")
        print(" ".join(command))

def main():
    parser = argparse.ArgumentParser(description="mini_models 版本管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 获取当前版本
    get_version_parser = subparsers.add_parser("get-version", help="获取当前版本")
    
    # 升级项目版本
    bump_parser = subparsers.add_parser("bump", help="升级项目版本")
    bump_parser.add_argument("type", choices=["major", "minor", "patch", "pre"], help="更新类型")
    bump_parser.add_argument("--pre", help="预发布标识（如alpha.1）")
    
    # 更新模型版本
    update_model_parser = subparsers.add_parser("update-model", help="更新模型版本")
    update_model_parser.add_argument("model", help="模型名称")
    update_model_parser.add_argument("--version", help="新版本号（如不指定则自动生成）")
    update_model_parser.add_argument("--type", choices=["major", "minor", "patch"], default="patch", help="更新类型")
    update_model_parser.add_argument("--config", help="要更新的批量配置文件")
    
    # 创建发布
    release_parser = subparsers.add_parser("release", help="创建发布")
    release_parser.add_argument("--version", help="发布版本号（如不指定则使用当前版本）")
    release_parser.add_argument("--models", nargs="*", help="包含的模型（如不指定则包含所有模型）")
    
    # 列出模型版本
    list_parser = subparsers.add_parser("list", help="列出所有模型版本")
    
    args = parser.parse_args()
    
    if args.command == "get-version":
        version = get_current_version()
        print(f"Current version: {version}")
        
    elif args.command == "bump":
        new_version = bump_project_version(args.type, args.pre)
        print(f"Project version bumped to: {new_version}")
        
    elif args.command == "update-model":
        model_info = update_model_version(args.model, args.version, args.type)
        print(f"Model {args.model} updated to version {model_info['version']}")
        
        # 如果指定了配置文件，同时更新配置
        if args.config:
            update_batch_config(args.model, args.config, model_info['version'])
        
    elif args.command == "release":
        version = args.version or str(get_current_version())
        create_release(version, args.models)
        
    elif args.command == "list":
        registry = load_model_registry()
        if not registry:
            print("No models registered")
        else:
            print("Registered models:")
            for model_name, info in registry.items():
                print(f"- {model_name}: v{info['version']} ({info['release_date']})")
                if info.get("history"):
                    print("  History:")
                    for hist in info["history"]:
                        print(f"  - v{hist['version']} ({hist['release_date']})")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
