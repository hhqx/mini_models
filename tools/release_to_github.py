"""
将模型发布到GitHub Releases的工具脚本
"""
import os
import sys
import argparse
import subprocess
import json
import yaml
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_info(info_path: str) -> Dict[str, Any]:
    """
    加载模型信息
    
    Args:
        info_path: 模型信息文件路径（JSON或YAML）
        
    Returns:
        模型信息字典
    """
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Model info file not found: {info_path}")
    
    # 根据扩展名确定加载方式
    if info_path.endswith('.yaml') or info_path.endswith('.yml'):
        with open(info_path, 'r') as f:
            return yaml.safe_load(f)
    elif info_path.endswith('.json'):
        with open(info_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {info_path}")

def get_files_to_upload(release_dir: str, model_info: Dict[str, Any]) -> List[str]:
    """
    获取需要上传的文件列表
    
    Args:
        release_dir: 发布目录
        model_info: 模型信息
        
    Returns:
        文件路径列表
    """
    files = []
    for model_name, info in model_info.items():
        file_path = os.path.join(release_dir, info['url'])
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    return files

def create_github_release(
    version: str, 
    files: List[str], 
    notes_path: str = None,
    repo: str = None,
    dry_run: bool = False
) -> bool:
    """
    创建GitHub Release
    
    Args:
        version: 发布版本
        files: 文件列表
        notes_path: 发布说明路径
        repo: GitHub仓库，格式为"用户名/仓库名"
        dry_run: 是否仅输出命令而不执行
        
    Returns:
        是否成功
    """
    # 确保已经安装了GitHub CLI
    try:
        subprocess.run(['gh', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("GitHub CLI (gh) not installed or not working. Please install it first.")
        logger.error("Installation guide: https://github.com/cli/cli#installation")
        return False
    
    # 检查是否已登录
    try:
        subprocess.run(['gh', 'auth', 'status'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error("Not logged into GitHub. Please run 'gh auth login' first.")
        return False
    
    # 构建发布命令
    cmd = ['gh', 'release', 'create', version]
    
    # 添加发布说明
    if notes_path and os.path.exists(notes_path):
        cmd.extend(['--notes-file', notes_path])
    else:
        cmd.extend(['--notes', f'Release {version}'])
    
    # 添加仓库
    if repo:
        cmd.extend(['--repo', repo])
    
    # 添加文件
    for file_path in files:
        cmd.append(file_path)
    
    # 执行或输出命令
    logger.info(f"Release command: {' '.join(cmd)}")
    
    if dry_run:
        logger.info("Dry run mode - command not executed")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Release created successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create release: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将模型发布到GitHub Releases")
    parser.add_argument("--release-dir", type=str, default="./release", help="发布文件目录")
    parser.add_argument("--info-file", type=str, help="模型信息文件路径，如model_info.yaml")
    parser.add_argument("--version", type=str, default="v0.1", help="发布版本")
    parser.add_argument("--notes", type=str, help="发布说明文件路径")
    parser.add_argument("--repo", type=str, help="GitHub仓库，格式为'用户名/仓库名'")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际执行")
    
    args = parser.parse_args()
    
    # 查找模型信息文件
    info_file = args.info_file
    if not info_file:
        candidates = [
            os.path.join(args.release_dir, "model_info.yaml"),
            os.path.join(args.release_dir, "model_info.yml"),
            os.path.join(args.release_dir, "model_info.json")
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                info_file = candidate
                break
        if not info_file:
            logger.error("Model info file not found. Please specify using --info-file")
            return 1
    
    # 加载模型信息
    try:
        model_info = load_model_info(info_file)
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        return 1
    
    # 查找发布说明
    notes_path = args.notes
    if not notes_path:
        default_notes = os.path.join(args.release_dir, f"release_notes_{args.version}.md")
        if os.path.exists(default_notes):
            notes_path = default_notes
    
    # 获取要上传的文件
    files = get_files_to_upload(args.release_dir, model_info)
    if not files:
        logger.error("No files to upload!")
        return 1
    
    # 创建发布
    success = create_github_release(
        version=args.version,
        files=files,
        notes_path=notes_path,
        repo=args.repo,
        dry_run=args.dry_run
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
