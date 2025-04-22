"""
模型权重下载管理
"""
import os
import time
import hashlib
import requests
from tqdm import tqdm
from typing import Optional

from mini_models.config import config
from mini_models.weights.registry import get_model_info

def download_file(url: str, filepath: str, proxies: Optional[dict] = None) -> bool:
    """
    下载文件到指定路径，显示进度条

    Args:
        url (str): 文件的下载链接
        filepath (str): 文件保存的本地路径
        proxies (Optional[dict]): 可选的代理配置，例如 {"http": "http://proxy.com:8080", "https": "http://proxy.com:8080"}

    Returns:
        bool: 下载成功返回 True，失败返回 False

    Example:
        >>> proxies = {"http": "http://proxy.com:8080", "https": "http://proxy.com:8080"}
        >>> download_file("http://example.com/file", "/path/to/save/file", proxies=proxies)
        True
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 设置临时文件路径
        tmp_filepath = f"{filepath}.tmp"
        
        # 发起请求
        response = requests.get(url, stream=True, proxies=proxies)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 下载文件并显示进度条
        with open(tmp_filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        # 下载完成后重命名
        os.rename(tmp_filepath, filepath)
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        # 清理可能的临时文件
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        return False

def check_file_hash(filepath: str, expected_hash: str) -> bool:
    """
    验证文件哈希值
    """
    if not expected_hash:
        return True  # 没有提供哈希值，跳过验证
        
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash
    except Exception as e:
        print(f"哈希校验失败: {e}")
        return False

def download_if_needed(model_name: str, force: bool = False) -> str:
    """
    如果模型权重不存在，则下载
    
    Args:
        model_name: 模型名称
        force: 是否强制重新下载
        
    Returns:
        str: 权重文件的本地路径
    """
    # 获取模型信息
    model_info = get_model_info(model_name)
    if not model_info:
        raise ValueError(f"未知模型: {model_name}")
    
    # 构建本地文件路径
    weights_dir = config["weights_dir"]
    local_path = os.path.join(weights_dir, model_name.replace("/", "_") + ".pth")
    
    # 检查文件是否已存在且有效
    if os.path.exists(local_path) and not force:
        # 验证哈希值
        if check_file_hash(local_path, model_info.get("sha256", "")):
            return local_path
        else:
            print(f"模型文件 {model_name} 哈希值不匹配，重新下载...")
    
    # 构建下载URL
    base_url = config.get("download_server")
    url = f"{base_url}{model_info['url']}"
    
    # 下载文件
    print(f"下载模型权重 {model_name}...")
    if download_file(url, local_path):
        # 校验文件
        if check_file_hash(local_path, model_info.get("sha256", "")):
            print(f"模型 {model_name} 下载完成")
            return local_path
        else:
            os.remove(local_path)
            raise ValueError(f"模型 {model_name} 哈希值校验失败")
    else:
        raise RuntimeError(f"下载模型 {model_name} 失败")

def download_weights(model_name: str, force: bool = False) -> str:
    """
    公开API：下载模型权重
    
    Args:
        model_name: 模型名称
        force: 是否强制重新下载
        
    Returns:
        str: 权重文件的本地路径
    """
    return download_if_needed(model_name, force)
