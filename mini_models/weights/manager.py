"""
模型权重管理器 - 提供更好的版本控制和缓存管理
"""
import os
import json
import shutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from mini_models.config import config
from mini_models.weights.registry import get_model_info, MODEL_REGISTRY
from mini_models.weights.downloader import download_file, check_file_hash

class WeightManager:
    """模型权重管理器，提供更好的权重管理体验"""
    
    def __init__(self):
        self.weights_dir = config.get("weights_dir")
        self.cache_dir = os.path.join(self.weights_dir, "cache")
        self.meta_file = os.path.join(self.weights_dir, "weight_meta.json")
        self.user_weights_dir = os.path.expanduser("~/mini_models_weights")
        
        # 确保目录存在
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.user_weights_dir, exist_ok=True)
        
        # 加载元数据
        self.meta_data = self._load_meta_data()
    
    def _load_meta_data(self) -> Dict[str, Any]:
        """加载权重元数据信息"""
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        # 如果加载失败或文件不存在，初始化一个新的
        return {"models": {}, "last_update_check": None}
    
    def _save_meta_data(self):
        """保存权重元数据信息"""
        with open(self.meta_file, 'w') as f:
            json.dump(self.meta_data, f, indent=2)
    
    def get_weight_path(self, model_name: str, version: str = "latest", 
                         prefer_user: bool = True) -> Optional[str]:
        """
        获取模型权重的路径
        
        Args:
            model_name: 模型名称
            version: 权重版本，默认为最新版本
            prefer_user: 是否优先使用用户自定义权重
            
        Returns:
            str: 权重文件路径，如果不存在则返回None
        """
        # 1. 检查用户目录中是否有此模型权重
        if prefer_user:
            user_path = os.path.join(self.user_weights_dir, f"{model_name}.pth")
            if os.path.exists(user_path):
                print(f"使用用户自定义权重: {user_path}")
                return user_path
        
        # 2. 检查缓存中是否有指定版本权重
        model_meta = self.meta_data["models"].get(model_name, {})
        versions = model_meta.get("versions", [])
        
        if not versions:
            return None
        
        if version == "latest":
            version_info = versions[-1]  # 最新版本
        else:
            # 查找指定版本
            version_info = None
            for v in versions:
                if v["version"] == version:
                    version_info = v
                    break
            if not version_info:
                return None
        
        # 获取缓存路径
        cache_path = os.path.join(self.cache_dir, 
                                  f"{model_name}_{version_info['version']}.pth")
        
        if os.path.exists(cache_path) and check_file_hash(
                cache_path, version_info.get("sha256", "")):
            return cache_path
            
        return None
    
    def download_weight(self, model_name: str, version: str = "latest", 
                       force: bool = False) -> Optional[str]:
        """
        下载模型权重
        
        Args:
            model_name: 模型名称
            version: 权重版本，默认为最新版本
            force: 是否强制重新下载
            
        Returns:
            str: 下载后的权重文件路径，如果下载失败则返回None
        """
        # 获取模型信息
        model_info = get_model_info(model_name)
        if not model_info:
            print(f"错误：未知模型 {model_name}")
            return None
        
        # 确定版本号
        model_version = model_info.get("version", "v0.1.0")
        if version != "latest":
            model_version = version
        
        # 构建缓存路径
        cache_path = os.path.join(self.cache_dir, f"{model_name}_{model_version}.pth")
        
        # 检查是否已存在且有效
        if os.path.exists(cache_path) and not force:
            if check_file_hash(cache_path, model_info.get("sha256", "")):
                print(f"模型 {model_name} ({model_version}) 已存在，跳过下载")
                self._update_meta_for_model(model_name, model_version, model_info)
                return cache_path
            else:
                print(f"模型 {model_name} 哈希值不匹配，重新下载")
        
        # 构建下载URL
        base_url = config.get("download_server")
        url = f"{base_url}{model_info['url']}"
        
        # 下载权重
        print(f"下载模型权重 {model_name} ({model_version})...")
        if download_file(url, cache_path):
            if check_file_hash(cache_path, model_info.get("sha256", "")):
                print(f"模型 {model_name} ({model_version}) 下载完成")
                # 更新元数据
                self._update_meta_for_model(model_name, model_version, model_info)
                return cache_path
            else:
                print(f"模型 {model_name} 哈希值校验失败")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        
        return None
    
    def _update_meta_for_model(self, model_name: str, version: str, model_info: Dict[str, Any]):
        """更新模型的元数据信息"""
        if model_name not in self.meta_data["models"]:
            self.meta_data["models"][model_name] = {"versions": []}
        
        # 检查此版本是否已存在
        versions = self.meta_data["models"][model_name]["versions"]
        version_exists = False
        for v in versions:
            if v["version"] == version:
                version_exists = True
                # 更新信息
                v["last_used"] = datetime.now().isoformat()
                v["sha256"] = model_info.get("sha256", "")
                break
        
        # 如果版本不存在，添加它
        if not version_exists:
            versions.append({
                "version": version,
                "downloaded_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "sha256": model_info.get("sha256", ""),
                "url": model_info.get("url", ""),
                "size": model_info.get("size", 0)
            })
        
        # 保存元数据
        self._save_meta_data()
    
    def import_user_weight(self, model_name: str, weight_path: str) -> bool:
        """
        导入用户自定义权重
        
        Args:
            model_name: 模型名称
            weight_path: 权重文件路径
            
        Returns:
            bool: 导入是否成功
        """
        if not os.path.exists(weight_path):
            print(f"错误：权重文件不存在 {weight_path}")
            return False
        
        # 目标路径
        target_path = os.path.join(self.user_weights_dir, f"{model_name}.pth")
        
        try:
            # 复制文件
            shutil.copy2(weight_path, target_path)
            print(f"已导入用户权重: {target_path}")
            return True
        except Exception as e:
            print(f"导入权重失败: {e}")
            return False
    
    def remove_weight(self, model_name: str, version: str = None, 
                     remove_all: bool = False, remove_user: bool = False) -> bool:
        """
        删除模型权重
        
        Args:
            model_name: 模型名称
            version: 要删除的特定版本，如果为None则按照其他参数决定
            remove_all: 是否删除所有缓存版本
            remove_user: 是否删除用户自定义权重
            
        Returns:
            bool: 是否成功删除
        """
        success = True
        
        # 删除用户自定义权重
        if remove_user:
            user_path = os.path.join(self.user_weights_dir, f"{model_name}.pth")
            if os.path.exists(user_path):
                try:
                    os.remove(user_path)
                    print(f"已删除用户权重: {user_path}")
                except Exception as e:
                    print(f"删除用户权重失败: {e}")
                    success = False
        
        # 删除缓存权重
        if remove_all:
            # 删除所有版本
            for file in os.listdir(self.cache_dir):
                if file.startswith(f"{model_name}_") and file.endswith(".pth"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                        print(f"已删除权重缓存: {file}")
                    except Exception as e:
                        print(f"删除权重缓存失败: {e}")
                        success = False
            
            # 更新元数据
            if model_name in self.meta_data["models"]:
                del self.meta_data["models"][model_name]
                self._save_meta_data()
        
        elif version:
            # 删除特定版本
            cache_path = os.path.join(self.cache_dir, f"{model_name}_{version}.pth")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    print(f"已删除权重缓存: {model_name}_{version}.pth")
                    
                    # 更新元数据
                    if model_name in self.meta_data["models"]:
                        versions = self.meta_data["models"][model_name]["versions"]
                        self.meta_data["models"][model_name]["versions"] = [
                            v for v in versions if v["version"] != version
                        ]
                        self._save_meta_data()
                except Exception as e:
                    print(f"删除权重缓存失败: {e}")
                    success = False
        
        return success
    
    def check_for_updates(self, model_name: Optional[str] = None) -> Dict[str, bool]:
        """
        检查模型权重是否有更新
        
        Args:
            model_name: 要检查的模型名称，如果为None则检查所有模型
            
        Returns:
            Dict[str, bool]: 模型名称到是否有更新的映射
        """
        # 记录上次检查时间
        self.meta_data["last_update_check"] = datetime.now().isoformat()
        
        updates = {}
        models_to_check = [model_name] if model_name else MODEL_REGISTRY.keys()
        
        for name in models_to_check:
            model_info = get_model_info(name)
            if not model_info:
                continue
            
            # 获取注册表中的最新版本
            registry_version = model_info.get("version", "v0.1.0")
            
            # 获取本地最新版本
            local_version = None
            if name in self.meta_data["models"] and self.meta_data["models"][name]["versions"]:
                local_version = self.meta_data["models"][name]["versions"][-1]["version"]
            
            # 检查是否有更新
            has_update = local_version is None or local_version != registry_version
            updates[name] = has_update
            
            if has_update:
                print(f"检测到模型 {name} 有新版本: {registry_version}")
        
        self._save_meta_data()
        return updates
    
    def list_weight_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        列出模型的所有可用版本
        
        Args:
            model_name: 模型名称
            
        Returns:
            List[Dict[str, Any]]: 版本信息列表
        """
        if model_name not in self.meta_data["models"]:
            return []
        
        return self.meta_data["models"][model_name]["versions"]
    
    def list_user_weights(self) -> List[str]:
        """
        列出所有用户自定义权重
        
        Returns:
            List[str]: 用户自定义权重的模型名称列表
        """
        user_weights = []
        for file in os.listdir(self.user_weights_dir):
            if file.endswith(".pth"):
                user_weights.append(file[:-4])  # 去除.pth后缀
        return user_weights
    
    def get_weight_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型权重的详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 权重信息
        """
        info = {
            "model_name": model_name,
            "has_user_weight": False,
            "cached_versions": [],
            "latest_version": None,
            "registry_info": get_model_info(model_name)
        }
        
        # 检查用户权重
        user_path = os.path.join(self.user_weights_dir, f"{model_name}.pth")
        info["has_user_weight"] = os.path.exists(user_path)
        
        # 获取缓存版本
        if model_name in self.meta_data["models"]:
            info["cached_versions"] = self.meta_data["models"][model_name]["versions"]
            
            if info["cached_versions"]:
                info["latest_version"] = info["cached_versions"][-1]["version"]
        
        return info

# 创建全局权重管理器实例
weight_manager = WeightManager()
