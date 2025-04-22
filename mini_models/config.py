"""
全局配置管理模块
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 默认配置
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/mini_models")
DEFAULT_CONFIG = {
    "cache_dir": DEFAULT_CACHE_DIR,
    "weights_dir": os.path.join(DEFAULT_CACHE_DIR, "weights"),
    "datasets_dir": os.path.join(DEFAULT_CACHE_DIR, "datasets"),
    "download_server": "https://github.com/username/mini_models/releases/download/",
    "use_cuda": True,
    "precision": "fp32",
    "log_level": "INFO",
}

class Config:
    """全局配置管理类"""
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化配置"""
        self._config = DEFAULT_CONFIG.copy()
        
        # 创建必要的目录
        os.makedirs(self._config["weights_dir"], exist_ok=True)
        os.makedirs(self._config["datasets_dir"], exist_ok=True)
        
        # 加载用户配置（如果存在）
        user_config_path = os.path.expanduser("~/.mini_models.yaml")
        if os.path.exists(user_config_path):
            self.load_from_file(user_config_path)
    
    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)
    
    def __setitem__(self, key: str, value: Any):
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def update(self, config_dict: Dict[str, Any]):
        """更新配置"""
        self._config.update(config_dict)
    
    def load_from_file(self, config_path: str):
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self.update(user_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: Optional[str] = None):
        """保存配置到YAML文件"""
        if config_path is None:
            config_path = os.path.expanduser("~/.mini_models.yaml")
        
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(self._config, f)
        except Exception as e:
            print(f"Warning: Failed to save config to {config_path}: {e}")

# 全局配置实例
config = Config()
