"""
mini_models: 轻量级深度学习模型库，为资源受限环境提供高效训练和部署方案。
"""

from mini_models.version import __version__

# 便捷导入
from mini_models.models import get_model
from mini_models.weights import download_weights
from mini_models.train import Trainer
from mini_models.evaluation import Evaluator
from mini_models.deployment import quantize_model, evaluate_precision

# 设置日志
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 初始化权重目录
from mini_models.weights.registry import init_weight_registry
init_weight_registry()

# 输出版本信息
import sys
if not sys.argv[0].endswith("sphinx-build"):
    print(f"mini_models version: {__version__}")
