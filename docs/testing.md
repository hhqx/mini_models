# 测试指南

本文档介绍如何测试 Mini Models 项目。

## 单元测试

运行项目的单元测试：

```bash
pytest tests/
```

运行特定测试文件：

```bash
pytest tests/test_specific_module.py
```

## 测试覆盖率

检查测试覆盖率：

```bash
pytest --cov=. tests/
```

生成HTML覆盖率报告：

```bash
pytest --cov=. --cov-report=html tests/
```

## 模型测试

测试特定模型：

1. 加载模型
2. 使用示例输入进行推理
3. 验证输出结果

示例：

```python
# 测试图像分类模型
from models import load_model

model = load_model("image_classifier_v1")
result = model.predict("path/to/test/image.jpg")
assert result["category"] in ["cat", "dog", "other"]
```

## 集成测试

测试模型发布流程：

```bash
# 创建测试配置
python tools/prepare_release.py --batch tests/test_batch_config.yaml --output-dir ./test_release --generate-notes

# 验证生成的文件
ls -la ./test_release
```
