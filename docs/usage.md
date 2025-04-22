# 使用指南

本文档介绍如何使用 Mini Models 项目。

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/hhqx/mini_models.git
cd mini_models
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 模型管理

Mini Models 项目支持多种模型类别：

- 视觉模型（vision）：图像分类、目标检测等
- 自然语言处理模型（nlp）：文本分类、翻译、文本生成等
- 音频模型（audio）：音频分类、语音识别等
- 其他模型（misc）：其他类型的模型

### 查看模型列表

使用版本管理工具查看所有注册模型：

```bash
python tools/version_manager.py list
```

### 获取项目版本

```bash
python tools/version_manager.py get-version
```

### 使用模型

具体模型的使用方法请参考每个模型的文档。
