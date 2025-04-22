# 开发指南

本文档介绍如何参与 Mini Models 项目的开发。

## 开发环境设置

1. 克隆仓库：

```bash
git clone https://github.com/hhqx/mini_models.git
cd mini_models
```

2. 创建并激活虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

## 项目结构

```
mini_models/
├── models/         # 模型实现
├── tools/          # 工具脚本
│   ├── prepare_release.py    # 准备模型发布
│   ├── release_to_github.py  # 发布到GitHub
│   └── version_manager.py    # 版本管理
├── tests/          # 测试代码
├── docs/           # 文档
└── release/        # 发布文件
```

## 开发新模型

1. 在适当的目录下创建新的模型实现
2. 确保模型符合项目的接口规范
3. 添加必要的测试
4. 更新模型注册表

## 代码风格

本项目遵循PEP 8代码风格指南。提交代码前，请确保运行以下命令检查代码格式：

```bash
flake8 .
```

## 提交代码

1. 创建新分支：

```bash
git checkout -b feature/your-feature-name
```

2. 提交代码：

```bash
git add .
git commit -m "Add your descriptive commit message"
git push origin feature/your-feature-name
```

3. 创建Pull Request
