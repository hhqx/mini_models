# 维护指南

本文档介绍 Mini Models 项目的维护流程。

## 依赖管理

定期检查和更新项目依赖：

```bash
pip list --outdated
```

更新依赖后，更新requirements.txt文件：

```bash
pip freeze > requirements.txt
```

## 模型注册表管理

模型注册表存储在项目中，用于跟踪所有可用模型及其版本。

### 查看模型注册表

```bash
python tools/version_manager.py list
```

### 更新模型版本

当模型有更新时，使用以下命令更新版本：

```bash
python tools/version_manager.py update-model <model-name> --version <new-version> --type [major|minor|patch]
```

如果有配置文件需要同步更新：

```bash
python tools/version_manager.py update-model <model-name> --version <new-version> --config path/to/config.yaml
```

## 版本号规范

本项目使用语义化版本控制(Semantic Versioning)：

- **主版本号(major)**：当进行不兼容的API更改
- **次版本号(minor)**：当添加向后兼容的功能
- **补丁版本号(patch)**：当进行向后兼容的bug修复

## 问题追踪

使用GitHub Issues跟踪项目问题和功能请求。
