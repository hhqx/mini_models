# 发布流程

本文档详细介绍 Mini Models 项目的版本控制和发布流程。

## 版本管理

### 查看当前版本

```bash
python tools/version_manager.py get-version
```

### 升级项目版本

```bash
python tools/version_manager.py bump [major|minor|patch|pre] [--pre alpha.1]
```

示例：
```bash
# 升级补丁版本
python tools/version_manager.py bump patch

# 升级次版本
python tools/version_manager.py bump minor

# 创建预发布版本
python tools/version_manager.py bump pre --pre alpha.1
```

## 准备发布

### 单个模型发布

```bash
python tools/prepare_release.py \
  --model-path path/to/model.pth \
  --model-name my_model \
  --task-type image_classification \
  --description "这是一个示例模型" \
  --output-dir ./release \
  --version v0.1 \
  --category vision \
  --generate-notes
```

### 批量模型发布

创建批量配置文件 `batch_config.yaml`：

```yaml
models:
  - name: model1
    path: path/to/model1.pth
    task: image_classification
    description: "这是模型1的描述"
    version: v0.1
    category: vision
  
  - name: model2
    path: path/to/model2.pth
    task: text_classification
    description: "这是模型2的描述"
    version: v0.1
    category: nlp
```

使用批量配置文件准备发布：

```bash
python tools/prepare_release.py \
  --batch batch_config.yaml \
  --output-dir ./release \
  --version v0.1 \
  --generate-notes
```

## 发布到GitHub

确保已安装和登录GitHub CLI：

```bash
# 安装GitHub CLI（仅需一次）
# macOS
brew install gh
# Linux
sudo apt install gh
# Windows
winget install --id GitHub.cli

# 登录GitHub
gh auth login
```

将准备好的发布文件上传到GitHub：

```bash
python tools/release_to_github.py \
  --release-dir ./release \
  --version v0.1 \
  --repo "hhqx/mini_models" \
  --notes ./release/release_notes_v0.1.md
```

测试发布过程（不实际执行）：

```bash
python tools/release_to_github.py \
  --release-dir ./release \
  --version v0.1 \
  --repo "hhqx/mini_models" \
  --dry-run
```

## 完整发布流程示例

```bash
# 1. 升级项目版本
python tools/version_manager.py bump minor

# 2. 准备模型发布文件
python tools/prepare_release.py --batch tools/batch_config.yaml --output-dir ./release --generate-notes

# 3. 发布到GitHub
python tools/release_to_github.py --version v0.1 --repo "hhqx/mini_models"
```
