
```bash
# 使用YAML配置生成发布文件
python tools/prepare_release.py --batch tools/batch_config_example.yaml --output-dir ./release --generate-notes

# 将发布文件上传到GitHub
python tools/release_to_github.py --version v0.1 --repo "hhqx/mini_models"
```




