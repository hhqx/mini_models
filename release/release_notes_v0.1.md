# Mini Models Release v0.1

## Image Generation Models

### dit_mnist

- **Description**: MNIST图像生成的DiT模型
- **Size**: 1.51MB
- **SHA256**: d7c1c77350694dac0854297860d350f37c1db85ca569565159965eab301d6ae9

```python
from mini_models.models import get_model

# Load the model
model = get_model("dit_mnist", pretrained=True)
```

