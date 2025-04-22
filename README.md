# mini_models

A lightweight deep learning model library for resource-constrained environments.

## Features

- Lightweight model implementations optimized for single GPU training and inference
- Easy-to-use training, evaluation, and deployment pipelines
- Built-in model quantization and precision testing tools
- Automatic model weights management and downloading
- Suitable for learning and experimentation in deep learning

## Installation

```bash
pip install mini_models
```

## Quick Start

### Training a Model

```python
from mini_models.models import get_model
from mini_models.train import Trainer
from mini_models.datasets import get_mnist_dataloaders

# Load data
dataloaders = get_mnist_dataloaders(batch_size=64)

# Create model
model = get_model("mnist_cnn", pretrained=False)

# Create trainer and train
trainer = Trainer(model=model, train_loader=dataloaders["train"])
trainer.train()
```

### Model Inference

```python
from mini_models.models import get_model
from mini_models.evaluation import Evaluator

# Load pretrained model
model = get_model("mnist_cnn", pretrained=True)

# Evaluate model
evaluator = Evaluator(model, test_loader)
results = evaluator.evaluate()
```

### Model Deployment

```python
from mini_models.deployment import quantize_model, evaluate_precision

# Quantize model
quantized_model = quantize_model(model)

# Compare precision
metrics = evaluate_precision(model, quantized_model, test_loader)
```

## Supported Models

Currently supported models include:

- Vision Models:
  - MNIST-CNN
  - ResNet18-Mini
  - DiT (Diffusion Transformer)
- More models coming soon...

## Development

To set up the development environment:

```bash
git clone https://github.com/hhqx/mini_models.git
cd mini_models
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```
