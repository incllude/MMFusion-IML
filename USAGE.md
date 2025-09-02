# MMFusion-IML Usage Guide

This guide shows how to use MMFusion-IML as a Python library in your own projects.

## Installation

### From Source
```bash
git clone <your-repo-url>
cd MMFusion-IML
pip install -e .
```

### From PyPI (if published)
```bash
pip install mmfusion-iml
```

## Quick Start

### Basic Model Usage

```python
import mmfusion_iml as mmf

# Load a pre-trained model
model = mmf.models.CMNeXtConf(config)

# Or use the base model
base_model = mmf.models.BaseModel(backbone='MiT-B0', num_classes=2, modals=['rgb'])

# Load datasets
dataset = mmf.data.ManipulationDataset(
    path='path/to/dataset.txt',
    image_size=512,
    train=True
)

# Mix multiple datasets
mix_dataset = mmf.data.MixDataset(
    paths=['dataset1.txt', 'dataset2.txt'],
    image_size=512,
    train=True
)
```

### Advanced Usage

```python
import torch
import mmfusion_iml as mmf

# Initialize utilities
srm_filter = mmf.common.SRMFilter()
bayar_conv = mmf.common.BayarConv2d(3, 3, padding=2)

# Use modal extractor
modal_extractor = mmf.models.ModalExtract(
    modals=['noiseprint', 'bayar', 'srm'],
    noiseprint_path='path/to/noiseprint/weights.pth'
)

# Extract modalities from image
image = torch.randn(1, 3, 512, 512)
modalities = modal_extractor(image)

# Use with backbone models
backbone_model = mmf.models.backbones.CMNeXtMHSA('B2', ['rgb', 'noiseprint'])
features = backbone_model([image, modalities[0]])
```

### Configuration

```python
# Create a simple config-like object
class Config:
    BACKBONE = 'CMNeXtMHSA-B2'
    NUM_CLASSES = 2
    MODALS = ['rgb', 'noiseprint']
    TRAIN_PHASE = 'localization'
    DETECTION = 'confpool'
    PRETRAINED = 'path/to/pretrained/weights.pth'

config = Config()
model = mmf.models.CMNeXtConf(config)
```

### Training Loop Example

```python
import torch
import torch.nn as nn
import mmfusion_iml as mmf

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mmf.models.CMNeXtConf(config).to(device)

# Create dataset
dataset = mmf.data.MixDataset(
    paths=['train_dataset1.txt', 'train_dataset2.txt'],
    image_size=512,
    train=True
)

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True,
    num_workers=4
)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):
    for batch_idx, (images, _, masks, labels) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle localization task
        if isinstance(outputs, tuple):
            # Detection phase - has confidence output
            out, conf, det = outputs
            loss = criterion(out.view(-1, 2), masks.view(-1))
        else:
            # Localization phase
            loss = criterion(outputs.view(-1, 2), masks.view(-1))
            
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Inference Example

```python
import torch
import cv2
import numpy as np
import mmfusion_iml as mmf

# Load model
model = mmf.models.CMNeXtConf(config)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Load and preprocess image
image_path = 'test_image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# Run inference
with torch.no_grad():
    output = model([image])
    
    if isinstance(output, tuple):
        # Detection phase
        localization, confidence, detection = output
        print(f"Detection probability: {torch.sigmoid(detection).item():.4f}")
    else:
        # Localization only
        localization = output
    
    # Get manipulation mask
    mask = torch.softmax(localization, dim=1)[:, 1, :, :]  # Get manipulation channel
    mask = mask.squeeze().cpu().numpy()
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='hot')
    plt.title('Manipulation Mask')
    plt.show()
```

## Available Models

- `CMNeXtConf`: Main model with confidence estimation
- `WSCMNeXtConf`: Weakly supervised variant
- `BaseModel`: Base model class for custom implementations
- `DnCNN` / `DnCNNNoiseprint`: Noise print extraction model
- `ModalExtract`: Multi-modal feature extractor

## Available Utilities

- `SRMFilter`: Spatial Rich Model filter
- `BayarConv2d`: Bayar constrained convolution
- `AverageMeter`: Training metrics utilities
- `ManipulationDataset`: Single dataset loader
- `MixDataset`: Multi-dataset loader

## Command Line Tools

After installation, you can use command-line tools:

```bash
# Training
mmfusion-train --exp experiments/config.yaml --phase localization

# Testing
mmfusion-test --exp experiments/config.yaml --ckpt model.pth --task localization --manip test_list.txt

# Inference
mmfusion-inference --exp experiments/config.yaml --ckpt model.pth --input image.jpg --output results/
```

## Configuration Files

See the `experiments/` directory for example configuration files:
- `ec_example.yaml`: Localization training configuration
- `ec_example_phase2.yaml`: Detection training configuration

## Requirements

See `requirements.txt` for the full list of dependencies. Main requirements:
- PyTorch >= 1.8
- torchvision
- OpenCV
- albumentations
- numpy
- matplotlib (for visualization)
