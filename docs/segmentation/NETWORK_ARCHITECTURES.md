# Network Architectures for Medical Image Segmentation

This document provides a comprehensive overview of all segmentation network architectures available in the framework.

## Table of Contents

1. [Overview](#overview)
2. [CNN-Based Architectures](#cnn-based-architectures)
3. [Transformer-Based Architectures](#transformer-based-architectures)
4. [Hybrid Architectures](#hybrid-architectures)
5. [Architecture Comparison](#architecture-comparison)
6. [Selection Guide](#selection-guide)

---

## Overview

The framework provides 8 state-of-the-art segmentation architectures, each designed for different use cases in medical imaging:

| Architecture | Type | Dimensionality | Best For |
|-------------|------|----------------|----------|
| U-Net | CNN | 2D/3D | General purpose, baseline |
| V-Net | CNN | 3D | Volumetric data, residual learning |
| SegResNet | CNN | 2D/3D | Efficient segmentation, limited resources |
| DeepLabV3+ | CNN | 2D | Multi-scale features, varied object sizes |
| TransUNet | Hybrid | 2D | Global context + local details |
| UNETR | Transformer | 3D | Long-range dependencies, 3D volumes |
| Swin-UNETR | Transformer | 3D | Hierarchical features, state-of-the-art |

---

## CNN-Based Architectures

### 1. U-Net (2D/3D)

**Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

**Description**: The foundational architecture for medical image segmentation with encoder-decoder structure and skip connections.

**Architecture**:
- **Encoder**: 4 downsampling blocks with max pooling
- **Decoder**: 4 upsampling blocks with transposed convolutions
- **Skip Connections**: Direct concatenation between encoder and decoder
- **Channels**: Progressive doubling (64 → 128 → 256 → 512)

**Key Features**:
- Simple and effective
- Skip connections preserve spatial information
- Works well with small datasets
- Fast training and inference

**Configuration**:
```json
{
  "type": "UNet2D",
  "config": {
    "in_channels": "1",
    "out_channels": "2",
    "base_channels": "64"
  }
}
```

**Use Cases**:
- General medical image segmentation
- Baseline model for comparison
- Limited training data scenarios
- 2D slice-based segmentation

**Pros**:
- Simple architecture
- Good performance on most tasks
- Fast training
- Low memory requirements

**Cons**:
- Limited receptive field
- No multi-scale feature aggregation
- May miss global context

---

### 2. V-Net

**Paper**: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

**Description**: 3D extension of U-Net with residual connections and PReLU activation.

**Architecture**:
- **Encoder**: 4 downsampling stages with stride-2 convolutions
- **Decoder**: 4 upsampling stages with transposed convolutions
- **Residual Connections**: Within each convolutional block
- **Skip Connections**: Between encoder and decoder
- **Activation**: PReLU (learnable activation)

**Key Features**:
- Residual learning for easier optimization
- 5×5×5 convolutions for larger receptive field
- PReLU activation learns negative slopes
- Designed specifically for 3D volumetric data

**Configuration**:
```json
{
  "type": "VNet",
  "config": {
    "in_channels": "1",
    "out_channels": "2",
    "base_channels": "16"
  }
}
```

**Use Cases**:
- 3D medical volume segmentation (CT, MRI)
- Organ segmentation
- Tumor detection in 3D
- Volumetric analysis

**Pros**:
- Residual learning improves convergence
- Effective for 3D data
- Good for volumetric context

**Cons**:
- Higher memory requirements (3D convolutions)
- Slower than 2D counterparts
- Requires more computational resources

---

### 3. SegResNet

**Paper**: [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/abs/1810.11654)

**Description**: Efficient segmentation network using residual blocks and group normalization.

**Architecture**:
- **Encoder**: 4 downsampling blocks with residual learning
- **Decoder**: 3 upsampling blocks
- **Skip Connections**: Addition-based (not concatenation)
- **Normalization**: Group normalization
- **Blocks**: 1-2-2-4 residual blocks per stage

**Key Features**:
- Efficient design with fewer parameters
- Addition-based skip connections save memory
- Group normalization (more stable than batch norm for small batches)
- Configurable for 2D or 3D

**Configuration**:
```json
{
  "type": "SegResNet",
  "config": {
    "in_channels": "1",
    "out_channels": "2",
    "init_filters": "8",
    "dimension": "3d"
  }
}
```

**Use Cases**:
- Limited GPU memory scenarios
- Brain tumor segmentation
- Real-time applications
- Resource-constrained environments

**Pros**:
- Memory efficient
- Fast inference
- Good performance-to-parameter ratio
- Stable training with group norm

**Cons**:
- May underperform on complex tasks
- Limited capacity compared to larger models

---

### 4. DeepLabV3+

**Paper**: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

**Description**: Multi-scale segmentation using Atrous Spatial Pyramid Pooling (ASPP).

**Architecture**:
- **Backbone**: ResNet-like encoder
- **ASPP Module**: Parallel atrous convolutions with rates [6, 12, 18]
- **Decoder**: Lightweight decoder with low-level feature fusion
- **Multi-scale**: Captures features at multiple scales

**Key Features**:
- ASPP captures multi-scale context
- Atrous (dilated) convolutions increase receptive field
- Low-level feature fusion in decoder
- Effective for objects of varying sizes

**Configuration**:
```json
{
  "type": "DeepLabV3Plus",
  "config": {
    "in_channels": "1",
    "num_classes": "2"
  }
}
```

**Use Cases**:
- Objects with varying sizes
- Multi-organ segmentation
- High-resolution 2D images
- Complex anatomical structures

**Pros**:
- Excellent multi-scale feature extraction
- Good for varying object sizes
- Strong performance on 2D images

**Cons**:
- 2D only (not designed for 3D)
- Higher computational cost
- More complex architecture

---

## Transformer-Based Architectures

### 5. UNETR

**Paper**: [UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)

**Description**: Pure Vision Transformer encoder with CNN decoder for 3D segmentation.

**Architecture**:
- **Encoder**: 12-layer Vision Transformer (ViT)
- **Patch Embedding**: 16×16×16 patches
- **Skip Connections**: From transformer layers 3, 6, 9, 12
- **Decoder**: CNN-based with skip connection fusion
- **Attention**: Self-attention on image patches

**Key Features**:
- Pure transformer encoder
- Long-range dependency modeling
- Multi-scale skip connections from different transformer layers
- Designed for 3D medical images

**Configuration**:
```json
{
  "type": "UNETR",
  "config": {
    "img_size": "96",
    "in_channels": "1",
    "out_channels": "2",
    "embed_dim": "768",
    "num_heads": "12",
    "num_layers": "12"
  }
}
```

**Use Cases**:
- 3D medical image segmentation
- Tasks requiring global context
- Large volumetric datasets
- State-of-the-art performance needed

**Pros**:
- Excellent global context modeling
- Effective for large 3D volumes
- State-of-the-art performance

**Cons**:
- High memory requirements
- Slower training
- Needs large datasets
- Computationally expensive

---

### 6. Swin-UNETR

**Paper**: [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors](https://arxiv.org/abs/2201.01266)

**Description**: Hierarchical Swin Transformer with shifted windows for efficient 3D segmentation.

**Architecture**:
- **Encoder**: 4-stage Swin Transformer
- **Shifted Windows**: Local and global attention
- **Patch Merging**: Hierarchical downsampling
- **Decoder**: CNN decoder with skip connections
- **Window Size**: 7×7×7 local attention windows

**Key Features**:
- Hierarchical architecture (like CNNs)
- Shifted window attention for efficiency
- Local and global feature learning
- Current state-of-the-art for many tasks

**Configuration**:
```json
{
  "type": "SwinUNETR",
  "config": {
    "img_size": "96",
    "in_channels": "1",
    "out_channels": "2",
    "embed_dim": "48",
    "window_size": "7"
  }
}
```

**Use Cases**:
- State-of-the-art 3D segmentation
- Brain tumor segmentation
- Organ segmentation in CT/MRI
- Research and competitions

**Pros**:
- Best performance on many benchmarks
- Efficient compared to UNETR
- Hierarchical features
- Strong multi-scale representation

**Cons**:
- Complex architecture
- High computational cost
- Requires large datasets
- Longer training time

---

## Hybrid Architectures

### 7. TransUNet

**Paper**: [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)

**Description**: Hybrid architecture combining CNN encoder with Transformer and U-Net decoder.

**Architecture**:
- **CNN Encoder**: 3-layer CNN for low-level features
- **Transformer**: 12-layer transformer on CNN features
- **Patch Embedding**: Convert CNN features to patches
- **Decoder**: U-Net style with skip connections

**Key Features**:
- Combines CNN local features with transformer global context
- CNN encoder captures fine details
- Transformer models long-range dependencies
- Best of both worlds

**Configuration**:
```json
{
  "type": "TransUNet",
  "config": {
    "img_size": "224",
    "in_channels": "1",
    "out_channels": "2",
    "base_channels": "64",
    "embed_dim": "512",
    "num_heads": "8",
    "num_layers": "12"
  }
}
```

**Use Cases**:
- 2D medical image segmentation
- Balance between local and global features
- Tasks needing both fine details and context
- Moderate dataset sizes

**Pros**:
- Good balance of local and global features
- Better than pure CNN or transformer alone
- Effective for 2D images
- Good performance with moderate data

**Cons**:
- 2D only
- Higher memory than pure CNN
- More complex than U-Net

---

## Architecture Comparison

### Performance Characteristics

| Architecture | Parameters | Memory | Speed | Accuracy | Best Use |
|-------------|-----------|--------|-------|----------|----------|
| U-Net | Low | Low | Fast | Good | Baseline, limited data |
| V-Net | Medium | High | Medium | Good | 3D volumes |
| SegResNet | Low | Low | Fast | Good | Efficiency |
| DeepLabV3+ | High | Medium | Medium | Very Good | Multi-scale 2D |
| TransUNet | High | High | Slow | Very Good | 2D hybrid |
| UNETR | Very High | Very High | Slow | Excellent | 3D transformer |
| Swin-UNETR | Very High | Very High | Slow | Excellent | State-of-the-art 3D |

### Computational Requirements

**Low Resource (< 8GB GPU)**:
- U-Net 2D
- SegResNet with small filters

**Medium Resource (8-16GB GPU)**:
- U-Net 3D
- V-Net with small base channels
- DeepLabV3+
- SegResNet 3D

**High Resource (> 16GB GPU)**:
- TransUNet
- UNETR
- Swin-UNETR
- V-Net with large base channels

---

## Selection Guide

### By Use Case

**General 2D Segmentation**:
1. **Start with**: U-Net 2D
2. **Need multi-scale**: DeepLabV3+
3. **Need global context**: TransUNet

**General 3D Segmentation**:
1. **Start with**: U-Net 3D or SegResNet
2. **Need better performance**: V-Net
3. **State-of-the-art**: UNETR or Swin-UNETR

**Limited Resources**:
1. **2D**: U-Net 2D
2. **3D**: SegResNet

**Maximum Performance** (resources available):
1. **2D**: TransUNet or DeepLabV3+
2. **3D**: Swin-UNETR or UNETR

**Small Dataset** (< 100 cases):
- U-Net (2D/3D)
- V-Net

**Large Dataset** (> 500 cases):
- Transformer-based (UNETR, Swin-UNETR)
- TransUNet

### By Medical Imaging Modality

**CT Scans**:
- V-Net (3D volumes)
- UNETR (large volumes)
- Swin-UNETR (state-of-the-art)

**MRI**:
- U-Net (general)
- TransUNet (2D slices)
- UNETR/Swin-UNETR (3D volumes)

**X-Ray / 2D Images**:
- U-Net 2D
- DeepLabV3+
- TransUNet

**Microscopy**:
- U-Net 2D (excellent for cells)
- DeepLabV3+ (varying sizes)

### By Anatomical Structure

**Small Structures** (lesions, tumors):
- U-Net (fine details)
- TransUNet (global context helps)

**Large Organs**:
- V-Net (3D context)
- Swin-UNETR (hierarchical)

**Multiple Organs**:
- DeepLabV3+ (multi-scale)
- Swin-UNETR (comprehensive)

**Brain**:
- U-Net (baseline)
- Swin-UNETR (state-of-the-art for brain tumors)

---

## Training Recommendations

### U-Net Family (U-Net, V-Net)

**Loss Function**: Dice Loss or Combined (Dice + CE)
**Optimizer**: Adam
**Learning Rate**: 1e-4
**Batch Size**: 4-8 (2D), 1-2 (3D)
**Epochs**: 100-200
**Augmentation**: Essential for small datasets

### SegResNet

**Loss Function**: Dice Loss
**Optimizer**: Adam
**Learning Rate**: 1e-4
**Batch Size**: 2-4 (3D)
**Epochs**: 100-200
**Augmentation**: Standard

### DeepLabV3+

**Loss Function**: Cross-Entropy or Combined
**Optimizer**: SGD with momentum (0.9)
**Learning Rate**: 1e-2 with polynomial decay
**Batch Size**: 4-8
**Epochs**: 100-150
**Augmentation**: Multi-scale training

### Transformer-Based (TransUNet, UNETR, Swin-UNETR)

**Loss Function**: Combined (Dice + CE or Focal)
**Optimizer**: AdamW
**Learning Rate**: 1e-4
**Weight Decay**: 1e-4
**Batch Size**: 2-4 (reduce if OOM)
**Epochs**: 150-300 (longer training needed)
**Warmup**: 10-20 epochs
**Gradient Clipping**: 1.0
**Mixed Precision**: Recommended
**Augmentation**: Essential

---

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. **V-Net**: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016
3. **SegResNet**: Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization", BraTS 2018
4. **DeepLabV3+**: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018
5. **TransUNet**: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv 2021
6. **UNETR**: Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation", WACV 2022
7. **Swin-UNETR**: Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors", BraTS 2021

---

## Next Steps

- [Data Loading Guide](DATALOADER.md) - Learn how to prepare your data
- [Training Guide](TRAINING.md) - Train models with different configurations
- [Testing and Visualization](TESTING_VISUALIZATION.md) - Evaluate and visualize results
- [Example Workflows](../examples/medical-segmentation/) - Ready-to-use training pipelines
