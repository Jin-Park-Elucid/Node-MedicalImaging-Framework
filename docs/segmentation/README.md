# Medical Image Segmentation - Complete Guide

Welcome to the comprehensive guide for medical image segmentation using the Medical Imaging Framework. This documentation covers everything from data preparation to model deployment.

## Quick Navigation

### 📚 Core Documentation

1. **[Network Architectures](NETWORK_ARCHITECTURES.md)** - Complete guide to all 8 available segmentation networks
   - U-Net (2D/3D) - Foundational architecture
   - V-Net - 3D volumetric with residual blocks
   - SegResNet - Efficient segmentation
   - DeepLabV3+ - Multi-scale ASPP
   - TransUNet - Hybrid CNN-Transformer
   - UNETR - Pure transformer encoder
   - Swin-UNETR - Hierarchical Swin Transformer
   - Architecture comparison and selection guide

2. **[Data Loading](DATALOADER.md)** - Data preparation and augmentation
   - Data organization strategies
   - Supported formats (NIfTI, DICOM, PNG, NPY)
   - Data augmentation techniques
   - Memory management
   - Best practices

3. **[Training](TRAINING.md)** - Loss functions, optimizers, and training strategies
   - Loss functions (Dice, CE, Focal, Combined)
   - Optimizers (Adam, AdamW, SGD)
   - Training parameters and hyperparameters
   - Advanced training strategies
   - Monitoring and debugging

4. **[Testing and Visualization](TESTING_VISUALIZATION.md)** - Model evaluation and result analysis
   - Evaluation metrics (Dice, IoU, HD95, ASD)
   - Visualization techniques
   - Performance analysis
   - Clinical validation
   - Error analysis

---

## Quick Start

### 1. Choose Your Path

**I want to...**

- **Train a model on my data** → Start with [Data Loading](DATALOADER.md)
- **Understand available models** → Read [Network Architectures](NETWORK_ARCHITECTURES.md)
- **Optimize training** → Check [Training Guide](TRAINING.md)
- **Evaluate my model** → See [Testing and Visualization](TESTING_VISUALIZATION.md)
- **Use pre-made workflows** → Browse [Example Workflows](../../examples/medical_segmentation_pipeline/workflows/)

### 2. Recommended Learning Path

**Beginner Path** (Getting started):
1. Read Network Architectures overview
2. Follow Data Loading guide to prepare data
3. Use a simple workflow (U-Net training)
4. Test and visualize results

**Intermediate Path** (Improving performance):
1. Understand loss functions and optimizers
2. Experiment with different architectures
3. Tune hyperparameters
4. Implement advanced training strategies

**Advanced Path** (State-of-the-art):
1. Try transformer-based architectures
2. Implement custom augmentations
3. Use ensemble methods
4. Optimize for clinical deployment

---

## Available Networks

### CNN-Based

| Network | Type | Best For | Difficulty |
|---------|------|----------|------------|
| **U-Net** | 2D/3D CNN | General purpose, baseline | ⭐ Easy |
| **V-Net** | 3D CNN | Volumetric data, 3D context | ⭐⭐ Moderate |
| **SegResNet** | 2D/3D CNN | Efficiency, limited resources | ⭐⭐ Moderate |
| **DeepLabV3+** | 2D CNN | Multi-scale, varying object sizes | ⭐⭐⭐ Advanced |

### Transformer-Based

| Network | Type | Best For | Difficulty |
|---------|------|----------|------------|
| **TransUNet** | Hybrid | 2D global context | ⭐⭐⭐ Advanced |
| **UNETR** | Pure Transformer | 3D long-range dependencies | ⭐⭐⭐⭐ Expert |
| **Swin-UNETR** | Hierarchical Transformer | State-of-the-art 3D | ⭐⭐⭐⭐ Expert |

### Quick Selection

**Quick Decision Tree**:
```
Do you have 2D or 3D images?
├── 2D
│   ├── Limited resources? → U-Net 2D
│   ├── Varying object sizes? → DeepLabV3+
│   └── Need global context? → TransUNet
│
└── 3D
    ├── Limited resources? → SegResNet
    ├── Standard segmentation? → U-Net 3D or V-Net
    └── State-of-the-art needed? → Swin-UNETR or UNETR
```

---

## Example Workflows

Pre-configured workflows are available in `examples/medical_segmentation_pipeline/workflows/`:

### Training Workflows

1. **[vnet_training.json](../../examples/medical_segmentation_pipeline/workflows/vnet_training.json)**
   - 3D volumetric segmentation with V-Net
   - Dice loss, Adam optimizer
   - Good for CT/MRI volumes

2. **[transunet_training.json](../../examples/medical_segmentation_pipeline/workflows/transunet_training.json)**
   - 2D hybrid CNN-Transformer
   - Combined loss, AdamW optimizer
   - Good for 2D medical images

3. **[unetr_training.json](../../examples/medical_segmentation_pipeline/workflows/unetr_training.json)**
   - 3D pure transformer encoder
   - Mixed precision, gradient clipping
   - State-of-the-art 3D segmentation

4. **[swin_unetr_training.json](../../examples/medical_segmentation_pipeline/workflows/swin_unetr_training.json)**
   - Hierarchical Swin Transformer
   - Best performance for 3D medical imaging
   - Resource intensive

5. **[deeplabv3plus_training.json](../../examples/medical_segmentation_pipeline/workflows/deeplabv3plus_training.json)**
   - Multi-scale ASPP architecture
   - SGD with momentum
   - Good for 2D with varying object sizes

### Testing Workflows

- **[testing_workflow.json](../../examples/medical_segmentation_pipeline/testing_workflow.json)**
  - Complete testing pipeline
  - Metrics calculation and visualization
  - Works with any trained model

---

## Common Use Cases

### Use Case 1: Brain MRI Segmentation

**Recommended**:
- **Architecture**: U-Net 3D or Swin-UNETR
- **Loss**: Dice + Cross-Entropy
- **Optimizer**: Adam (U-Net) or AdamW (Swin-UNETR)
- **Data**: NIfTI format, 3D volumes
- **Augmentation**: Standard (rotation, flip, elastic)

**Workflow**: `examples/medical_segmentation_pipeline/workflows/swin_unetr_training.json`

### Use Case 2: CT Organ Segmentation

**Recommended**:
- **Architecture**: V-Net or SegResNet
- **Loss**: Dice Loss
- **Optimizer**: Adam
- **Data**: NIfTI or DICOM, 3D volumes
- **Preprocessing**: HU windowing, resampling

**Workflow**: `examples/medical_segmentation_pipeline/workflows/vnet_training.json`

### Use Case 3: 2D X-Ray Segmentation

**Recommended**:
- **Architecture**: U-Net 2D or TransUNet
- **Loss**: Combined (Dice + CE)
- **Optimizer**: Adam
- **Data**: PNG or DICOM, 2D images
- **Augmentation**: Strong (limited data)

**Workflow**: `examples/medical_segmentation_pipeline/workflows/transunet_training.json`

### Use Case 4: Small Dataset (<50 cases)

**Recommended**:
- **Architecture**: U-Net (simpler is better)
- **Loss**: Dice Loss
- **Optimizer**: Adam with higher weight decay
- **Augmentation**: Very strong
- **Training**: More epochs, early stopping

**Tips**:
- Heavy data augmentation essential
- Consider transfer learning
- Use cross-validation
- Monitor overfitting carefully

### Use Case 5: Large Dataset (>500 cases)

**Recommended**:
- **Architecture**: Transformer-based (UNETR, Swin-UNETR)
- **Loss**: Combined
- **Optimizer**: AdamW or SGD
- **Augmentation**: Moderate
- **Training**: Longer training, learning rate schedule

**Workflow**: `examples/medical_segmentation_pipeline/workflows/swin_unetr_training.json`

---

## Performance Benchmarks

### Expected Performance (Dice Coefficient)

**General Medical Segmentation**:
- U-Net: 0.80 - 0.90
- V-Net: 0.82 - 0.92
- SegResNet: 0.80 - 0.88
- DeepLabV3+: 0.82 - 0.92
- TransUNet: 0.84 - 0.93
- UNETR: 0.86 - 0.94
- Swin-UNETR: 0.88 - 0.96 (state-of-the-art)

**By Task**:
- Brain tumor segmentation: 0.85 - 0.92
- Organ segmentation: 0.90 - 0.96
- Lesion detection: 0.70 - 0.85
- Multi-organ: 0.88 - 0.94

*Note: Actual performance depends on dataset quality, size, and complexity.*

---

## Training Time Estimates

**Approximate training time per epoch** (on NVIDIA A100 GPU):

| Architecture | 2D (256×256) | 3D (96×96×96) |
|-------------|--------------|---------------|
| U-Net | 2-5 min | 10-20 min |
| V-Net | N/A | 15-25 min |
| SegResNet | 1-3 min | 8-15 min |
| DeepLabV3+ | 3-6 min | N/A |
| TransUNet | 5-10 min | N/A |
| UNETR | N/A | 25-40 min |
| Swin-UNETR | N/A | 30-50 min |

*Batch size: 4 (2D), 2 (3D); Dataset: ~100 cases*

---

## Hardware Requirements

### Minimum Requirements

**For 2D Segmentation (U-Net, DeepLabV3+)**:
- GPU: 6GB VRAM (GTX 1060, RTX 2060)
- RAM: 16GB
- Storage: 50GB SSD

**For 3D Segmentation (V-Net, SegResNet)**:
- GPU: 8GB VRAM (RTX 2070, RTX 3060)
- RAM: 32GB
- Storage: 100GB SSD

**For Transformer Models (UNETR, Swin-UNETR)**:
- GPU: 16GB+ VRAM (RTX 3090, RTX 4090, A100)
- RAM: 64GB
- Storage: 200GB SSD

### Recommended Setup

- **GPU**: NVIDIA RTX 3090/4090 or A100 (24GB+ VRAM)
- **RAM**: 64GB+
- **Storage**: 500GB+ NVMe SSD
- **CPU**: Modern multi-core (for data loading)

---

## Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution | Documentation |
|-------|---------------|----------|---------------|
| Out of memory | Batch size too large | Reduce batch size, use mixed precision | [Training](TRAINING.md#common-issues) |
| Poor performance | Wrong architecture | Try different network | [Network Architectures](NETWORK_ARCHITECTURES.md#selection-guide) |
| Slow training | Data loading bottleneck | Increase num_workers | [Data Loading](DATALOADER.md#common-issues) |
| Overfitting | Insufficient regularization | More augmentation, weight decay | [Training](TRAINING.md#common-issues) |
| Model predicts only background | Class imbalance | Use Dice or Focal loss | [Training](TRAINING.md#loss-functions) |

---

## Best Practices Summary

### Data Preparation
✅ Organize data in clear folder structure
✅ Use consistent file naming
✅ Verify data integrity (matching images/masks)
✅ Apply appropriate preprocessing
✅ Use data augmentation (especially for small datasets)

### Model Selection
✅ Start simple (U-Net) before trying complex models
✅ Match architecture to your data (2D vs 3D)
✅ Consider computational resources
✅ Review literature for your specific task

### Training
✅ Use combined loss (Dice + CE) for robustness
✅ Start with recommended hyperparameters
✅ Monitor both training and validation metrics
✅ Save checkpoints regularly
✅ Use early stopping to prevent overfitting

### Evaluation
✅ Use multiple metrics (Dice, IoU, HD95)
✅ Perform per-case analysis
✅ Visualize predictions qualitatively
✅ Identify and analyze failure cases
✅ Consider clinical validation

---

## Additional Resources

### Documentation
- [Quick Reference](../getting-started/QUICK_REFERENCE.md)
- [Getting Started](../getting-started/GETTING_STARTED.md)
- [GUI Guide](../gui/VISUAL_GUI_COMPLETE.md)
- [Project Status](../project/PROJECT_STATUS.md)

### Example Code
- Training workflows: `examples/medical_segmentation_pipeline/workflows/`
- Sample data: `examples/medical_segmentation_pipeline/data/`
- Jupyter notebooks: `examples/notebooks/`

### External Resources
- **Papers**: Links in [Network Architectures](NETWORK_ARCHITECTURES.md#references)
- **Datasets**: Medical Segmentation Decathlon, BRATS, LiTS, KiTS
- **Tools**: ITK-SNAP, 3D Slicer, MITK

---

## Support and Contributing

### Getting Help
- Check documentation for your specific issue
- Review example workflows
- Check [Common Issues](#troubleshooting) section

### Contributing
- Found a bug? Report it
- Have a feature request? Let us know
- Want to contribute? See [Contributing Guide](../project/CONTRIBUTING.md)

---

## Quick Reference Card

### Essential Commands

```bash
# Load and train a model using GUI
python -m medical_imaging_framework.gui

# Run training workflow from command line
python run_workflow.py --workflow examples/medical_segmentation_pipeline/workflows/vnet_training.json

# Run testing workflow
python run_workflow.py --workflow examples/medical_segmentation_pipeline/testing_workflow.json

# Visualize results
python visualize_results.py --predictions results/predictions --output results/visualizations
```

### Essential Configurations

**Simple U-Net Training**:
```json
{
  "architecture": "UNet2D",
  "loss": "combined",
  "optimizer": "Adam",
  "lr": 0.0001,
  "epochs": 100
}
```

**Advanced Swin-UNETR Training**:
```json
{
  "architecture": "SwinUNETR",
  "loss": "combined",
  "optimizer": "AdamW",
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "epochs": 200,
  "mixed_precision": true,
  "gradient_clip": 1.0
}
```

---

## Version History

- **v1.0.0** (2024-01-31): Initial comprehensive documentation
  - 8 network architectures
  - Complete training pipeline
  - Testing and visualization
  - Example workflows

---

## License

This documentation is part of the Medical Imaging Framework.
See [LICENSE](../../LICENSE) for details.

---

**Happy Segmenting! 🏥🔬**

For questions or issues, please refer to the appropriate documentation section above or check the [Troubleshooting](#troubleshooting) guide.
