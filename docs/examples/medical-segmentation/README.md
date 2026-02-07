# Medical Image Segmentation Pipeline

Complete example demonstrating medical image segmentation using the Medical Imaging Framework.

## Overview

This example shows a full pipeline for medical image segmentation:
- Data generation/loading
- Custom dataloader node
- Training pipeline with U-Net
- Testing pipeline with visualization
- Comparison of ground truth vs predictions

## Directory Structure

```
medical_segmentation_pipeline/
├── README.md                    # This file
├── download_dataset.py          # Generate synthetic medical data
├── custom_dataloader.py         # Custom dataloader node
├── train_pipeline.py            # Training pipeline
├── test_pipeline.py             # Testing pipeline with visualization
├── data/                        # Dataset (generated)
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── masks/              # Training masks
│   ├── test/
│   │   ├── images/             # Test images
│   │   └── masks/              # Test masks
│   └── dataset_info.json       # Dataset metadata
├── models/                      # Saved models
│   └── best_model_*.pth        # Best trained model
└── results/                     # Test results
    ├── test_metrics.txt        # Evaluation metrics
    └── visualizations/         # Comparison images
        └── comparison_*.png    # GT vs Prediction
```

## Quick Start

You can run the pipeline either via **command line** or **GUI**.

### GUI Method (Recommended for Visual Workflow)

```bash
# Launch GUI with custom nodes
python examples/medical_segmentation_pipeline/launch_gui.py

# In the GUI:
# 1. File → Load Workflow → training_workflow.json
# 2. Workflow → Validate
# 3. Workflow → Execute
```

See [GUI_GUIDE.md](GUI_GUIDE.md) for detailed GUI instructions.

### Command Line Method

### 1. Generate Dataset

```bash
cd examples/medical_segmentation_pipeline
python download_dataset.py
```

This creates synthetic medical imaging data:
- 50 training samples (256x256 grayscale images)
- 20 test samples
- Binary segmentation task (background vs foreground)

**Note**: In production, replace with real medical imaging data.

### 2. Test Dataloader (Optional)

```bash
python custom_dataloader.py
```

Verifies the custom dataloader works correctly.

### 3. Train Model

```bash
python train_pipeline.py
```

Training configuration:
- **Model**: U-Net 2D (3 levels, 32 base channels)
- **Loss**: Dice loss
- **Optimizer**: Adam (lr=1e-3)
- **Epochs**: 20
- **Batch size**: 4

The trained model is saved to `models/best_model_*.pth`.

### 4. Test Model & Generate Visualizations

```bash
python test_pipeline.py
```

This will:
- Load the trained model
- Run inference on test set
- Calculate metrics (Accuracy, Dice, IoU, etc.)
- Generate comparison visualizations

Results saved to `results/`:
- `test_metrics.txt`: Quantitative metrics
- `visualizations/comparison_*.png`: Visual comparisons

## Visualization Output

Each comparison image contains 4 panels:

1. **Input Image**: Original grayscale medical image
2. **Ground Truth**: True segmentation mask
3. **Prediction**: Model's predicted mask
4. **Overlay**: Color-coded comparison
   - 🔴 Red: Ground truth only (missed by model)
   - 🟢 Green: Prediction only (false positive)
   - 🟡 Yellow: Both (correct prediction)

## Custom Dataloader Node

The example includes a custom node `MedicalSegmentationLoaderNode` that:
- Loads medical images and masks from disk
- Creates PyTorch DataLoaders
- Integrates seamlessly with the framework
- Can be used in other pipelines

```python
from custom_dataloader import MedicalSegmentationLoaderNode

# Create and configure the node
loader = NodeRegistry.create_node('MedicalSegmentationLoader', 'loader', config={
    'data_dir': 'path/to/data',
    'batch_size': 4,
    'num_workers': 0,
    'shuffle_train': True
})
```

## Pipeline Architecture

### Training Pipeline

```
MedicalSegmentationLoader → UNet2D
                          ↓
                     LossFunction (Dice)
                          ↓
                      Optimizer (Adam)
                          ↓
                       Trainer
```

### Testing Pipeline

```
MedicalSegmentationLoader → UNet2D (trained) → BatchPredictor
                                                      ↓
                                                Visualization
```

## Dataset Format

The dataloader expects this structure:

```
data/
├── train/
│   ├── images/
│   │   ├── image_0000.png
│   │   ├── image_0001.png
│   │   └── ...
│   └── masks/
│       ├── mask_0000.png
│       ├── mask_0001.png
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── masks/
        └── ...
```

**Image Requirements**:
- Format: PNG, JPG, or similar
- Type: Grayscale (single channel)
- Resolution: Any (will be processed as-is)

**Mask Requirements**:
- Format: Same as images
- Type: Binary (0 for background, 255 for foreground)
- Resolution: Must match corresponding image

## Extending the Example

### Use Real Medical Data

Replace synthetic data with real medical images:

1. Download public dataset (e.g., Medical Segmentation Decathlon)
2. Organize into the expected format
3. Update `data_dir` in train/test scripts

### Modify Network Architecture

Change model configuration in `train_pipeline.py`:

```python
model_node = NodeRegistry.create_node('UNet2D', 'model', config={
    'in_channels': 1,
    'out_channels': 2,
    'base_channels': 64,  # Increase capacity
    'depth': 4            # Deeper network
})
```

### Use Different Loss Function

Replace Dice loss with cross-entropy or custom loss:

```python
loss_node = NodeRegistry.create_node('LossFunction', 'loss', config={
    'loss_type': 'cross_entropy'
})
```

### Multi-class Segmentation

For multi-class segmentation (e.g., multiple organs):

1. Update masks to have multiple classes (0, 1, 2, ...)
2. Change `out_channels` in model to number of classes
3. Adjust visualization to handle multiple classes

## Metrics Explained

- **Accuracy**: Overall pixel accuracy
- **Precision**: Of predicted foreground, how much is correct
- **Recall**: Of actual foreground, how much is detected
- **F1 Score**: Harmonic mean of precision and recall
- **Dice Coefficient**: Overlap between prediction and ground truth (2*TP/(2*TP+FP+FN))
- **IoU (Intersection over Union)**: Jaccard index (TP/(TP+FP+FN))

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in configuration:
```python
batch_size = 2  # Instead of 4
```

### Training Too Slow

Options:
- Use smaller network (reduce `base_channels` or `depth`)
- Reduce number of epochs
- Use GPU if available

### Poor Segmentation Results

- Increase training epochs
- Use larger network
- Adjust learning rate
- Use data augmentation
- Replace synthetic data with real data

## Next Steps

1. **Try Different Architectures**: Use UNet3D, AttentionUNet2D
2. **Add Data Augmentation**: Use RandomFlip, RandomRotation nodes
3. **Implement Cross-Validation**: Split data into multiple folds
4. **Export to ONNX**: For deployment
5. **Use Real Medical Data**: Download from public repositories

## Related Documentation

- [Framework Getting Started](../../docs/GETTING_STARTED.md)
- [Available Nodes](../../docs/README.md#available-nodes)
- [Creating Custom Nodes](../../docs/README.md#creating-custom-nodes)
- [Training Nodes](../../docs/README.md#training-nodes)

## Citation

If using real medical data, cite the original dataset:

```bibtex
@article{dataset_name,
  title={Dataset Title},
  author={Authors},
  journal={Journal},
  year={Year}
}
```

---

**Medical Imaging Framework** - Node-based deep learning for medical imaging
