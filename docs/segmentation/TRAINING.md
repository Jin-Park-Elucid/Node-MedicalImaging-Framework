# Training Guide for Medical Image Segmentation

This comprehensive guide covers loss functions, optimizers, training strategies, and best practices for medical image segmentation.

## Table of Contents

1. [Overview](#overview)
2. [Loss Functions](#loss-functions)
3. [Optimizers](#optimizers)
4. [Training Parameters](#training-parameters)
5. [Training Strategies](#training-strategies)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Common Issues](#common-issues)
8. [Example Configurations](#example-configurations)

---

## Overview

Training a medical image segmentation model involves:
1. Selecting appropriate loss functions
2. Choosing and configuring an optimizer
3. Setting training hyperparameters
4. Implementing training strategies
5. Monitoring training progress

---

## Loss Functions

### 1. Dice Loss

**Description**: Measures overlap between prediction and ground truth. Based on Dice Similarity Coefficient (DSC).

**Formula**:
```
Dice = 2 * |X ∩ Y| / (|X| + |Y|)
Dice Loss = 1 - Dice
```

**Characteristics**:
- Range: [0, 1]
- Differentiable approximation using softmax
- Handles class imbalance naturally
- Directly optimizes the evaluation metric

**When to Use**:
- Binary segmentation
- Multi-class segmentation
- Imbalanced datasets (small foreground)
- Medical imaging (standard choice)

**Configuration**:
```json
{
  "type": "LossFunction",
  "name": "dice_loss",
  "config": {
    "loss_type": "dice",
    "smooth": "1.0"
  }
}
```

**Pros**:
- Effective for imbalanced data
- Directly optimizes segmentation metric
- Works well for small objects

**Cons**:
- Can be unstable for very small objects
- Gradient might vanish for perfect predictions

---

### 2. Cross-Entropy Loss

**Description**: Standard classification loss measuring pixel-wise classification error.

**Formula**:
```
CE = -∑ y_true * log(y_pred)
```

**Characteristics**:
- Pixel-wise classification
- Can be weighted for class imbalance
- Well-studied and stable

**When to Use**:
- Balanced datasets
- Combined with other losses
- Multi-class segmentation
- When pixel-wise accuracy matters

**Configuration**:
```json
{
  "type": "LossFunction",
  "name": "ce_loss",
  "config": {
    "loss_type": "cross_entropy",
    "class_weights": "[0.3, 0.7]"
  }
}
```

**Pros**:
- Stable gradients
- Well-understood behavior
- Fast computation

**Cons**:
- Struggles with class imbalance
- Doesn't directly optimize overlap metrics

---

### 3. Focal Loss

**Description**: Modified cross-entropy that focuses on hard examples. Reduces weight of easy examples.

**Formula**:
```
FL = -α * (1 - p)^γ * log(p)
```
- α: class weight
- γ: focusing parameter (typically 2)
- p: predicted probability

**Characteristics**:
- Down-weights easy examples
- Focuses on hard negatives
- Addresses extreme class imbalance

**When to Use**:
- Severe class imbalance (>99% background)
- Small object detection
- Difficult segmentation tasks

**Configuration**:
```json
{
  "type": "LossFunction",
  "name": "focal_loss",
  "config": {
    "loss_type": "focal",
    "alpha": "0.25",
    "gamma": "2.0"
  }
}
```

**Pros**:
- Excellent for extreme imbalance
- Automatic hard example mining
- Improves learning on difficult cases

**Cons**:
- More hyperparameters to tune
- Can be slower to converge

---

### 4. Combined Loss

**Description**: Weighted combination of multiple loss functions, typically Dice + Cross-Entropy.

**Formula**:
```
Combined = λ₁ * Dice + λ₂ * CE
```

**Characteristics**:
- Combines benefits of multiple losses
- Dice handles imbalance, CE provides stable gradients
- Most commonly used in practice

**When to Use**:
- Medical image segmentation (recommended)
- Both overlap and pixel accuracy matter
- General purpose segmentation

**Configuration**:
```json
{
  "type": "LossFunction",
  "name": "combined_loss",
  "config": {
    "loss_type": "combined",
    "dice_weight": "0.5",
    "ce_weight": "0.5"
  }
}
```

**Common Combinations**:
- **Dice + CE** (50/50): General purpose
- **Dice + Focal** (50/50): Extreme imbalance
- **Dice + CE + Boundary** (40/40/20): Fine boundaries

**Pros**:
- Best of both worlds
- Stable training
- Strong performance

**Cons**:
- More hyperparameters
- Slightly slower computation

---

### 5. Boundary Loss

**Description**: Focuses on boundary accuracy by penalizing boundary errors more heavily.

**When to Use**:
- Fine boundary delineation needed
- Medical structures with complex boundaries
- Combined with other losses

**Configuration**:
```json
{
  "type": "LossFunction",
  "name": "combined_boundary",
  "config": {
    "loss_type": "combined",
    "dice_weight": "0.4",
    "ce_weight": "0.4",
    "boundary_weight": "0.2"
  }
}
```

---

### Loss Function Selection Guide

| Dataset Characteristic | Recommended Loss |
|------------------------|------------------|
| Balanced classes | Cross-Entropy |
| Moderate imbalance (<10:1) | Dice or Combined |
| Severe imbalance (>100:1) | Focal or Dice + Focal |
| Small objects | Dice or Focal |
| Complex boundaries | Combined + Boundary |
| General medical imaging | **Combined (Dice + CE)** |

---

## Optimizers

### 1. Adam (Adaptive Moment Estimation)

**Description**: Adaptive learning rate optimizer with momentum.

**Characteristics**:
- Adaptive learning rates per parameter
- Combines momentum and RMSprop
- Generally works out of the box

**When to Use**:
- Default choice for most tasks
- CNN-based models (U-Net, V-Net)
- When you want reliable convergence

**Configuration**:
```json
{
  "type": "Optimizer",
  "name": "adam",
  "config": {
    "optimizer_type": "Adam",
    "learning_rate": "0.0001",
    "betas": "[0.9, 0.999]",
    "eps": "1e-8",
    "weight_decay": "0.00001"
  }
}
```

**Recommended Settings**:
- **Learning Rate**: 1e-4 to 1e-3
- **Beta1**: 0.9 (momentum)
- **Beta2**: 0.999 (RMSprop)
- **Weight Decay**: 1e-5 to 1e-4

**Pros**:
- Works well out of the box
- Fast convergence
- Handles sparse gradients well

**Cons**:
- Can overfit with insufficient regularization
- May not generalize as well as SGD

---

### 2. AdamW (Adam with Decoupled Weight Decay)

**Description**: Adam with proper weight decay regularization.

**Characteristics**:
- Decoupled weight decay from gradient
- Better regularization than Adam
- Recommended for transformers

**When to Use**:
- **Transformer models** (TransUNet, UNETR, Swin-UNETR)
- When regularization is important
- Large models prone to overfitting

**Configuration**:
```json
{
  "type": "Optimizer",
  "name": "adamw",
  "config": {
    "optimizer_type": "AdamW",
    "learning_rate": "0.0001",
    "betas": "[0.9, 0.999]",
    "weight_decay": "0.0001"
  }
}
```

**Recommended Settings**:
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4 to 1e-3 (higher than Adam)
- **Betas**: [0.9, 0.999]

**Pros**:
- Better generalization
- Proper regularization
- Best for transformers

**Cons**:
- Requires tuning weight decay
- Slightly more sensitive to hyperparameters

---

### 3. SGD (Stochastic Gradient Descent) with Momentum

**Description**: Classic optimizer with momentum.

**Characteristics**:
- Simple and interpretable
- Often better final performance
- Requires more tuning

**When to Use**:
- DeepLabV3+ (originally trained with SGD)
- When you have time to tune learning rate
- When best generalization is critical

**Configuration**:
```json
{
  "type": "Optimizer",
  "name": "sgd",
  "config": {
    "optimizer_type": "SGD",
    "learning_rate": "0.01",
    "momentum": "0.9",
    "weight_decay": "0.0001",
    "nesterov": "true"
  }
}
```

**Recommended Settings**:
- **Learning Rate**: 1e-2 to 1e-1 (higher than Adam)
- **Momentum**: 0.9
- **Weight Decay**: 1e-4
- **Nesterov**: true

**Pros**:
- Often best final performance
- Better generalization
- More interpretable

**Cons**:
- Requires learning rate scheduling
- Slower convergence
- More hyperparameter tuning needed

---

### Optimizer Selection Guide

| Model Type | Recommended Optimizer | Learning Rate |
|------------|----------------------|---------------|
| U-Net, V-Net, SegResNet | **Adam** | 1e-4 |
| DeepLabV3+ | SGD with momentum | 1e-2 |
| TransUNet, UNETR, Swin-UNETR | **AdamW** | 1e-4 |
| Small dataset (<100 cases) | Adam | 1e-4 |
| Large dataset (>500 cases) | SGD or AdamW | 1e-2 / 1e-4 |

---

## Training Parameters

### Essential Parameters

#### 1. Learning Rate

**Most critical hyperparameter**

**Guidelines**:
- **Adam/AdamW**: 1e-4 (typical), range: 1e-5 to 1e-3
- **SGD**: 1e-2 (typical), range: 1e-3 to 1e-1

**Finding Optimal LR**:
1. Use learning rate finder (range test)
2. Start with suggested value
3. Reduce if loss explodes
4. Increase if learning is too slow

**Learning Rate Schedules**:

**Step Decay**:
```json
{
  "lr_schedule": "step",
  "step_size": "50",
  "gamma": "0.1"
}
```
- Reduce LR by factor of 10 every 50 epochs

**Cosine Annealing**:
```json
{
  "lr_schedule": "cosine",
  "T_max": "200",
  "eta_min": "1e-6"
}
```
- Smooth decay following cosine curve

**ReduceLROnPlateau**:
```json
{
  "lr_schedule": "plateau",
  "patience": "10",
  "factor": "0.5"
}
```
- Reduce when validation loss plateaus

#### 2. Batch Size

**Impact**: Affects gradient stability and memory usage

**Guidelines**:
- **2D Images**: 4-16
- **3D Volumes**: 1-4
- **Transformer Models**: 2-4 (memory intensive)

**Memory vs. Performance Trade-off**:
```
Smaller Batch:
+ Less memory
+ Better generalization (noise in gradients)
- Unstable gradients
- Slower training

Larger Batch:
+ Stable gradients
+ Faster training (more parallelism)
+ Better batch norm statistics
- More memory
- May generalize worse
```

**Gradient Accumulation** (for small batch):
```json
{
  "batch_size": "2",
  "gradient_accumulation_steps": "4"
}
```
Effective batch size = 2 × 4 = 8

#### 3. Number of Epochs

**Guidelines by Dataset Size**:
- **Small (<50 cases)**: 200-500 epochs
- **Medium (50-200 cases)**: 100-200 epochs
- **Large (>200 cases)**: 50-150 epochs

**Early Stopping**:
```json
{
  "epochs": "300",
  "early_stopping": "true",
  "patience": "30",
  "min_delta": "0.001"
}
```

#### 4. Weight Decay (L2 Regularization)

**Purpose**: Prevent overfitting

**Guidelines**:
- **Adam**: 1e-5 to 1e-4
- **AdamW**: 1e-4 to 1e-3
- **SGD**: 1e-4 to 1e-3

**Rule**: Increase for larger models or smaller datasets

---

## Training Strategies

### 1. Mixed Precision Training

**Description**: Use FP16 for computation, FP32 for critical operations.

**Benefits**:
- 2× faster training
- 50% less memory
- Enables larger batch sizes

**Configuration**:
```json
{
  "type": "Trainer",
  "name": "trainer",
  "config": {
    "mixed_precision": "true"
  }
}
```

**When to Use**:
- Modern GPUs (Volta, Turing, Ampere, Ada)
- Transformer models
- Large 3D volumes
- Limited GPU memory

---

### 2. Gradient Clipping

**Description**: Limit gradient magnitude to prevent exploding gradients.

**Configuration**:
```json
{
  "gradient_clip": "1.0",
  "gradient_clip_type": "norm"
}
```

**When to Use**:
- **Transformer models** (essential)
- Deep networks
- Unstable training
- RNNs or attention mechanisms

**Recommended Values**:
- CNN models: 5.0 (if needed)
- Transformer models: 1.0

---

### 3. Learning Rate Warmup

**Description**: Gradually increase learning rate at the start of training.

**Configuration**:
```json
{
  "warmup_epochs": "10",
  "warmup_type": "linear"
}
```

**When to Use**:
- **Transformer models** (recommended)
- Large batch sizes
- Transfer learning
- Unstable initial training

---

### 4. Data Augmentation

**Covered in**: [Data Loading Guide](DATALOADER.md)

**Key Points**:
- Essential for small datasets
- Apply only to training data
- Check augmented samples visually

---

### 5. Checkpointing

**Description**: Save model periodically during training.

**Configuration**:
```json
{
  "checkpoint_dir": "checkpoints/",
  "save_frequency": "10",
  "save_best_only": "true",
  "monitor": "val_dice"
}
```

**Best Practices**:
- Save best model based on validation metric
- Keep last N checkpoints
- Save optimizer state for resuming

---

## Monitoring and Debugging

### Key Metrics to Monitor

#### Training Metrics

1. **Loss**:
   - Should decrease steadily
   - Fluctuations are normal
   - Should stabilize eventually

2. **Learning Rate**:
   - Track if using schedules
   - Verify warmup works correctly

3. **Gradient Norm**:
   - Detect exploding/vanishing gradients
   - Should be stable

#### Validation Metrics

1. **Dice Coefficient**:
   - Primary segmentation metric
   - Range: 0-1 (higher is better)

2. **IoU (Intersection over Union)**:
   - Alternative overlap metric
   - Closely related to Dice

3. **Pixel Accuracy**:
   - Overall pixel correctness
   - Can be misleading for imbalanced data

4. **Precision and Recall**:
   - Precision: % of predicted positives that are correct
   - Recall: % of actual positives that are detected

### Visualizing Training Progress

**TensorBoard Integration**:
```python
# Automatically logs to TensorBoard
# View with: tensorboard --logdir=logs/
```

**Tracked Information**:
- Loss curves (train and validation)
- Metric curves (Dice, IoU)
- Learning rate schedule
- Sample predictions
- Gradient histograms

### Common Training Curves

**Healthy Training**:
```
Loss ↓ smoothly
Val Loss ↓ follows train loss
Dice ↑ steadily
Gap between train/val is small
```

**Overfitting**:
```
Train Loss ↓ continues
Val Loss ↑ starts increasing
Large gap between train/val metrics
```

**Underfitting**:
```
Both train and val loss high
Metrics plateau early
Model capacity too small
```

**Unstable Training**:
```
Loss fluctuates wildly
Gradients explode (NaN loss)
→ Reduce learning rate
→ Add gradient clipping
```

---

## Common Issues

### Issue 1: Loss is NaN

**Causes**:
- Learning rate too high
- Exploding gradients
- Numerical instability

**Solutions**:
1. Reduce learning rate (try 0.1×)
2. Add gradient clipping
3. Check data for inf/nan values
4. Use mixed precision carefully

```json
{
  "learning_rate": "0.00001",
  "gradient_clip": "1.0"
}
```

---

### Issue 2: Model Not Learning

**Symptoms**: Loss doesn't decrease, metrics don't improve

**Causes**:
- Learning rate too low
- Wrong loss function
- Data issues
- Model too small

**Solutions**:
1. Increase learning rate
2. Verify data is correct
3. Check loss function matches task
4. Try larger model
5. Verify gradients are flowing

---

### Issue 3: Overfitting

**Symptoms**: Train loss ↓, val loss ↑, large train/val gap

**Solutions**:
1. **Increase regularization**:
   ```json
   {"weight_decay": "0.001"}
   ```

2. **More data augmentation**:
   ```json
   {"augmentation": "true", "augmentation_strength": "strong"}
   ```

3. **Early stopping**:
   ```json
   {"early_stopping": "true", "patience": "20"}
   ```

4. **Dropout** (add to model):
   ```python
   nn.Dropout(p=0.5)
   ```

5. **Reduce model capacity**:
   - Fewer channels
   - Fewer layers

---

### Issue 4: Poor Convergence

**Symptoms**: Slow learning, metrics plateau

**Solutions**:
1. **Better learning rate**:
   - Use LR finder
   - Try different schedule

2. **Different optimizer**:
   - Adam usually fastest
   - SGD may need higher LR

3. **Batch normalization issues**:
   - Check batch size (>1 for batch norm)
   - Try group norm for small batches

4. **Architecture mismatch**:
   - Try different network
   - Adjust capacity

---

### Issue 5: Class Imbalance

**Symptoms**: Model predicts only majority class

**Solutions**:
1. **Use Dice Loss**:
   ```json
   {"loss_type": "dice"}
   ```

2. **Weighted loss**:
   ```json
   {"loss_type": "cross_entropy", "class_weights": "[0.1, 0.9]"}
   ```

3. **Focal Loss**:
   ```json
   {"loss_type": "focal", "alpha": "0.25", "gamma": "2.0"}
   ```

4. **Oversample minority class**

---

## Example Configurations

### Configuration 1: U-Net on Moderate Dataset

```json
{
  "workflow": "U-Net Training",
  "nodes": [
    {
      "type": "MedicalSegmentationLoader",
      "name": "data_loader",
      "config": {
        "data_dir": "data/medical",
        "batch_size": "8",
        "image_size": "256",
        "augmentation": "true"
      }
    },
    {
      "type": "UNet2D",
      "name": "model",
      "config": {
        "in_channels": "1",
        "out_channels": "2",
        "base_channels": "64"
      }
    },
    {
      "type": "LossFunction",
      "name": "loss",
      "config": {
        "loss_type": "combined",
        "dice_weight": "0.5",
        "ce_weight": "0.5"
      }
    },
    {
      "type": "Optimizer",
      "name": "optimizer",
      "config": {
        "optimizer_type": "Adam",
        "learning_rate": "0.0001",
        "weight_decay": "0.00001"
      }
    },
    {
      "type": "Trainer",
      "name": "trainer",
      "config": {
        "epochs": "100",
        "device": "cuda",
        "checkpoint_dir": "checkpoints/unet",
        "save_best_only": "true",
        "early_stopping": "true",
        "patience": "20"
      }
    }
  ]
}
```

---

### Configuration 2: Transformer Model (Swin-UNETR)

```json
{
  "workflow": "Swin-UNETR Training",
  "nodes": [
    {
      "type": "MedicalSegmentationLoader",
      "name": "data_loader",
      "config": {
        "data_dir": "data/3d_volumes",
        "batch_size": "2",
        "image_size": "96",
        "augmentation": "true"
      }
    },
    {
      "type": "SwinUNETR",
      "name": "model",
      "config": {
        "img_size": "96",
        "in_channels": "1",
        "out_channels": "2",
        "embed_dim": "48"
      }
    },
    {
      "type": "LossFunction",
      "name": "loss",
      "config": {
        "loss_type": "combined",
        "dice_weight": "0.5",
        "focal_weight": "0.5"
      }
    },
    {
      "type": "Optimizer",
      "name": "optimizer",
      "config": {
        "optimizer_type": "AdamW",
        "learning_rate": "0.0001",
        "weight_decay": "0.0001"
      }
    },
    {
      "type": "Trainer",
      "name": "trainer",
      "config": {
        "epochs": "200",
        "device": "cuda",
        "mixed_precision": "true",
        "gradient_clip": "1.0",
        "warmup_epochs": "10",
        "lr_schedule": "cosine",
        "checkpoint_dir": "checkpoints/swin_unetr"
      }
    }
  ]
}
```

---

### Configuration 3: Small Dataset with Heavy Augmentation

```json
{
  "workflow": "Small Dataset Training",
  "nodes": [
    {
      "type": "MedicalSegmentationLoader",
      "name": "data_loader",
      "config": {
        "data_dir": "data/small",
        "batch_size": "4",
        "image_size": "256",
        "augmentation": "true",
        "augmentation_strength": "strong"
      }
    },
    {
      "type": "UNet2D",
      "name": "model",
      "config": {
        "in_channels": "1",
        "out_channels": "2",
        "base_channels": "32"
      }
    },
    {
      "type": "LossFunction",
      "name": "loss",
      "config": {
        "loss_type": "dice"
      }
    },
    {
      "type": "Optimizer",
      "name": "optimizer",
      "config": {
        "optimizer_type": "Adam",
        "learning_rate": "0.0001",
        "weight_decay": "0.0001"
      }
    },
    {
      "type": "Trainer",
      "name": "trainer",
      "config": {
        "epochs": "300",
        "device": "cuda",
        "early_stopping": "true",
        "patience": "50"
      }
    }
  ]
}
```

---

## Best Practices Summary

### Quick Start Guide

1. **Start Simple**:
   - U-Net architecture
   - Adam optimizer (LR=1e-4)
   - Combined loss (Dice + CE)
   - 100 epochs

2. **Monitor Training**:
   - Watch train/val loss
   - Check Dice coefficient
   - Visualize predictions

3. **Iterate**:
   - Adjust learning rate if needed
   - Add regularization for overfitting
   - Try different architectures

### Architecture-Specific Settings

**U-Net / V-Net / SegResNet**:
- Optimizer: Adam, LR=1e-4
- Loss: Combined (Dice + CE)
- Epochs: 100-200

**DeepLabV3+**:
- Optimizer: SGD, LR=1e-2, momentum=0.9
- Loss: Cross-Entropy or Combined
- LR Schedule: Polynomial decay

**TransUNet / UNETR / Swin-UNETR**:
- Optimizer: AdamW, LR=1e-4, WD=1e-4
- Loss: Combined (Dice + CE/Focal)
- Epochs: 150-300
- Warmup: 10-20 epochs
- Gradient Clip: 1.0
- Mixed Precision: Recommended

---

## Next Steps

- [Testing and Visualization](TESTING_VISUALIZATION.md) - Evaluate your trained model
- [Network Architectures](NETWORK_ARCHITECTURES.md) - Understand model choices
- [Data Loading](DATALOADER.md) - Prepare your data properly
- [Example Workflows](../examples/medical-segmentation/) - Complete training pipelines
