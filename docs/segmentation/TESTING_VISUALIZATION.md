# Testing and Visualization Guide

This guide covers model evaluation, metrics calculation, visualization techniques, and result analysis for medical image segmentation.

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Testing Workflow](#testing-workflow)
4. [Visualization Techniques](#visualization-techniques)
5. [Performance Analysis](#performance-analysis)
6. [Clinical Validation](#clinical-validation)
7. [Common Issues](#common-issues)

---

## Overview

After training a segmentation model, proper evaluation is crucial to:
- Measure model performance quantitatively
- Visualize predictions for qualitative assessment
- Identify failure cases
- Validate clinical applicability
- Compare different models

---

## Evaluation Metrics

### 1. Dice Similarity Coefficient (DSC)

**Description**: Measures overlap between prediction and ground truth.

**Formula**:
```
DSC = 2 * |X ∩ Y| / (|X| + |Y|)
```

**Range**: 0 (no overlap) to 1 (perfect overlap)

**Interpretation**:
- **DSC > 0.9**: Excellent segmentation
- **0.7 < DSC < 0.9**: Good segmentation
- **0.5 < DSC < 0.7**: Moderate segmentation
- **DSC < 0.5**: Poor segmentation

**When to Use**:
- Primary metric for segmentation
- Standard in medical imaging
- Works well for imbalanced data

**Code**:
```python
def dice_coefficient(pred, target):
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum())
```

**Configuration**:
```json
{
  "type": "MetricsCalculator",
  "name": "metrics",
  "config": {
    "metrics": ["dice", "iou", "accuracy"]
  }
}
```

---

### 2. Intersection over Union (IoU) / Jaccard Index

**Description**: Ratio of intersection to union of prediction and ground truth.

**Formula**:
```
IoU = |X ∩ Y| / |X ∪ Y|
```

**Relationship to Dice**:
```
DSC = 2 * IoU / (1 + IoU)
IoU = DSC / (2 - DSC)
```

**Range**: 0 to 1

**Interpretation**:
- **IoU > 0.8**: Excellent
- **0.5 < IoU < 0.8**: Good
- **IoU < 0.5**: Poor

**When to Use**:
- Alternative to Dice
- Object detection benchmarks
- Computing metrics per class

---

### 3. Hausdorff Distance (HD)

**Description**: Maximum distance from a point in one set to the nearest point in the other set.

**Formula**:
```
HD(X, Y) = max(h(X, Y), h(Y, X))
where h(X, Y) = max_{x∈X} min_{y∈Y} ||x - y||
```

**Unit**: Same as image spacing (mm, pixels)

**Interpretation**:
- Lower is better
- Sensitive to outliers
- Measures worst-case boundary error

**95th Percentile Hausdorff Distance (HD95)**:
- More robust to outliers
- Preferred in medical imaging
- Ignores worst 5% of errors

**When to Use**:
- Boundary accuracy is critical
- Organ segmentation
- Tumor margin delineation
- Clinical applications

---

### 4. Average Surface Distance (ASD)

**Description**: Average distance between predicted and ground truth surfaces.

**Formula**:
```
ASD = (mean_distance(surface_pred, surface_gt) +
       mean_distance(surface_gt, surface_pred)) / 2
```

**Unit**: Same as image spacing (mm, pixels)

**Interpretation**:
- Lower is better
- More robust than HD
- Measures average boundary error

**When to Use**:
- Complement to HD
- More stable metric
- Radiation therapy planning

---

### 5. Sensitivity (Recall / True Positive Rate)

**Description**: Proportion of actual positives correctly identified.

**Formula**:
```
Sensitivity = TP / (TP + FN)
```

**Range**: 0 to 1

**Interpretation**:
- High sensitivity: Few false negatives
- Critical when missing positives is costly
- Example: Tumor detection (don't miss tumors)

**When to Use**:
- Disease detection tasks
- Screening applications
- When false negatives are dangerous

---

### 6. Specificity (True Negative Rate)

**Description**: Proportion of actual negatives correctly identified.

**Formula**:
```
Specificity = TN / (TN + FP)
```

**Range**: 0 to 1

**Interpretation**:
- High specificity: Few false positives
- Important when false alarms are costly

---

### 7. Precision (Positive Predictive Value)

**Description**: Proportion of predicted positives that are correct.

**Formula**:
```
Precision = TP / (TP + FP)
```

**Range**: 0 to 1

**Interpretation**:
- High precision: Few false alarms
- Important for avoiding unnecessary interventions

---

### 8. F1 Score

**Description**: Harmonic mean of precision and recall.

**Formula**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Range**: 0 to 1

**When to Use**:
- Balance between precision and recall
- Alternative to Dice (closely related)

---

### Metric Selection Guide

| Use Case | Primary Metrics | Secondary Metrics |
|----------|----------------|-------------------|
| General Segmentation | Dice, IoU | Precision, Recall |
| Boundary-Critical | HD95, ASD | Dice |
| Tumor Detection | Sensitivity | Dice, Specificity |
| Organ Segmentation | Dice, HD95 | IoU, ASD |
| Multi-Organ | Per-class Dice | Mean Dice, HD95 |

---

## Testing Workflow

### Basic Testing Pipeline

```json
{
  "name": "Testing Workflow",
  "nodes": [
    {
      "type": "MedicalSegmentationLoader",
      "name": "test_data",
      "config": {
        "data_dir": "data/test",
        "batch_size": "1",
        "shuffle": "false"
      }
    },
    {
      "type": "UNet2D",
      "name": "trained_model",
      "config": {
        "checkpoint": "checkpoints/best_model.pth"
      }
    },
    {
      "type": "BatchPredictor",
      "name": "predictor",
      "config": {
        "save_predictions": "true",
        "output_dir": "results/predictions"
      }
    },
    {
      "type": "MetricsCalculator",
      "name": "metrics",
      "config": {
        "metrics": ["dice", "iou", "hd95", "asd"],
        "per_case": "true"
      }
    },
    {
      "type": "Visualizer",
      "name": "visualizer",
      "config": {
        "output_dir": "results/visualizations",
        "overlay_alpha": "0.5"
      }
    }
  ],
  "links": [
    {
      "source_node": "test_data",
      "source_port": "test_loader",
      "target_node": "predictor",
      "target_port": "dataloader"
    },
    {
      "source_node": "trained_model",
      "source_port": "output",
      "target_node": "predictor",
      "target_port": "model"
    },
    {
      "source_node": "predictor",
      "source_port": "predictions",
      "target_node": "metrics",
      "target_port": "predictions"
    },
    {
      "source_node": "predictor",
      "source_port": "all_labels",
      "target_node": "metrics",
      "target_port": "labels"
    },
    {
      "source_node": "predictor",
      "source_port": "predictions",
      "target_node": "visualizer",
      "target_port": "predictions"
    }
  ]
}
```

### Batch Predictor Node

**Purpose**: Run inference on test data and save predictions.

**Configuration**:
```json
{
  "type": "BatchPredictor",
  "name": "predictor",
  "config": {
    "device": "cuda",
    "save_predictions": "true",
    "output_dir": "results/predictions",
    "output_format": "nifti",
    "apply_tta": "false"
  }
}
```

**Parameters**:
- `device`: cuda or cpu
- `save_predictions`: Save predicted masks
- `output_dir`: Where to save predictions
- `output_format`: nifti, png, npy
- `apply_tta`: Test-time augmentation

### Test-Time Augmentation (TTA)

**Description**: Apply augmentations at test time and average predictions.

**Benefits**:
- Improved accuracy (typically 1-2% Dice improvement)
- More robust predictions
- Reduced variance

**Augmentations Used**:
- Horizontal/vertical flips
- Rotations (90°, 180°, 270°)
- Average predictions

**Configuration**:
```json
{
  "apply_tta": "true",
  "tta_transforms": ["flip_h", "flip_v", "rotate_90"]
}
```

**Trade-off**:
- Better accuracy
- 4-8× slower inference

---

## Visualization Techniques

### 1. Overlay Visualization

**Description**: Overlay predicted mask on original image.

**Code**:
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_overlay(image, mask_gt, mask_pred, alpha=0.5):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image with ground truth
    axes[0].imshow(image, cmap='gray')
    axes[0].imshow(mask_gt, cmap='Reds', alpha=alpha)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Original image with prediction
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask_pred, cmap='Blues', alpha=alpha)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Comparison (GT=Red, Pred=Blue, Overlap=Purple)
    overlay = np.zeros((*image.shape, 3))
    overlay[mask_gt == 1] = [1, 0, 0]  # Red for GT
    overlay[mask_pred == 1] = [0, 0, 1]  # Blue for Pred
    overlap = (mask_gt == 1) & (mask_pred == 1)
    overlay[overlap] = [0.5, 0, 0.5]  # Purple for overlap

    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(overlay, alpha=alpha)
    axes[2].set_title('Comparison')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
```

### 2. Side-by-Side Comparison

**Description**: Display image, ground truth, and prediction side by side.

**Best for**: Detailed comparison, publications

```python
def visualize_comparison(image, mask_gt, mask_pred):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')

    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title('Ground Truth')

    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Prediction')

    # Difference (error map)
    error = np.abs(mask_gt - mask_pred)
    axes[3].imshow(error, cmap='hot')
    axes[3].set_title('Error Map')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

### 3. Contour Visualization

**Description**: Show boundaries of predicted and ground truth masks.

**Best for**: Boundary evaluation, clinical review

```python
from skimage import measure

def visualize_contours(image, mask_gt, mask_pred):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image, cmap='gray')

    # Ground truth contours (green)
    contours_gt = measure.find_contours(mask_gt, 0.5)
    for contour in contours_gt:
        ax.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=2, label='GT')

    # Prediction contours (red)
    contours_pred = measure.find_contours(mask_pred, 0.5)
    for contour in contours_pred:
        ax.plot(contour[:, 1], contour[:, 0], 'r--', linewidth=2, label='Pred')

    ax.axis('off')
    ax.legend()
    plt.title('Contour Comparison')
    plt.show()
```

### 4. 3D Volume Visualization

**Description**: Interactive 3D visualization for volumetric data.

**Tools**:
- **ITK-SNAP**: Manual inspection and editing
- **3D Slicer**: Comprehensive visualization
- **matplotlib 3D**: Quick programmatic viz

```python
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d(volume, threshold=0.5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh grid
    z, y, x = (volume > threshold).nonzero()
    ax.scatter(x, y, z, c='b', marker='o', s=1, alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Volume Rendering')
    plt.show()
```

### 5. Multi-Slice Visualization

**Description**: Show multiple slices from 3D volume.

**Best for**: Quick overview of 3D predictions

```python
def visualize_volume_slices(volume_img, volume_gt, volume_pred, num_slices=9):
    indices = np.linspace(0, volume_img.shape[0]-1, num_slices, dtype=int)

    fig, axes = plt.subplots(3, num_slices, figsize=(20, 8))

    for i, idx in enumerate(indices):
        # Image
        axes[0, i].imshow(volume_img[idx], cmap='gray')
        axes[0, i].set_title(f'Slice {idx}')
        axes[0, i].axis('off')

        # Ground truth
        axes[1, i].imshow(volume_gt[idx], cmap='gray')
        axes[1, i].axis('off')

        # Prediction
        axes[2, i].imshow(volume_pred[idx], cmap='gray')
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Image', size=14)
    axes[1, 0].set_ylabel('Ground Truth', size=14)
    axes[2, 0].set_ylabel('Prediction', size=14)

    plt.tight_layout()
    plt.show()
```

### 6. Uncertainty Visualization

**Description**: Visualize model uncertainty (for ensemble or dropout-based uncertainty).

```python
def visualize_uncertainty(image, pred_mean, pred_std):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')

    axes[1].imshow(pred_mean, cmap='gray')
    axes[1].set_title('Mean Prediction')

    axes[2].imshow(pred_std, cmap='hot')
    axes[2].set_title('Uncertainty (Std Dev)')
    axes[2].colorbar()

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## Performance Analysis

### 1. Per-Case Analysis

**Purpose**: Identify which cases the model struggles with.

**Output**:
```
Case ID    | Dice  | IoU   | HD95  | ASD
-----------|-------|-------|-------|-------
case_001   | 0.92  | 0.85  | 2.1   | 0.8
case_002   | 0.88  | 0.78  | 3.5   | 1.2
case_003   | 0.95  | 0.90  | 1.8   | 0.6
case_004   | 0.65  | 0.48  | 12.3  | 5.4  ← Poor case
...
Mean       | 0.87  | 0.77  | 4.2   | 1.8
Std        | 0.08  | 0.11  | 2.5   | 1.1
```

**Analysis**:
1. Sort by metric (ascending)
2. Inspect worst cases
3. Identify patterns (size, quality, anatomy)

### 2. Class-Wise Metrics

**For multi-class segmentation**:

```
Class        | Dice  | IoU   | Sensitivity | Specificity
-------------|-------|-------|-------------|-------------
Background   | 0.99  | 0.98  | 0.99        | 0.99
Organ 1      | 0.92  | 0.85  | 0.94        | 0.98
Organ 2      | 0.88  | 0.78  | 0.87        | 0.99
Tumor        | 0.72  | 0.56  | 0.68        | 0.99  ← Difficult
```

**Insights**:
- Small structures typically have lower Dice
- High specificity, low sensitivity = model is conservative
- Low specificity, high sensitivity = model over-predicts

### 3. Error Analysis

**Types of Errors**:

1. **False Positives (FP)**: Predicted but not in ground truth
   - Over-segmentation
   - May indicate artifacts or unclear boundaries

2. **False Negatives (FN)**: In ground truth but not predicted
   - Under-segmentation
   - Missing small structures

3. **Boundary Errors**: Predicted mask shifted from ground truth
   - High HD, low Dice
   - May need post-processing

**Visualization**:
```python
def error_analysis(mask_gt, mask_pred):
    TP = (mask_gt == 1) & (mask_pred == 1)
    FP = (mask_gt == 0) & (mask_pred == 1)
    FN = (mask_gt == 1) & (mask_pred == 0)
    TN = (mask_gt == 0) & (mask_pred == 0)

    error_map = np.zeros((*mask_gt.shape, 3))
    error_map[TP] = [0, 1, 0]    # Green: True Positive
    error_map[FP] = [1, 0, 0]    # Red: False Positive
    error_map[FN] = [0, 0, 1]    # Blue: False Negative

    plt.imshow(error_map)
    plt.title('Error Map (TP=Green, FP=Red, FN=Blue)')
    plt.axis('off')
    plt.show()
```

### 4. Confusion Matrix

**For multi-class segmentation**:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
```

### 5. Statistical Significance Testing

**Comparing two models**:

```python
from scipy import stats

def compare_models(scores_model1, scores_model2):
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_model1, scores_model2)

    print(f"Model 1 mean: {np.mean(scores_model1):.3f} ± {np.std(scores_model1):.3f}")
    print(f"Model 2 mean: {np.mean(scores_model2):.3f} ± {np.std(scores_model2):.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Difference is statistically significant")
    else:
        print("Difference is not statistically significant")
```

---

## Clinical Validation

### 1. Clinical Metrics

Beyond technical metrics, consider:

**Volume Metrics**:
```python
def volume_metrics(mask_gt, mask_pred, voxel_volume):
    vol_gt = mask_gt.sum() * voxel_volume
    vol_pred = mask_pred.sum() * voxel_volume

    vol_diff = abs(vol_gt - vol_pred)
    vol_error_pct = (vol_diff / vol_gt) * 100

    print(f"GT Volume: {vol_gt:.2f} mm³")
    print(f"Predicted Volume: {vol_pred:.2f} mm³")
    print(f"Volume Error: {vol_error_pct:.2f}%")
```

**Clinical Acceptability**:
- Expert review of predictions
- Clinical use case requirements
- Safety considerations

### 2. Expert Review Workflow

1. **Random Sampling**: Select representative cases
2. **Blinded Review**: Expert doesn't know which is prediction
3. **Rating Scale**: 1-5 quality score
4. **Agreement Metrics**: Inter-rater reliability

### 3. Failure Case Analysis

**Document**:
- Image quality issues
- Pathological variations
- Edge cases
- Systematic errors

**Example Report**:
```
Failure Case: case_042
Dice: 0.52 (below threshold)

Observations:
- Severe motion artifacts
- Unusual anatomy (post-surgical)
- Poor image quality

Recommendation:
- Add quality control checks
- Expand training data with similar cases
```

---

## Common Issues

### Issue 1: Good Training Metrics, Poor Test Metrics

**Cause**: Overfitting or dataset shift

**Solutions**:
1. Check train/test data distribution
2. Review data augmentation
3. Increase regularization
4. Collect more diverse data

### Issue 2: High Dice but Poor Visual Quality

**Cause**: Metric-target mismatch

**Solutions**:
1. Check boundary quality (use HD95, ASD)
2. Add boundary loss
3. Post-processing (morphological operations)

### Issue 3: Inconsistent Predictions

**Cause**: Model uncertainty or preprocessing issues

**Solutions**:
1. Ensemble multiple models
2. Use test-time augmentation
3. Check preprocessing consistency

### Issue 4: Poor Performance on Small Structures

**Cause**: Class imbalance, resolution limitations

**Solutions**:
1. Use weighted or focal loss
2. Crop/patch around region of interest
3. Multi-scale training
4. Higher resolution

---

## Example Results Report

### Model Evaluation Report

**Model**: Swin-UNETR
**Dataset**: Brain Tumor Segmentation (150 test cases)
**Date**: 2024-01-31

#### Quantitative Results

| Metric | Mean ± Std | Median | Min | Max |
|--------|------------|--------|-----|-----|
| Dice   | 0.87 ± 0.08 | 0.89 | 0.52 | 0.96 |
| IoU    | 0.77 ± 0.11 | 0.80 | 0.35 | 0.92 |
| HD95   | 4.2 ± 2.5 mm | 3.8 | 1.2 | 15.3 |
| ASD    | 1.8 ± 1.1 mm | 1.5 | 0.4 | 6.2 |

#### Per-Class Results

| Class | Dice | Sensitivity | Specificity |
|-------|------|-------------|-------------|
| Tumor Core | 0.89 | 0.91 | 0.99 |
| Edema | 0.85 | 0.87 | 0.98 |

#### Failure Cases

- 5 cases with Dice < 0.6
- Common issues: Motion artifacts (3), unusual anatomy (2)

#### Recommendations

1. Add motion artifact detection
2. Expand training data with edge cases
3. Consider ensemble for improved robustness

---

## Next Steps

- [Network Architectures](NETWORK_ARCHITECTURES.md) - Choose different models
- [Training Guide](TRAINING.md) - Improve training for better results
- [Data Loading](DATALOADER.md) - Prepare better test data
- [Example Workflows](../examples/medical-segmentation/) - Complete testing pipelines

---

## Useful Tools

### Visualization Libraries
- **matplotlib**: Standard Python plotting
- **ITK-SNAP**: Interactive medical image viewer
- **3D Slicer**: Comprehensive medical imaging platform
- **napari**: N-dimensional array viewer

### Metric Computation
- **scikit-image**: Image processing and metrics
- **SimpleITK**: Medical image metrics (HD, ASD)
- **MedPy**: Medical image processing metrics
- **MONAI**: Medical imaging metrics library

### Statistical Analysis
- **scipy.stats**: Statistical tests
- **pandas**: Data organization and analysis
- **seaborn**: Statistical visualization
