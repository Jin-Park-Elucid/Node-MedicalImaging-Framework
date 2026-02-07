# Data Loading Guide for Medical Image Segmentation

This guide covers everything you need to know about preparing, loading, and augmenting medical imaging data for segmentation tasks.

## Table of Contents

1. [Overview](#overview)
2. [Data Organization](#data-organization)
3. [Data Loader Node](#data-loader-node)
4. [Supported Formats](#supported-formats)
5. [Data Augmentation](#data-augmentation)
6. [Best Practices](#best-practices)
7. [Common Issues](#common-issues)

---

## Overview

The `MedicalSegmentationLoader` node handles loading medical images and their corresponding segmentation masks for training and evaluation. It supports various medical imaging formats and includes built-in augmentation capabilities.

**Key Features**:
- Support for 2D and 3D medical images
- Multiple file format support (NIfTI, DICOM, PNG, NPY)
- Built-in data augmentation
- Automatic train/validation/test splitting
- Memory-efficient batch loading
- Multi-threaded data loading

---

## Data Organization

### Directory Structure

The framework expects data to be organized in one of two structures:

#### Structure 1: Separate Image and Mask Folders

```
data/medical_segmentation/
├── images/
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   ├── case003.nii.gz
│   └── ...
└── masks/
    ├── case001.nii.gz
    ├── case002.nii.gz
    ├── case003.nii.gz
    └── ...
```

**Requirements**:
- Image and mask filenames must match exactly
- Both folders must have the same number of files
- Files are automatically paired by name

#### Structure 2: Case-Based Folders

```
data/medical_segmentation/
├── case001/
│   ├── image.nii.gz
│   └── mask.nii.gz
├── case002/
│   ├── image.nii.gz
│   └── mask.nii.gz
├── case003/
│   ├── image.nii.gz
│   └── mask.nii.gz
└── ...
```

**Requirements**:
- Each case has its own folder
- Standard filenames: `image.nii.gz` and `mask.nii.gz`
- Or custom filenames specified in configuration

---

## Data Loader Node

### Node Configuration

The `MedicalSegmentationLoader` node can be configured in the GUI or JSON workflow:

#### GUI Configuration

```json
{
  "type": "MedicalSegmentationLoader",
  "name": "data_loader",
  "config": {
    "data_dir": "data/medical_segmentation",
    "batch_size": "4",
    "image_size": "256",
    "num_workers": "4",
    "train_split": "0.7",
    "val_split": "0.15",
    "test_split": "0.15",
    "normalize": "true",
    "augmentation": "true"
  }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | string | required | Path to dataset directory |
| `batch_size` | int | 4 | Number of samples per batch |
| `image_size` | int/tuple | 256 | Target image size (resized if needed) |
| `num_workers` | int | 4 | Number of data loading threads |
| `train_split` | float | 0.7 | Fraction of data for training |
| `val_split` | float | 0.15 | Fraction of data for validation |
| `test_split` | float | 0.15 | Fraction of data for testing |
| `normalize` | bool | true | Normalize intensities to [0, 1] |
| `augmentation` | bool | true | Apply data augmentation |
| `shuffle` | bool | true | Shuffle data each epoch |
| `seed` | int | 42 | Random seed for reproducibility |

### Output Ports

The data loader provides three output ports:

1. **`train_loader`**: Training data iterator
   - Batches of images and masks for training
   - Augmentation applied if enabled

2. **`val_loader`**: Validation data iterator
   - Batches of images and masks for validation
   - No augmentation applied

3. **`test_loader`**: Test data iterator
   - Batches of images and masks for testing
   - No augmentation applied

---

## Supported Formats

### 3D Medical Imaging Formats

#### NIfTI (.nii, .nii.gz)

**Description**: Standard format for neuroimaging and medical research.

**Advantages**:
- Compact compression (.nii.gz)
- Preserves spatial metadata
- Wide software support
- Standard for many medical imaging tasks

**Usage**:
```python
# Files automatically detected by extension
images/scan001.nii.gz
masks/scan001.nii.gz
```

**Typical Use Cases**:
- MRI scans
- CT scans
- Brain imaging
- Organ segmentation

#### DICOM (.dcm)

**Description**: Standard format for medical imaging equipment.

**Advantages**:
- Industry standard
- Rich metadata (patient info, acquisition parameters)
- Direct from scanners

**Usage**:
```
# DICOM series organized by case
case001/
  ├── image_series/
  │   ├── IM0001.dcm
  │   ├── IM0002.dcm
  │   └── ...
  └── mask_series/
      ├── SEG0001.dcm
      └── ...
```

**Note**: DICOM series are automatically stacked into 3D volumes.

#### NumPy Arrays (.npy, .npz)

**Description**: Preprocessed arrays saved from Python.

**Advantages**:
- Fast loading
- No conversion needed
- Good for preprocessed data

**Usage**:
```python
# Save preprocessed data
np.save('image.npy', preprocessed_image)
np.save('mask.npy', preprocessed_mask)
```

### 2D Image Formats

#### PNG (.png)

**Description**: 2D slices or single-slice images.

**Usage**:
```
images/
  ├── slice001.png
  ├── slice002.png
  └── ...
masks/
  ├── slice001.png
  ├── slice002.png
  └── ...
```

**Typical Use Cases**:
- X-ray images
- Microscopy
- 2D slice-based training

---

## Data Augmentation

### Built-in Augmentations

When `augmentation: true` is set, the following augmentations are applied during training:

#### Spatial Augmentations

1. **Random Rotation**
   - Range: ±15 degrees
   - Applied to both image and mask
   - Preserves anatomical structure

2. **Random Flip**
   - Horizontal and vertical flipping
   - 50% probability each
   - Useful for symmetry

3. **Random Affine**
   - Translation: ±10% of image size
   - Scaling: 0.9-1.1×
   - Shearing: ±5 degrees

4. **Elastic Deformation**
   - Simulates tissue deformation
   - Controlled by alpha and sigma parameters
   - Realistic medical image variations

#### Intensity Augmentations

1. **Random Brightness**
   - Range: ±20%
   - Simulates different scanner settings

2. **Random Contrast**
   - Range: 0.8-1.2×
   - Handles varying image quality

3. **Gaussian Noise**
   - Small amount of noise
   - Improves robustness

4. **Gaussian Blur**
   - Simulates different resolutions
   - Random kernel size 3-7

### Custom Augmentation

You can define custom augmentation pipelines:

```python
# In custom data loader
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
    # Add your custom transforms
])
```

### Augmentation Recommendations

**Small Dataset (< 50 cases)**:
- Enable all augmentations
- Use stronger parameters
- Consider MixUp or CutMix

**Medium Dataset (50-200 cases)**:
- Standard augmentation pipeline
- Moderate parameters
- Focus on spatial transforms

**Large Dataset (> 200 cases)**:
- Lighter augmentation
- Focus on intensity variations
- May not need all augmentations

---

## Best Practices

### 1. Data Preprocessing

#### Intensity Normalization

**Z-score Normalization** (recommended for CT):
```python
mean = image.mean()
std = image.std()
image_normalized = (image - mean) / (std + 1e-8)
```

**Min-Max Normalization** (recommended for MRI):
```python
image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
```

**Percentile Clipping** (for outliers):
```python
p1, p99 = np.percentile(image, [1, 99])
image_clipped = np.clip(image, p1, p99)
image_normalized = (image_clipped - p1) / (p99 - p1)
```

#### Resampling

For consistent voxel spacing:
```python
# Resample to 1mm³ isotropic
from scipy.ndimage import zoom

zoom_factors = original_spacing / target_spacing  # e.g., [2.0, 1.0, 1.0]
resampled = zoom(image, zoom_factors, order=3)  # cubic interpolation
```

### 2. Memory Management

**For Large 3D Volumes**:

1. **Patch-Based Training**:
   ```json
   {
     "patch_size": "96",
     "samples_per_volume": "4",
     "overlap": "0.5"
   }
   ```

2. **Reduce Batch Size**:
   - 3D: batch_size = 1-2
   - 2D: batch_size = 4-8

3. **Use Mixed Precision**:
   ```json
   {
     "mixed_precision": "true"
   }
   ```

4. **Gradient Checkpointing**:
   - Trades computation for memory
   - Enables larger models/batches

### 3. Data Splitting

**Random Split** (default):
```json
{
  "train_split": "0.7",
  "val_split": "0.15",
  "test_split": "0.15",
  "shuffle": "true",
  "seed": "42"
}
```

**Stratified Split** (for imbalanced data):
- Ensures equal distribution of classes
- Important for rare diseases

**Patient-Based Split**:
- Never split slices from same patient
- Prevents data leakage
- More realistic evaluation

### 4. Class Imbalance

**Weighted Loss**:
```python
# Calculate class weights
background_pixels = (masks == 0).sum()
foreground_pixels = (masks == 1).sum()
total = background_pixels + foreground_pixels

weight_background = total / (2 * background_pixels)
weight_foreground = total / (2 * foreground_pixels)

weights = [weight_background, weight_foreground]
```

**Focal Loss**:
- Automatically handles imbalance
- Focuses on hard examples

**Oversampling**:
- Sample positive cases more frequently
- Use weighted random sampler

---

## Common Issues

### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size
2. Reduce image size
3. Use gradient checkpointing
4. Enable mixed precision training
5. Use patch-based training for 3D

**Example**:
```json
{
  "batch_size": "1",
  "image_size": "96",
  "mixed_precision": "true"
}
```

### Issue 2: Slow Data Loading

**Symptoms**: GPU underutilized, slow training

**Solutions**:
1. Increase `num_workers`
2. Use SSD for data storage
3. Preprocess data to .npy format
4. Reduce augmentation complexity
5. Enable pin_memory

**Example**:
```json
{
  "num_workers": "8",
  "pin_memory": "true"
}
```

### Issue 3: Label Mismatch

**Symptoms**: Labels don't match predictions

**Solutions**:
1. Verify mask values are in correct range
2. Check mask and image alignment
3. Ensure correct number of classes
4. Verify data pairing

**Verification**:
```python
# Check mask values
unique_values = np.unique(mask)
print(f"Mask values: {unique_values}")  # Should be [0, 1] for binary

# Check alignment
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1); plt.imshow(image[0])
plt.subplot(1, 2, 2); plt.imshow(mask[0])
plt.show()
```

### Issue 4: Poor Augmentation

**Symptoms**: Unrealistic augmented images

**Solutions**:
1. Reduce augmentation strength
2. Check augmentation parameters
3. Visualize augmented samples
4. Disable problematic augmentations

**Debugging**:
```python
# Visualize augmented batches
for images, masks in train_loader:
    plt.subplot(1, 2, 1); plt.imshow(images[0, 0])
    plt.subplot(1, 2, 2); plt.imshow(masks[0, 0])
    plt.show()
    break
```

### Issue 5: Data Imbalance

**Symptoms**: Model predicts only background

**Solutions**:
1. Use weighted loss functions
2. Use Focal Loss
3. Oversample positive cases
4. Check class distribution

**Check Distribution**:
```python
# Calculate class distribution
background_ratio = (masks == 0).sum() / masks.numel()
foreground_ratio = (masks == 1).sum() / masks.numel()
print(f"Background: {background_ratio:.2%}")
print(f"Foreground: {foreground_ratio:.2%}")

# If > 95% background, use weighted loss
```

---

## Example Configurations

### Configuration 1: 2D CT Scan Segmentation

```json
{
  "type": "MedicalSegmentationLoader",
  "name": "ct_loader",
  "config": {
    "data_dir": "data/ct_scans",
    "batch_size": "8",
    "image_size": "512",
    "num_workers": "4",
    "normalize": "true",
    "normalization_type": "percentile",
    "augmentation": "true",
    "train_split": "0.7",
    "val_split": "0.15",
    "test_split": "0.15"
  }
}
```

### Configuration 2: 3D MRI Brain Segmentation

```json
{
  "type": "MedicalSegmentationLoader",
  "name": "mri_loader",
  "config": {
    "data_dir": "data/brain_mri",
    "batch_size": "2",
    "image_size": "128",
    "patch_size": "96",
    "num_workers": "4",
    "normalize": "true",
    "normalization_type": "zscore",
    "augmentation": "true",
    "train_split": "0.8",
    "val_split": "0.1",
    "test_split": "0.1"
  }
}
```

### Configuration 3: Limited Data Scenario

```json
{
  "type": "MedicalSegmentationLoader",
  "name": "small_dataset_loader",
  "config": {
    "data_dir": "data/small_dataset",
    "batch_size": "4",
    "image_size": "256",
    "num_workers": "2",
    "normalize": "true",
    "augmentation": "true",
    "augmentation_strength": "strong",
    "train_split": "0.8",
    "val_split": "0.2",
    "test_split": "0.0"
  }
}
```

---

## Next Steps

- [Network Architectures](NETWORK_ARCHITECTURES.md) - Choose the right model
- [Training Guide](TRAINING.md) - Train your segmentation model
- [Testing and Visualization](TESTING_VISUALIZATION.md) - Evaluate results
- [Example Workflows](../examples/medical-segmentation/) - Complete pipelines

---

## Additional Resources

### Data Preparation Tools

1. **Medical Imaging Toolkits**:
   - **ITK-SNAP**: Manual segmentation and viewing
   - **3D Slicer**: Image processing and analysis
   - **MITK**: Medical imaging toolkit
   - **SimpleITK**: Python medical image processing

2. **Data Conversion**:
   - **dcm2niix**: DICOM to NIfTI conversion
   - **nibabel**: Python NIfTI handling
   - **pydicom**: Python DICOM handling

3. **Public Datasets**:
   - **Medical Segmentation Decathlon**: 10 medical segmentation tasks
   - **BRATS**: Brain tumor segmentation
   - **LiTS**: Liver tumor segmentation
   - **KiTS**: Kidney tumor segmentation
   - **CHAOS**: Multi-organ segmentation

### Useful Commands

```bash
# Check data statistics
python -c "import nibabel as nib; import numpy as np; \
img = nib.load('image.nii.gz').get_fdata(); \
print(f'Shape: {img.shape}'); \
print(f'Range: [{img.min():.2f}, {img.max():.2f}]'); \
print(f'Mean: {img.mean():.2f}, Std: {img.std():.2f}')"

# Convert DICOM to NIfTI
dcm2niix -o output_dir -f scan_name input_dicom_dir/

# Verify data integrity
python scripts/verify_dataset.py --data_dir data/medical_segmentation
```
