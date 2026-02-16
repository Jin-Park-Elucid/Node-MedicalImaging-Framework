# Comparison: Raw NIfTI Data vs. Preprocessed nnUNet Data

## File Overview

### Original Raw Data (NIfTI format)
```
/data/avte_training/nnUNet_raw/Dataset006_model_9_4/
├── imagesTr/00001MTY_0000.nii.gz  (~125 MB compressed, ~198 MB uncompressed)
└── labelsTr/00001MTY.nii.gz       (~166 KB compressed)
```

### Preprocessed Data (NumPy format)
```
/data/avte_training/nnUNet_preprocessed/Dataset006_model_9_4/nnUNetPlans_3d_fullres_224_size_0_25_spacing/
├── 00001MTY.npy        (~1.1 GB) - Preprocessed image
├── 00001MTY_seg.npy    (~278 MB) - Preprocessed segmentation
├── 00001MTY.npz        (~473 MB) - Compressed archive (contains both above)
└── 00001MTY.pkl        (~1.3 MB) - Metadata for reversing transformations
```

## Detailed Comparison

### 1. **File Format**

| Aspect | Raw Data | Preprocessed Data |
|--------|----------|-------------------|
| **Format** | NIfTI (.nii.gz) | NumPy (.npy/.npz/.pkl) |
| **Standard** | Medical imaging standard | ML framework compatible |
| **Compression** | Gzip compressed | Uncompressed (.npy) or deflate compressed (.npz) |
| **Metadata** | Embedded in NIfTI header | Separate .pkl file |

**Why the change?**
- NumPy arrays are faster to load during training (no decompression overhead for .npy)
- Direct memory mapping possible
- Better integration with PyTorch/TensorFlow

---

### 2. **Dimensions and Shape**

| Property | Raw Data | Preprocessed Data |
|----------|----------|-------------------|
| **Shape** | (Z, Y, X) - 3D volume | **(1, 455, 800, 800)** = (C, Z, Y, X) |
| **Dimensions** | 3D medical image | 4D tensor with channel dimension |
| **Size increase** | Original dimensions | Resampled to 224mm³ voxel size at 0.25 spacing |

From header inspection:
- **Raw .npy shape**: `(1, 455, 800, 800)`
- **Raw _seg.npy shape**: `(1, 455, 800, 800)`

**Transformations applied:**
1. **Resampling**: Changed voxel spacing to target resolution (0.25mm isotropic based on folder name)
2. **Cropping**: Removed empty background space
3. **Padding**: Ensured dimensions are compatible with network architecture
4. **Channel addition**: Added channel dimension (first axis) for multi-modal support

---

### 3. **Data Type and Precision**

| Data Type | Raw Image | Raw Segmentation | Preprocessed Image | Preprocessed Seg |
|-----------|-----------|------------------|-------------------|------------------|
| **Dtype** | float64 (typical) | uint8/int16 | **float32** (`<f4`) | **int8** (`\|i1`) |
| **Bytes/voxel** | 8 bytes | 1-2 bytes | 4 bytes | 1 byte |
| **Memory** | Higher precision | Integer labels | Reduced precision | Compact labels |

**Why the change?**
- float32 sufficient for neural networks (halves memory vs float64)
- int8 for segmentation masks (supports up to 127 classes, saves memory)

---

### 4. **Value Ranges and Normalization**

| Aspect | Raw Data | Preprocessed Data |
|--------|----------|-------------------|
| **Image values** | Original HU values (CT: typically -1024 to +3071) | **Z-score normalized** (mean≈0, std≈1) |
| **Segmentation** | Integer class labels | Same integer labels (preserved) |
| **Intensity** | Scanner-dependent | Standardized across dataset |

**nnUNet normalization:**
```python
# Typical preprocessing:
# 1. Clip to foreground percentiles (0.5, 99.5)
# 2. Z-score normalization: (x - mean) / std
# Result: normalized values typically in range [-3, 3]
```

---

### 5. **Spacing and Resolution**

Based on folder name: `nnUNetPlans_3d_fullres_224_size_0_25_spacing`

| Property | Raw Data | Preprocessed Data |
|----------|----------|-------------------|
| **Spacing** | Original scanner spacing (variable) | **0.25mm isotropic** (resampled) |
| **Target size** | Original dimensions | 224mm³ patch size |
| **Resolution** | Anisotropic (different X,Y,Z) | Isotropic (same in all directions) |

**Why resample?**
- Standardize resolution across different scanners
- Isotropic voxels work better with 3D CNNs
- Target spacing chosen based on dataset statistics

---

### 6. **Storage Size Comparison**

#### Raw Data:
```
Image:  125 MB (compressed) → ~198 MB (uncompressed)
Label:  166 KB (compressed) → ~few MB (uncompressed)
Total:  ~125 MB (compressed) → ~200 MB (uncompressed)
```

#### Preprocessed Data:
```
.npy files (uncompressed):
  - Image:  1.1 GB (1,164,800,128 bytes)
  - Seg:    278 MB (291,200,128 bytes)
  - Total:  ~1.4 GB

.npz (compressed):
  - Archive: 473 MB (contains both image and seg)

.pkl: 1.3 MB (metadata)
```

**Storage analysis:**
- **5.6x size increase** from raw (200 MB) to preprocessed uncompressed (1.4 GB)
- **2.4x size increase** when using compressed .npz (473 MB vs 200 MB)

**Why so much larger?**
1. Resampling to higher resolution (0.25mm spacing)
2. No compression in .npy files (trading space for speed)
3. Padding to network-compatible dimensions

---

### 7. **Metadata Storage**

#### Raw NIfTI Header (embedded):
- Voxel spacing (qform/sform)
- Origin coordinates
- Orientation matrix (RAS/LAS/etc.)
- Data type and dimensions
- Scanner information

#### Preprocessed .pkl (separate file):
```python
{
    'spacing': array([0.25, 0.25, 0.25]),  # New spacing after resampling
    'origin': array([x, y, z]),             # Image origin
    'direction': array([...]),              # Orientation matrix
    'shape_before_cropping': (z, y, x),    # Original shape
    'bbox_used_for_cropping': [[z1,y1,x1], [z2,y2,x2]],
    'shape_after_cropping_and_before_resampling': (z, y, x),
    'class_locations': {...}                # Where each class appears
}
```

**Why separate metadata?**
- NumPy arrays don't have headers like NIfTI
- Need transformation info to reverse preprocessing during inference
- Enables proper post-processing and metrics calculation

---

### 8. **Usage in Training Pipeline**

#### Raw Data (nnUNet preprocessing):
```python
# Load once, preprocess, save
image = nib.load('00001MTY_0000.nii.gz').get_fdata()
label = nib.load('00001MTY.nii.gz').get_fdata()
# → crop, resample, normalize, save as .npy
```

#### Preprocessed Data (training loop):
```python
# Fast loading during training
image = np.load('00001MTY.npy')          # Shape: (1, 455, 800, 800)
label = np.load('00001MTY_seg.npy')      # Shape: (1, 455, 800, 800)
# → ready for network input (no preprocessing needed)
```

**Speed advantage:**
- .npy loading: ~0.1-0.5 seconds
- NIfTI loading + preprocessing: ~5-10 seconds
- **10-100x speedup during training**

---

## Summary of Key Differences

| Feature | Raw NIfTI | Preprocessed NumPy |
|---------|-----------|-------------------|
| **Format** | .nii.gz (medical standard) | .npy/.npz (ML standard) |
| **Compression** | Gzip | None (.npy) or deflate (.npz) |
| **Dimensions** | (Z, Y, X) | (C, Z, Y, X) |
| **Dtype** | float64 / int16 | float32 / int8 |
| **Spacing** | Variable (scanner-specific) | 0.25mm isotropic |
| **Values** | Original HU/intensity | Z-score normalized |
| **Size** | ~125 MB compressed | ~1.4 GB uncompressed, ~473 MB compressed |
| **Load speed** | Slower (decompress + parse) | Faster (memory map) |
| **Metadata** | Embedded in header | Separate .pkl file |
| **Use case** | Clinical/research storage | Neural network training |

---

## When to Use Each Format

### Use Raw NIfTI when:
- Clinical review or diagnosis
- Working with medical imaging software (3D Slicer, ITK-SNAP)
- Long-term archival storage
- Interoperability with other medical imaging tools

### Use Preprocessed NumPy when:
- Training deep learning models
- Need fast I/O during training
- Working with PyTorch/TensorFlow
- Batch processing for inference

---

## File Size Breakdown

```
Case: 00001MTY

Raw (compressed):          125 MB
Raw (uncompressed est.):  ~200 MB
Preprocessed .npy:        1,380 MB  (1.1GB + 278MB)
Preprocessed .npz:         473 MB   (compressed)
Preprocessed .pkl:          1.3 MB  (metadata)

Total preprocessed storage:
  - If keeping .npy files:  1,381 MB
  - If keeping .npz only:     474 MB
  - You can delete .npy after training and keep .npz for archival
```

---

## nnUNet Preprocessing Pipeline

```
Raw NIfTI                    Preprocessed NumPy
├── Load .nii.gz         →   ├── Crop non-zero region
├── Original spacing     →   ├── Resample to 0.25mm isotropic
├── Variable dimensions  →   ├── Pad to (1, 455, 800, 800)
├── Original HU values   →   ├── Z-score normalize
├── float64 dtype        →   ├── Convert to float32
└── Gzip compressed      →   └── Save as .npy (fast) + .npz (space-efficient)
                              └── Save metadata as .pkl
```

The preprocessing ensures:
1. **Consistent resolution** across all images
2. **Normalized intensities** for stable training
3. **Fast loading** during training iterations
4. **Reversible transformations** via metadata for proper inference
