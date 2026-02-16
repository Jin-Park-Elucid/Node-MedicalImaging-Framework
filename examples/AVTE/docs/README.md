# AVTE 2D Segmentation Preprocessing

This directory contains scripts for preprocessing 3D NIfTI volumes into 2D slices with multi-slice context for training 2D segmentation networks.

## Overview

The preprocessing pipeline converts 3D medical imaging volumes into 2D slices where each slice includes adjacent slices as input channels. This provides spatial context for 2D networks while being more memory-efficient than 3D networks.

### Key Features

- **Multi-slice windows**: Include adjacent slices as input channels
- **Flexible border handling**: Zero padding, replication, or mirroring
- **Z-score normalization**: Per-volume intensity normalization
- **Efficient storage**: Compressed .npz format per slice
- **Train/Val/Test splits**: Automatic dataset splitting
- **Metadata preservation**: All preprocessing info saved with each slice
- **Multiprocessing support**: 3-7x faster with parallel processing (NEW!)

## Files

- `preprocess_2d_slices.py` - Main preprocessing script (with multiprocessing)
- `avte_dataloader.py` - PyTorch Dataset and DataLoader
- `example_usage.py` - Example training script
- `README.md` - This file
- `MULTIPROCESSING_GUIDE.md` - Detailed guide on parallel processing
- `GETTING_STARTED.md` - Quick start guide
- `QUICK_REFERENCE.md` - Command cheat sheet
- `PATH_CONFIGURATION.md` - Path setup documentation

## Quick Start

### 1. Preprocess the Data

```bash
python preprocess_2d_slices.py \
    --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data \
    --window_size 2 \
    --padding_mode replicate \
    --num_workers 8
```

**Multiprocessing**: Use `--num_workers 8` for 3-7x faster preprocessing (default: 4). Use `-1` to auto-detect CPU count.

**Note**: Data splitting (train/val/test) is now handled by the dataloader, not preprocessing!

### 2. Use the DataLoader

```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    batch_size=16,
    num_workers=4
)

# Training loop
for images, labels in train_loader:
    # images shape: (B, C, H, W) where C = 2*window_size + 1
    # labels shape: (B, H, W)
    outputs = model(images)
    loss = criterion(outputs, labels)
    ...
```

## Preprocessing Options

### Window Size

The `--window_size` parameter controls how many adjacent slices are included as channels:

- `window_size=0`: Single slice (1 channel)
- `window_size=1`: 3 channels (prev, current, next)
- `window_size=2`: 5 channels (prev2, prev1, current, next1, next2)
- `window_size=N`: 2N+1 channels

**Example**: For window_size=2, the center slice at index 100 will include:
```
Channel 0: Slice 98
Channel 1: Slice 99
Channel 2: Slice 100 (center)
Channel 3: Slice 101
Channel 4: Slice 102
```

### Border Handling

The `--padding_mode` parameter controls how to handle slices at volume boundaries:

#### 1. Zero Padding (`--padding_mode zero`)
Missing slices are filled with zeros.
```
Volume: [0, 1, 2, ..., N-1]
Slice 0 with window=1:
  Channel 0: zeros (no slice -1)
  Channel 1: slice 0
  Channel 2: slice 1
```

#### 2. Replication (`--padding_mode replicate`) [Default]
First/last slice is duplicated.
```
Slice 0 with window=1:
  Channel 0: slice 0 (replicated)
  Channel 1: slice 0
  Channel 2: slice 1
```

#### 3. Mirroring (`--padding_mode mirror`)
Slices are mirrored at boundaries.
```
Slice 0 with window=1:
  Channel 0: slice 1 (mirrored)
  Channel 1: slice 0
  Channel 2: slice 1
```

### Normalization

By default, z-score normalization is applied per volume:
```python
normalized = (volume - mean) / std
```

- Statistics computed on foreground (non-zero) voxels
- Can be disabled with `--no_normalize`
- Optional HU clipping with `--clip_min` and `--clip_max`

## Output Format

### Directory Structure

```
AVTE_2D_processed/
├── dataset_info.json          # Dataset statistics and metadata
├── train/
│   ├── 00001MTY_000.npz      # First slice of case 00001MTY
│   ├── 00001MTY_001.npz      # Second slice
│   ├── ...
│   └── 00001MTY_454.npz      # Last slice (455 total slices)
├── val/
│   └── ...
└── test/
    └── ...
```

### NPZ File Contents

Each `.npz` file contains:

```python
{
    'image': np.ndarray,    # Shape: (C, H, W), dtype: float32
    'label': np.ndarray,    # Shape: (H, W), dtype: int8
    'metadata': str         # JSON string with preprocessing info
}
```

### Metadata Fields

```json
{
    "source_file": "00001MTY_0000.nii.gz",
    "slice_index": 100,
    "total_slices": 455,
    "window_size": 2,
    "padding_mode": "replicate",
    "num_channels": 5,
    "spacing": [0.3906, 0.3906, 0.3000],
    "normalized": true,
    "norm_stats": {
        "mean": 125.4,
        "std": 234.7,
        "min": -1024.0,
        "max": 3071.0
    },
    "shape": {
        "image": [5, 512, 512],
        "label": [512, 512]
    }
}
```

## Usage Examples

### Basic Preprocessing

```bash
# Default settings (window_size=2, replicate padding)
python preprocess_2d_slices.py \
    --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

### Single-Slice (No Context)

```bash
# No adjacent slices, just 1 channel
python preprocess_2d_slices.py \
    --window_size 0 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data_single
```

### Larger Context Window

```bash
# 7 channels (3 slices before + center + 3 after)
python preprocess_2d_slices.py \
    --window_size 3 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data_w3
```

### With HU Clipping

```bash
# Clip CT values to typical range before normalization
python preprocess_2d_slices.py \
    --clip_min -1024 \
    --clip_max 3071 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data_clipped
```

### Custom Train/Val/Test Split

Splits are now handled by the dataloader:

```python
from avte_dataloader import create_avte_dataloaders

# 70% train, 15% val, 15% test
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    train_ratio=0.7,
    val_ratio=0.15
)
```

### Without Normalization

```bash
# Keep original HU values
python preprocess_2d_slices.py \
    --no_normalize \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data_raw
```

## DataLoader Usage

### Basic Usage

```python
from avte_dataloader import AVTE2DDataset, create_avte_dataloaders

# Option 1: Use helper function
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    batch_size=16,
    num_workers=4,
    pin_memory=True
)

# Option 2: Create dataset manually
from torch.utils.data import DataLoader

train_dataset = AVTE2DDataset(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    split='train',
    load_to_memory=False  # Set True to load all data to RAM
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

### With Augmentation

```python
import torchvision.transforms as transforms

# Define augmentation transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
])

train_dataset = AVTE2DDataset(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    split='train',
    transform=transform  # Apply to images only
)
```

### Loading to Memory

For faster training on systems with sufficient RAM:

```python
train_dataset = AVTE2DDataset(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    split='train',
    load_to_memory=True  # Load all data to RAM
)
```

### Accessing Metadata

```python
# Get metadata for a specific sample
metadata = train_dataset.get_metadata(0)
print(metadata['source_file'])
print(metadata['slice_index'])

# Get formatted sample info
info = train_dataset.get_sample_info(0)
print(info)
```

## Testing

### Test Preprocessing

```bash
# Test on a small subset first
python preprocess_2d_slices.py \
    --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \
    --output_dir /tmp/test_output
# Note: All cases are processed; use dataloader to limit to specific cases
```

### Test DataLoader

```bash
# Run the dataloader test
python avte_dataloader.py /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

## Performance Tips

1. **Multi-worker DataLoading**: Use `num_workers=4` or more for faster data loading
2. **Pin Memory**: Enable `pin_memory=True` when using GPU
3. **Load to Memory**: Use `load_to_memory=True` if you have enough RAM (faster training)
4. **Preprocessing**: Preprocess once, train many times (preprocessing is the bottleneck)

## Expected Processing Time

For Dataset006_model_9_4 (~100 cases):
- Preprocessing: ~30-60 minutes (depends on disk I/O)
- Total output size: ~100-200 GB (depends on window size and dimensions)

## Troubleshooting

### Out of Disk Space

Reduce output size by:
- Using smaller window size
- Preprocessing only training split first
- Using different output directory on larger disk

### Out of Memory

- Don't use `load_to_memory=True`
- Reduce `batch_size`
- Reduce `num_workers`

### Slow Data Loading

- Increase `num_workers`
- Enable `pin_memory=True`
- Consider using SSD for data storage
- Try `load_to_memory=True` if you have enough RAM

## Citation

If you use this preprocessing pipeline, please cite the original nnUNet paper:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature methods, 18(2), 203-211.
```

## Support

For issues or questions:
1. Check this README for common solutions
2. Review the example scripts
3. Open an issue in the repository
