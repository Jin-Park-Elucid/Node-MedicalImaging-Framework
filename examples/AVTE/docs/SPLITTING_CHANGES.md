# Data Splitting Changes

## Summary

The preprocessing script no longer handles train/val/test splitting. All data is processed into a single output directory, and the dataloader handles splitting dynamically.

## What Changed

### Before (Old Behavior)
- Preprocessing script split data into `train/`, `val/`, `test/` subdirectories
- Fixed split ratios at preprocessing time
- Required reprocessing to change splits

### After (New Behavior)
- Preprocessing script processes all data into a single directory
- Dataloader splits data on-the-fly based on case names
- Can change split ratios without reprocessing
- More flexible for experiments

## Preprocessing Script Changes

### Removed Parameters
```bash
# These arguments are REMOVED
--train_ratio 0.8
--val_ratio 0.1
```

### New Output Structure
```
output_dir/
├── dataset_info.json
├── case1_000.npz
├── case1_001.npz
├── case2_000.npz
└── ...
```

No more subdirectories for train/val/test!

### Updated Command
```bash
# Old (no longer works)
python preprocess_2d_slices.py \
    --train_ratio 0.8 \
    --val_ratio 0.1

# New (simplified)
python preprocess_2d_slices.py \
    --num_workers 8
```

## DataLoader Changes

### New Features

The dataloader now handles splitting internally:

```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/path/to/processed/data',
    batch_size=16,
    train_ratio=0.8,  # NEW: Set split ratio in dataloader
    val_ratio=0.1,    # NEW: Set split ratio in dataloader
    random_seed=42    # NEW: Reproducible splits
)
```

### How It Works

1. **Case-level splitting**: Data is split by cases (patients), not individual slices
2. **Reproducible**: Same `random_seed` always produces the same split
3. **Flexible**: Change ratios without reprocessing

Example:
```python
# 70% train, 15% val, 15% test
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    train_ratio=0.7,
    val_ratio=0.15
)

# Different split without reprocessing!
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,  # Same data
    train_ratio=0.9,    # Different ratio
    val_ratio=0.05
)
```

## Migration Guide

### If You Haven't Preprocessed Yet

Just use the new scripts - no changes needed!

```bash
# Preprocess
python preprocess_2d_slices.py

# Use in training
from avte_dataloader import create_avte_dataloaders
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/path/to/data',
    train_ratio=0.8,
    val_ratio=0.1
)
```

### If You Already Preprocessed with Old Version

You have two options:

#### Option 1: Reprocess (Recommended)
```bash
# Delete old preprocessed data
rm -rf /path/to/old/preprocessed/data

# Reprocess with new script
python preprocess_2d_slices.py \
    --output_dir /path/to/new/output
```

#### Option 2: Reorganize Manually
```bash
# Move all files to single directory
cd /path/to/preprocessed/data
mkdir all_data
mv train/*.npz all_data/
mv val/*.npz all_data/
mv test/*.npz all_data/
rm -rf train val test

# Update dataset_info.json manually
```

## Benefits

### 1. Faster Iteration
```python
# Experiment with different splits
for train_ratio in [0.7, 0.8, 0.9]:
    train_loader, val_loader, test_loader = create_avte_dataloaders(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=0.1
    )
    # Train model...
```

### 2. No Reprocessing Needed
- Change split ratios instantly
- Try k-fold cross-validation
- Adjust for small validation sets

### 3. Consistent Case-Level Splitting
- Cases stay together in one split
- No data leakage between splits
- Proper evaluation on unseen patients

### 4. Reproducible Experiments
```python
# Always get same split with same seed
create_avte_dataloaders(
    data_dir=data_dir,
    random_seed=42  # Reproducible
)
```

## API Reference

### Preprocessing

```bash
python preprocess_2d_slices.py \
    --input_dir <input> \
    --output_dir <output> \
    --window_size 2 \
    --padding_mode replicate \
    --num_workers 8
    # NO train_ratio or val_ratio!
```

### DataLoader

```python
AVTE2DDataset(
    data_dir: str,              # Directory with .npz files
    split: str = 'train',       # 'train', 'val', or 'test'
    train_ratio: float = 0.8,   # Proportion for training
    val_ratio: float = 0.1,     # Proportion for validation
    random_seed: int = 42,      # For reproducibility
    transform: Optional = None,
    target_transform: Optional = None,
    load_to_memory: bool = False
)

create_avte_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_to_memory: bool = False,
    train_ratio: float = 0.8,   # NEW
    val_ratio: float = 0.1,     # NEW
    random_seed: int = 42       # NEW
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

## Examples

### Basic Usage
```python
from avte_dataloader import create_avte_dataloaders

# Default: 80% train, 10% val, 10% test
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    batch_size=16
)
```

### Custom Splits
```python
# 90% train, 5% val, 5% test
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    batch_size=16,
    train_ratio=0.9,
    val_ratio=0.05
)
```

### Reproducible Splits
```python
# Same seed = same split every time
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    random_seed=42  # Always same split
)

# Different seed = different split
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    random_seed=123  # Different split
)
```

### K-Fold Cross-Validation
```python
# Easy to implement k-fold now!
for fold in range(5):
    train_loader, val_loader, test_loader = create_avte_dataloaders(
        data_dir=data_dir,
        train_ratio=0.8,
        val_ratio=0.2,  # No test set for k-fold
        random_seed=fold  # Different split per fold
    )
    # Train on this fold...
```

## Important Notes

1. **Case-Level Splitting**: Data is split by patient/case, not by slice. All slices from one case stay in the same split.

2. **Reproducibility**: Use the same `random_seed` to get the same split across runs.

3. **File Format**: The `.npz` filename format must be `<casename>_<###>.npz` where `###` is the slice index.

4. **No Empty Splits**: If you set `val_ratio=0`, the dataloader will still create a val set with at least 1 case. Adjust ratios if needed.

## Troubleshooting

### Error: "No slices found for train split"
- Check that .npz files exist in data_dir
- Verify filename format: `casename_000.npz`
- Make sure train_ratio + val_ratio < 1.0

### Different number of slices than expected
- Normal! Cases have different numbers of slices
- Split is by case, not by slice count

### Want to see which cases are in which split
```python
# Get dataset info
dataset = AVTE2DDataset(data_dir=data_dir, split='train')
print(f"Train cases: {len(dataset.slice_files)} slices")

# Extract case names
case_names = set('_'.join(f.stem.split('_')[:-1]) for f in dataset.slice_files)
print(f"Case names in train: {sorted(case_names)}")
```

## Summary

✅ **Preprocessing**: Simpler, faster, no split arguments
✅ **DataLoader**: More flexible, handles splitting
✅ **Experiments**: Change splits without reprocessing
✅ **Reproducible**: Consistent splits with random_seed

The new approach separates concerns: preprocessing focuses on data preparation, while the dataloader handles dataset management and splitting.
