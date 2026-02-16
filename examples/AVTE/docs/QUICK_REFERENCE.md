# AVTE 2D Segmentation - Quick Reference

## Commands

### Preprocessing

```bash
# Basic usage
python preprocess_2d_slices.py

# Custom settings
python preprocess_2d_slices.py \
    --input_dir <input_path> \
    --output_dir <output_path> \
    --window_size 2 \
    --padding_mode replicate \
    --num_workers 8

# All options
python preprocess_2d_slices.py \
    --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data \
    --window_size 2 \
    --padding_mode replicate \
    --clip_min -1024 \
    --clip_max 3071 \
    --no_normalize \  # Optional: disable normalization
    --num_workers 8

# Note: Train/val/test splits are now handled by the dataloader
```

### Testing DataLoader

```bash
# Test with default path
python avte_dataloader.py

# Test with custom path
python avte_dataloader.py /path/to/preprocessed/data
```

### Example Training

```bash
python example_usage.py
```

## Python API

### Load Dataset

```python
from avte_dataloader import AVTE2DDataset

dataset = AVTE2DDataset(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    split='train',
    load_to_memory=False
)

image, label = dataset[0]  # Get first sample
```

### Create DataLoaders

```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    batch_size=16,
    num_workers=4,
    pin_memory=True
)
```

### Training Loop

```python
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # images: (B, C, H, W) where C = 2*window_size + 1
        # labels: (B, H, W)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Key Parameters

### Window Size
- `0`: 1 channel (no context)
- `1`: 3 channels (prev, current, next)
- `2`: 5 channels (default)
- `3`: 7 channels
- `N`: 2N+1 channels

### Padding Modes
- `zero`: Fill missing slices with zeros
- `replicate`: Duplicate first/last slice (default)
- `mirror`: Mirror slices at boundaries

### Data Splits (handled by dataloader)
```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    train_ratio=0.8,   # 80% training
    val_ratio=0.1,     # 10% validation (remaining = test)
    random_seed=42
)
```

### Parallel Processing
- `--num_workers 0`: Single process (debugging)
- `--num_workers 4`: 4 workers (default)
- `--num_workers 8`: 8 workers (recommended for servers)
- `--num_workers -1`: Auto-detect CPU count
- Speed-up: 3-7x with multiprocessing

## File Naming Convention

```
Input:  00001MTY_0000.nii.gz  (3D volume)
Output: 00001MTY_000.npz      (first 2D slice)
        00001MTY_001.npz      (second 2D slice)
        ...
        00001MTY_454.npz      (last 2D slice)
```

## Output Structure

```
output_dir/
├── dataset_info.json
├── train/
│   ├── case1_000.npz
│   ├── case1_001.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## NPZ File Contents

```python
data = np.load('sample.npz')
image = data['image']      # Shape: (C, H, W), float32
label = data['label']      # Shape: (H, W), int8
metadata = json.loads(data['metadata'])  # Dictionary
```

## Common Issues

### "No images found"
- Check input directory path
- Ensure files end with `_0000.nii.gz`

### Out of disk space
- Use smaller window size
- Process only training split
- Check available space: `df -h`

### CUDA out of memory
- Reduce batch_size
- Use fewer num_workers
- Don't use load_to_memory=True

### Slow data loading
- Increase num_workers (4-8 recommended)
- Enable pin_memory=True
- Use SSD storage
- Consider load_to_memory=True

## Performance

### Expected Times (100 cases)
- Preprocessing: 30-60 minutes
- Loading one batch: <0.1s (with num_workers=4)
- Training one epoch: depends on model

### Storage Requirements
- Window size 0: ~50 GB
- Window size 2: ~100-200 GB
- Window size 3: ~150-300 GB

## Next Steps

1. Preprocess your data
2. Test the dataloader
3. Implement/adapt your segmentation model
4. Add data augmentation
5. Train and evaluate
6. Adjust hyperparameters

## Help

For detailed documentation, see `README.md`
For example usage, see `example_usage.py`
For implementation details, see the source code
