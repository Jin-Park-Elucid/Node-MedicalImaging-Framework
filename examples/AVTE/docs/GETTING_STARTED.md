# Getting Started with AVTE 2D Segmentation

This guide will help you quickly get started with preprocessing and training on AVTE data.

## 🚀 Quick Start (3 Steps)

### Step 1: Preprocess the Data

Convert 3D NIfTI volumes to 2D slices:

```bash
cd /home/jin.park/Code_Hendrix/Node-MedicalImaging-Framework/examples/AVTE

python preprocess_2d_slices.py \
    --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \
    --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data \
    --window_size 2 \
    --padding_mode replicate \
    --num_workers 8  # Use 8 parallel workers for faster processing
```

This will:
- Read all NIfTI files from the input directory
- Extract 2D slices with 5 channels (window_size=2)
- Apply z-score normalization
- Split into train/val/test (80%/10%/10%)
- Save to `/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data`

**Expected time**: ~10-20 minutes with 8 workers (single process: ~60 minutes)
**Expected output size**: ~100-200 GB

### Step 2: Test the DataLoader

Verify preprocessing worked correctly:

```bash
python avte_dataloader.py /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

You should see output like:
```
✓ AVTE2DDataset initialized:
  Split: train
  Samples: 38000
  ...
✓ DATALOADER TEST PASSED!
```

### Step 3: Start Training

Use the example training script or integrate with your own code:

```bash
python example_usage.py
```

Or use the dataloader in your own training script:

```python
from avte_dataloader import create_avte_dataloaders

train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
    batch_size=16,
    num_workers=4
)

for images, labels in train_loader:
    # images: (B, 5, H, W) - batch of 5-channel images
    # labels: (B, H, W) - batch of segmentation masks
    ...
```

## 📁 What You Get

After preprocessing, your output directory will look like:

```
/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data/
├── dataset_info.json          # Dataset statistics
├── train/                     # Training slices (~80%)
│   ├── 00001MTY_000.npz
│   ├── 00001MTY_001.npz
│   ├── ...
├── val/                       # Validation slices (~10%)
│   └── ...
└── test/                      # Test slices (~10%)
    └── ...
```

Each `.npz` file contains:
- **image**: Multi-channel 2D slice (C, H, W)
- **label**: Segmentation mask (H, W)
- **metadata**: Preprocessing parameters and statistics

## 🎛️ Configuration Options

### Window Size

Controls how many adjacent slices to include:

| Window Size | Channels | Example |
|-------------|----------|---------|
| 0 | 1 | Just the current slice |
| 1 | 3 | Previous, current, next |
| 2 | 5 | 2 before, current, 2 after |
| 3 | 7 | 3 before, current, 3 after |

**Recommendation**: Start with `window_size=2` (5 channels)

### Padding Modes

How to handle slices at volume boundaries:

- **replicate** (recommended): Duplicate first/last slice
- **zero**: Fill with zeros
- **mirror**: Mirror slices at boundaries

### Data Splits

**Important**: Splitting is now handled by the dataloader, not preprocessing!

The preprocessing script processes all data into one directory. Split ratios are set when creating dataloaders:

```python
from avte_dataloader import create_avte_dataloaders

# Default: 80% train, 10% val, 10% test
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42  # Reproducible splits
)

# Customize easily without reprocessing
train_loader, val_loader, test_loader = create_avte_dataloaders(
    data_dir=data_dir,
    train_ratio=0.7,
    val_ratio=0.15  # 70% train, 15% val, 15% test
)
```

## 💡 Tips for Best Results

### 1. Storage Planning

Check available disk space before preprocessing:
```bash
df -h /data/avte_training
```

Expected output size:
- Window size 0: ~50 GB
- Window size 2: ~100-200 GB
- Window size 3: ~150-300 GB

### 2. Speed Up with Multiprocessing

Use parallel processing for 3-7x faster preprocessing:

```bash
# Use 8 workers (recommended for most servers)
python preprocess_2d_slices.py --num_workers 8

# Auto-detect CPU count
python preprocess_2d_slices.py --num_workers -1

# Single process (for debugging)
python preprocess_2d_slices.py --num_workers 0
```

**Guidelines**:
- Desktop/Workstation (8+ cores): `--num_workers 8`
- Server (16+ cores): `--num_workers 16`
- Laptop (4 cores): `--num_workers 2`

See [MULTIPROCESSING_GUIDE.md](MULTIPROCESSING_GUIDE.md) for detailed performance tuning.

### 3. Memory Optimization

If training on GPU:
```python
train_loader = create_avte_dataloaders(
    data_dir=data_dir,
    batch_size=16,        # Adjust based on GPU memory
    num_workers=4,        # More workers = faster loading
    pin_memory=True,      # Faster GPU transfer
    load_to_memory=False  # Set True if you have enough RAM
)
```

### 4. Data Augmentation

Add augmentation to improve generalization:

```python
import torchvision.transforms as T

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

dataset = AVTE2DDataset(
    data_dir=data_dir,
    split='train',
    transform=transform
)
```

### 5. Model Architecture

For 5-channel input, your model needs to accept `in_channels=5`:

```python
# Example with U-Net
model = UNet(
    in_channels=5,      # Match your window_size
    num_classes=10,     # Number of segmentation classes
    depth=4,
    initial_features=64
)
```

## 📊 Monitoring Progress

During preprocessing, you'll see:
```
Processing train set (80 cases)
Processing train: 100%|████████| 80/80 [30:00<00:00, 22.5s/it]
```

During training, you can monitor:
- Training/validation loss
- Pixel accuracy
- Dice coefficient per class
- IoU (Intersection over Union)

## 🔧 Troubleshooting

### "No images found in directory"
**Solution**: Check that your input directory contains `imagesTr/` and `labelsTr/` folders with `.nii.gz` files

### Out of disk space
**Solutions**:
- Use smaller window size: `--window_size 1`
- Use different output directory on larger disk
- Process a subset of cases for testing first

### CUDA out of memory during training
**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Reduce model size

### Slow data loading
**Solutions**:
- Increase `num_workers` (try 4-8)
- Enable `pin_memory=True`
- Use SSD for data storage
- Try `load_to_memory=True` if you have enough RAM

## 📚 Next Steps

1. **Read the full documentation**: See `README.md` for detailed information
2. **Explore the example**: Run `example_usage.py` to see a complete training loop
3. **Customize preprocessing**: Adjust window size, padding mode, and normalization
4. **Implement your model**: Use your preferred segmentation architecture
5. **Add evaluation metrics**: Implement Dice, IoU, and other metrics
6. **Tune hyperparameters**: Experiment with learning rate, batch size, etc.

## 📖 Additional Resources

- **README.md**: Full documentation
- **QUICK_REFERENCE.md**: Command cheat sheet
- **example_usage.py**: Complete training example
- **preprocess_2d_slices.py**: Source code with detailed comments
- **avte_dataloader.py**: DataLoader implementation

## 🆘 Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section
3. Verify your input data format
4. Test with a small subset first
5. Check available disk space and memory

## ✅ Verification Checklist

Before training, verify:

- [ ] Preprocessing completed without errors
- [ ] Output directory contains train/val/test folders
- [ ] `dataset_info.json` exists with correct statistics
- [ ] DataLoader test passes
- [ ] Sample visualization looks correct
- [ ] Sufficient disk space for training (model checkpoints, logs)
- [ ] GPU available (if using CUDA)

## 🎯 Expected Results

After preprocessing ~100 cases with default settings:
- **Training samples**: ~30,000-40,000 slices
- **Validation samples**: ~4,000-5,000 slices
- **Test samples**: ~4,000-5,000 slices
- **Total storage**: ~100-200 GB
- **Preprocessing time**: ~10-20 minutes (8 workers) or ~60 minutes (single process)

You're now ready to train your 2D segmentation model! 🚀
