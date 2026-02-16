# Path Configuration

## Updated Default Paths

All scripts and documentation have been updated with the following default paths:

### Input Directory (Raw Data)
```
/data/avte_training/nnUNet_raw/Dataset006_model_9_4
```
This directory should contain:
- `imagesTr/` - Training images (*.nii.gz)
- `labelsTr/` - Training labels (*.nii.gz)

### Output Directory (Preprocessed Data)
```
/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```
This directory will be created and will contain:
- `dataset_info.json` - Dataset statistics
- `train/` - Training slices (*.npz)
- `val/` - Validation slices (*.npz)
- `test/` - Test slices (*.npz)

## Files Updated

The following files now use the new default output directory:

1. **preprocess_2d_slices.py** - Preprocessing script
2. **avte_dataloader.py** - DataLoader implementation
3. **example_usage.py** - Example training script
4. **README.md** - Full documentation
5. **GETTING_STARTED.md** - Quick start guide
6. **QUICK_REFERENCE.md** - Command reference

## Quick Commands

### Create Output Directory
```bash
mkdir -p /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

### Run Preprocessing with Default Paths
```bash
cd /home/jin.park/Code_Hendrix/Node-MedicalImaging-Framework/examples/AVTE

python preprocess_2d_slices.py
```

### Run with Custom Paths
```bash
python preprocess_2d_slices.py \
    --input_dir /path/to/your/input \
    --output_dir /path/to/your/output
```

### Test DataLoader
```bash
python avte_dataloader.py /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

### Run Example Training
```bash
python example_usage.py
```

## Directory Structure

After preprocessing, your directory structure will be:

```
/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/
└── 2D_data/
    ├── dataset_info.json
    ├── train/
    │   ├── 00001MTY_000.npz
    │   ├── 00001MTY_001.npz
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

## Storage Requirements

Estimated space needed in `/home/jin.park/Code_Hendrix/Data/`:

- **Window size 0**: ~50 GB
- **Window size 1**: ~75 GB
- **Window size 2**: ~100-200 GB (default)
- **Window size 3**: ~150-300 GB

Check available space before preprocessing:
```bash
df -h /home/jin.park/Code_Hendrix/Data/
```

## Changing Paths

If you need to use different paths, you can:

### Option 1: Use Command-Line Arguments
```bash
python preprocess_2d_slices.py \
    --input_dir /your/custom/input \
    --output_dir /your/custom/output
```

### Option 2: Modify the Default in Code

Edit `preprocess_2d_slices.py` line 364:
```python
default='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
```

### Option 3: Set Environment Variable (Future Enhancement)
```bash
export AVTE_DATA_DIR="/your/custom/path"
python preprocess_2d_slices.py
```

## Verification

Verify paths are correct:
```bash
# Check input data exists
ls /data/avte_training/nnUNet_raw/Dataset006_model_9_4/imagesTr/ | head -5

# Check output directory is writable
mkdir -p /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
touch /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data/test_write
rm /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data/test_write
echo "✓ Output directory is writable"
```

## Notes

- The output directory will be created automatically if it doesn't exist
- Make sure you have write permissions for the output directory
- The preprocessing script checks for sufficient disk space (future enhancement)
- All paths use absolute paths to avoid confusion with relative paths
