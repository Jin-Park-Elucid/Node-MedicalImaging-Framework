# AVTE 2D Dataloader GUI Node Guide

This guide explains how to use the AVTE 2D dataloader node in the medical imaging framework's visual pipeline editor.

## Overview

The **AVTE2DLoader** node provides a graphical interface for loading preprocessed AVTE 2D segmentation data with multi-slice context windows.

### Node Information
- **Category**: Data
- **Name**: AVTE2DLoader
- **Description**: Load AVTE 2D segmentation dataset with multi-slice context

## Node Setup

### 1. Register the Node

The node is automatically registered when you import the AVTE module:

```python
# In your pipeline script or GUI initialization
import sys
from pathlib import Path

# Add AVTE module to path
sys.path.insert(0, '/home/jin.park/Code_Hendrix/Node-MedicalImaging-Framework/examples/AVTE')

# Import to register the node
import avte_dataloader_node
```

Or add to your node discovery paths:
```python
# In your framework configuration
NODE_PATHS = [
    '/home/jin.park/Code_Hendrix/Node-MedicalImaging-Framework/examples/AVTE'
]
```

### 2. Node Outputs

The AVTE2DLoader node provides 7 outputs:

| Output | Type | Description |
|--------|------|-------------|
| `train_loader` | BATCH | PyTorch DataLoader for training |
| `val_loader` | BATCH | PyTorch DataLoader for validation |
| `test_loader` | BATCH | PyTorch DataLoader for testing |
| `num_train` | ANY | Number of training samples |
| `num_val` | ANY | Number of validation samples |
| `num_test` | ANY | Number of test samples |
| `num_channels` | ANY | Number of input channels (2*window_size + 1) |

### 3. Configuration Fields

The node exposes the following configuration fields in the GUI:

#### Data Directory
- **Type**: Text field
- **Default**: `/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data`
- **Description**: Path to preprocessed AVTE 2D data
- **Required**: Yes

#### Batch Size
- **Type**: Text field (integer)
- **Default**: `8`
- **Description**: Number of samples per batch
- **Range**: 1-256 (recommended)

#### Train Ratio
- **Type**: Text field (float)
- **Default**: `0.8`
- **Description**: Proportion of cases for training
- **Range**: 0.0-1.0
- **Example**: 0.8 = 80% training

#### Validation Ratio
- **Type**: Text field (float)
- **Default**: `0.1`
- **Description**: Proportion of cases for validation
- **Range**: 0.0-1.0
- **Note**: Test ratio = 1.0 - train_ratio - val_ratio

#### Random Seed
- **Type**: Text field (integer)
- **Default**: `42`
- **Description**: Seed for reproducible splits
- **Note**: Same seed = same split every time

#### Num Workers
- **Type**: Text field (integer)
- **Default**: `4`
- **Description**: Number of parallel data loading workers
- **Recommended**: 4-8 for most systems

#### Shuffle Training
- **Type**: Dropdown
- **Options**: `True`, `False`
- **Default**: `True`
- **Description**: Whether to shuffle training data

#### Pin Memory
- **Type**: Dropdown
- **Options**: `True`, `False`
- **Default**: `True`
- **Description**: Pin memory for faster GPU transfer
- **Note**: Only effective when using CUDA

#### Load to Memory
- **Type**: Dropdown
- **Options**: `False`, `True`
- **Default**: `False`
- **Description**: Load all data to RAM (faster but memory intensive)
- **Warning**: Requires significant RAM (~1-2 GB per 1000 slices)

## Usage Examples

### Example 1: Basic Training Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  AVTE2DLoader   │────▶│  Model       │────▶│  Training   │
│                 │     │  (U-Net)     │     │  Loop       │
└─────────────────┘     └──────────────┘     └─────────────┘
  train_loader              model              loss
```

**Configuration**:
- Data Directory: `/path/to/preprocessed/data`
- Batch Size: `16`
- Train Ratio: `0.8`
- Val Ratio: `0.1`
- Num Workers: `4`

### Example 2: Experiment with Different Splits

Create multiple AVTE2DLoader nodes with different configurations:

**Node 1: Standard Split**
- Random Seed: `42`
- Train Ratio: `0.8`
- Val Ratio: `0.1`

**Node 2: Large Training Set**
- Random Seed: `42`
- Train Ratio: `0.9`
- Val Ratio: `0.05`

**Node 3: Different Random Split**
- Random Seed: `123`
- Train Ratio: `0.8`
- Val Ratio: `0.1`

### Example 3: K-Fold Cross-Validation

Create 5 AVTE2DLoader nodes with different seeds:

```python
for fold in range(5):
    node = AVTE2DLoaderNode()
    node.set_config('random_seed', str(fold))
    node.set_config('train_ratio', '0.8')
    node.set_config('val_ratio', '0.2')  # No test set
```

## Pipeline Integration

### Connecting to Training Node

```python
# Node connections
avte_loader = AVTE2DLoaderNode()
model_node = YourModelNode()
training_node = YourTrainingNode()

# Connect outputs to inputs
model_node.set_input('train_loader', avte_loader.get_output('train_loader'))
model_node.set_input('val_loader', avte_loader.get_output('val_loader'))
training_node.set_input('num_channels', avte_loader.get_output('num_channels'))
```

### Accessing Outputs in Code

```python
# Execute the node
if avte_loader.execute():
    # Get the dataloaders
    train_loader = avte_loader.get_output_value('train_loader')
    val_loader = avte_loader.get_output_value('val_loader')
    test_loader = avte_loader.get_output_value('test_loader')

    # Get statistics
    num_train = avte_loader.get_output_value('num_train')
    num_val = avte_loader.get_output_value('num_val')
    num_test = avte_loader.get_output_value('num_test')
    num_channels = avte_loader.get_output_value('num_channels')

    # Use in training
    for epoch in range(epochs):
        for images, labels in train_loader:
            # images: (batch, channels, height, width)
            # labels: (batch, height, width)
            train_step(images, labels)
```

## Testing the Node

### Standalone Test

```bash
cd /home/jin.park/Code_Hendrix/Node-MedicalImaging-Framework/examples/AVTE
python avte_dataloader_node.py
```

Expected output:
```
============================================================
TESTING AVTE 2D LOADER NODE
============================================================

Executing node...
Loading AVTE 2D dataset from: /path/to/data
Configuration:
  Batch size: 4
  Workers: 0
  Train ratio: 0.8
  Val ratio: 0.1
  Random seed: 42

✓ AVTE2DDataset initialized:
  Split: train
  Cases: 80 (100 total)
  Slices: 32000
  ...

✓ Created AVTE 2D data loaders:
  Training:   32000 slices, 8000 batches
  Validation: 4000 slices, 1000 batches
  Test:       4000 slices, 1000 batches
  Channels:   5

============================================================
✓ NODE TEST PASSED!
============================================================
```

### GUI Test

1. Launch the GUI:
   ```bash
   python launch_gui.py
   ```

2. Add AVTE2DLoader node from Data category

3. Configure the node in the properties panel

4. Connect to downstream nodes (model, training)

5. Execute the pipeline

## Troubleshooting

### Node Not Appearing in GUI

**Problem**: AVTE2DLoader not showing in the Data category

**Solutions**:
1. Check node registration:
   ```python
   from medical_imaging_framework.core import NodeRegistry
   print(NodeRegistry.get_all_nodes())
   ```

2. Verify import path:
   ```python
   import sys
   sys.path.insert(0, '/path/to/AVTE')
   import avte_dataloader_node
   ```

3. Check for import errors:
   ```bash
   python -c "import avte_dataloader_node"
   ```

### Data Directory Not Found

**Problem**: Error about missing data directory

**Solutions**:
1. Verify the path exists:
   ```bash
   ls -la /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
   ```

2. Check for .npz files:
   ```bash
   ls /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data/*.npz | head -5
   ```

3. Run preprocessing if needed:
   ```bash
   python preprocess_2d_slices.py
   ```

### Out of Memory

**Problem**: System runs out of RAM

**Solutions**:
1. Set "Load to Memory" to `False`
2. Reduce batch size
3. Reduce num_workers
4. Close other applications

### Slow Data Loading

**Problem**: Training is bottlenecked by data loading

**Solutions**:
1. Increase num_workers (try 4-8)
2. Enable pin_memory for GPU training
3. Use SSD storage
4. Consider load_to_memory=True if you have enough RAM

## Best Practices

### 1. Configuration

```python
# For development/debugging
node.set_config('batch_size', '4')
node.set_config('num_workers', '0')
node.set_config('load_to_memory', 'False')

# For training on GPU
node.set_config('batch_size', '16')
node.set_config('num_workers', '4')
node.set_config('pin_memory', 'True')

# For maximum speed (if RAM available)
node.set_config('batch_size', '32')
node.set_config('num_workers', '8')
node.set_config('load_to_memory', 'True')
```

### 2. Reproducibility

Always use the same random seed for reproducible experiments:
```python
node.set_config('random_seed', '42')
```

### 3. Split Ratios

Choose appropriate ratios for your dataset size:

| Dataset Size | Recommended Split |
|--------------|-------------------|
| < 50 cases | 0.8 / 0.1 / 0.1 |
| 50-100 cases | 0.8 / 0.1 / 0.1 |
| 100-500 cases | 0.7 / 0.15 / 0.15 |
| > 500 cases | 0.7 / 0.2 / 0.1 |

### 4. Monitoring

Check the console output for statistics:
- Number of cases in each split
- Number of slices in each split
- Number of batches
- Number of input channels

## Advanced Usage

### Custom Field Validation

Add validation to field definitions:

```python
def get_field_definitions(self):
    fields = super().get_field_definitions()

    # Add validation
    fields['batch_size']['validate'] = lambda x: 1 <= int(x) <= 256
    fields['train_ratio']['validate'] = lambda x: 0.0 <= float(x) <= 1.0

    return fields
```

### Dynamic Configuration

Update configuration based on other nodes:

```python
# Get num_channels from preprocessed data
num_channels = avte_loader.get_output_value('num_channels')

# Configure model with correct input channels
model_node.set_config('in_channels', str(num_channels))
```

### Pipeline Validation

Validate the pipeline before execution:

```python
def validate_pipeline(avte_node):
    # Check data directory exists
    data_dir = Path(avte_node.get_config('data_dir'))
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Check for .npz files
    npz_files = list(data_dir.glob('*.npz'))
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")

    # Validate split ratios
    train_ratio = float(avte_node.get_config('train_ratio'))
    val_ratio = float(avte_node.get_config('val_ratio'))
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
```

## Summary

The AVTE2DLoader node provides:

✅ Easy integration with visual pipeline editor
✅ Configurable train/val/test splitting
✅ Multi-slice context windows
✅ Reproducible experiments
✅ GPU optimization support
✅ Comprehensive output ports
✅ Detailed error messages

For more information, see:
- `avte_dataloader.py` - DataLoader implementation
- `preprocess_2d_slices.py` - Data preprocessing
- `README.md` - Full documentation
