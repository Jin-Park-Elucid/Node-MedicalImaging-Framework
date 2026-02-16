# GUI Launch Guide

## Quick Start

### Launch the GUI

From the project root:
```bash
python examples/launch_gui.py
```

Or from the examples directory:
```bash
cd examples
python launch_gui.py
```

## What Gets Registered

The launch script automatically registers all custom nodes:

### 1. MedicalSegmentationLoader
- **Category**: Data
- **Source**: `examples/medical_segmentation_pipeline/custom_dataloader.py`
- **Purpose**: Load synthetic medical segmentation data
- **Outputs**: train_loader, val_loader, test_loader, dataset_info

### 2. AVTE2DLoader
- **Category**: Data
- **Source**: `examples/AVTE/avte_dataloader_node.py`
- **Purpose**: Load AVTE 2D segmentation data with multi-slice context
- **Outputs**: train_loader, val_loader, test_loader, num_train, num_val, num_test, num_channels
- **Configuration**: 9 configurable fields (data_dir, batch_size, train_ratio, etc.)

## Expected Output

When you launch the GUI, you should see:

```
================================================================================
MEDICAL IMAGING FRAMEWORK - GUI EDITOR
================================================================================

Custom nodes registered:
  ✓ MedicalSegmentationLoader
  ✓ AVTE2DLoader

Total nodes available: [number]

Workflow files available:
  • synthetic_data_workflow.json

================================================================================
```

## Troubleshooting

### AVTE2DLoader Not Available

If you see:
```
✗ AVTE2DLoader (not available: No module named 'torch')
```

**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

### Custom Dataloader Import Error

If MedicalSegmentationLoader fails to import, ensure:
1. You're running from the correct directory (project root or examples/)
2. The file `examples/medical_segmentation_pipeline/custom_dataloader.py` exists
3. The framework is properly installed

## File Structure

```
examples/
├── launch_gui.py                          # GUI launcher (moved here)
├── LAUNCH_GUI_GUIDE.md                    # This file
├── medical_segmentation_pipeline/
│   ├── custom_dataloader.py              # MedicalSegmentationLoader node
│   ├── synthetic_data_workflow.json      # Example workflow
│   └── test_node_registration.py         # Test script
└── AVTE/
    ├── avte_dataloader_node.py           # AVTE2DLoader node
    ├── avte_dataloader.py                # Dataset implementation
    ├── preprocess_2d_slices.py           # Preprocessing script
    ├── __init__.py                       # Module initialization
    └── docs/                             # Documentation
        ├── README.md
        ├── GUI_NODE_GUIDE.md
        └── ...
```

## Path Configuration

The launcher sets up paths automatically:

```python
# From examples/launch_gui.py

# Framework root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Medical segmentation pipeline (for custom_dataloader)
sys.path.insert(0, str(Path(__file__).parent / 'medical_segmentation_pipeline'))

# AVTE module (for AVTE nodes)
sys.path.insert(0, str(Path(__file__).parent / 'AVTE'))
```

## Using the Nodes in GUI

### Creating a Pipeline with AVTE2DLoader

1. **Add AVTE2DLoader node** from Data category
2. **Configure the node**:
   - Data Directory: `/path/to/preprocessed/2D_data`
   - Batch Size: `16`
   - Train Ratio: `0.8`
   - Val Ratio: `0.1`
   - Random Seed: `42`
3. **Connect outputs**:
   - `train_loader` → Model input
   - `num_channels` → Model configuration
4. **Execute pipeline**

### Creating a Pipeline with MedicalSegmentationLoader

1. **Add MedicalSegmentationLoader node** from Data category
2. **Configure the node**:
   - Image Size: `128`
   - Num Samples: `1000`
   - Batch Size: `16`
3. **Connect outputs**:
   - `train_loader` → Model input
   - `dataset_info` → Logging/monitoring
4. **Execute pipeline**

## Workflow Files

The GUI can load/save workflows as JSON files. Example workflows are located in:
- `examples/medical_segmentation_pipeline/synthetic_data_workflow.json`

## Development

### Testing Node Registration

Without launching the full GUI:
```bash
python examples/medical_segmentation_pipeline/test_node_registration.py
```

This will:
- Import both nodes
- Check NodeRegistry
- List all registered nodes
- Report success/failure

### Adding New Nodes

To add a new custom node:

1. Create your node file (e.g., `examples/my_module/my_node.py`)
2. Import in `examples/launch_gui.py`:
   ```python
   sys.path.insert(0, str(Path(__file__).parent / 'my_module'))
   from my_node import MyCustomNode
   ```
3. Update the startup message to show your node

## Requirements

- Python 3.7+
- PyTorch (for AVTE2DLoader)
- Medical Imaging Framework
- Additional dependencies as needed by specific nodes

## Related Documentation

- AVTE Module: `examples/AVTE/docs/README.md`
- AVTE GUI Node: `examples/AVTE/docs/GUI_NODE_GUIDE.md`
- Framework Documentation: See main README.md

---

**Last Updated**: 2026-02-08
**Location**: `examples/launch_gui.py`
