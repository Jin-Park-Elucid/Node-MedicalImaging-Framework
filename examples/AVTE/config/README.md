# AVTE Workflow Configurations

This directory contains pre-configured workflow files for the GUI pipeline editor.

## Available Workflows

### 1. simple_training_workflow.json
**Purpose**: Test and validate AVTE2DLoader node

**Nodes**:
- AVTE2DLoader (data loading)
- PrintInfo (display statistics)

**Use Cases**:
- Verify data is preprocessed correctly
- Test dataloader configuration
- Check dataset splits and statistics
- Quick validation before training

**How to Use**:
1. Open GUI: `python examples/launch_gui.py`
2. File → Open Workflow → `examples/AVTE/config/simple_training_workflow.json`
3. Update `data_dir` in AVTE2DLoader node properties
4. Execute workflow
5. Check console for dataset statistics

### 2. training_workflow.json
**Purpose**: Complete 2D segmentation training pipeline

**Nodes**:
- AVTE2DLoader (data loading)
- ModelConfig (U-Net configuration)
- TrainingConfig (hyperparameters)
- Trainer (training loop)
- MetricsLogger (TensorBoard logging)

**Use Cases**:
- Full training pipeline
- Experiment tracking
- Production training runs

**How to Use**:
1. Open GUI: `python examples/launch_gui.py`
2. File → Open Workflow → `examples/AVTE/config/training_workflow.json`
3. Configure paths:
   - `data_dir` in AVTE2DLoader
   - `checkpoint_dir` in TrainingConfig
   - `log_dir` in MetricsLogger
4. Adjust hyperparameters as needed
5. Execute workflow
6. Monitor with: `tensorboard --logdir=/path/to/logs`

## Configuration Guide

### AVTE2DLoader Configuration

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| data_dir | (path) | Preprocessed data location | Update to your path |
| batch_size | 8-16 | Samples per batch | Increase for faster training (if GPU allows) |
| train_ratio | 0.8 | Training proportion | Standard: 0.7-0.8 |
| val_ratio | 0.1 | Validation proportion | Standard: 0.1-0.2 |
| random_seed | 42 | Reproducibility seed | Keep same for comparisons |
| num_workers | 4 | Parallel data loading | 4-8 for most systems |
| shuffle_train | True | Shuffle training data | Always True for training |
| pin_memory | True | GPU memory pinning | True for GPU training |
| load_to_memory | False | Load all to RAM | True if RAM available (faster) |

### Model Configuration (U-Net)

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| in_channels | 5 | Input channels | Auto from num_channels output |
| out_channels | 2 | Output classes | Background + vessel = 2 |
| features | [64,128,256,512] | Feature maps per level | Increase for larger images |
| dropout | 0.1 | Dropout rate | 0.1-0.3 to prevent overfitting |

### Training Configuration

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| num_epochs | 100 | Training epochs | 50-150 typical |
| learning_rate | 0.001 | Initial learning rate | 0.0001-0.01 range |
| optimizer | Adam | Optimizer type | Adam or SGD |
| loss_function | CrossEntropyLoss | Loss function | DiceLoss for imbalanced data |
| device | cuda | Training device | cuda or cpu |

## Workflow Customization

### Creating a Custom Workflow

1. **Start with Simple Workflow**:
   ```bash
   cp simple_training_workflow.json my_workflow.json
   ```

2. **Edit in Text Editor** or use GUI:
   - Add/remove nodes
   - Modify configurations
   - Update connections

3. **Update Metadata**:
   ```json
   {
     "name": "My Custom AVTE Workflow",
     "description": "Description here",
     "version": "1.0.0"
   }
   ```

4. **Save and Load in GUI**

### Example: Changing Batch Size

Edit the workflow JSON:
```json
{
  "id": "avte_dataloader_1",
  "type": "AVTE2DLoader",
  "config": {
    "batch_size": "32"  // Change from 8 to 32
  }
}
```

### Example: Adding Data Augmentation

Add a new node:
```json
{
  "id": "augmentation_1",
  "type": "DataAugmentation",
  "position": {"x": 300, "y": 200},
  "config": {
    "rotate": "True",
    "flip": "True",
    "elastic_transform": "True"
  }
}
```

Connect between dataloader and model:
```json
{
  "source_node": "avte_dataloader_1",
  "source_port": "train_loader",
  "target_node": "augmentation_1",
  "target_port": "input"
}
```

## Common Workflow Patterns

### Pattern 1: Data Validation
```
AVTE2DLoader → PrintInfo
```
Quick check of dataset before training.

### Pattern 2: Basic Training
```
AVTE2DLoader → Model → Trainer
```
Minimal training setup.

### Pattern 3: Full Training Pipeline
```
AVTE2DLoader → Augmentation → Model → Trainer → Logger → Checkpointer
```
Production training with all features.

### Pattern 4: Hyperparameter Search
```
AVTE2DLoader → [Multiple Model Configs] → [Multiple Trainers] → Comparison
```
Test different configurations in parallel.

## Troubleshooting Workflows

### Workflow Won't Load
**Check**:
- JSON syntax is valid (use `python -m json.tool workflow.json`)
- File path is correct
- All node types exist in registry

### Node Execution Fails
**Check**:
- All required inputs are connected
- Configuration values are valid (correct types, ranges)
- File paths exist (data_dir, checkpoint_dir, etc.)
- Required dependencies installed (PyTorch, etc.)

### Connections Invalid
**Check**:
- Source port type matches target port type
- Node IDs are unique
- Port names are correct

## Best Practices

### 1. Version Control
- Keep workflow files in git
- Use descriptive names
- Document changes in metadata

### 2. Path Configuration
- Use absolute paths for reproducibility
- Create a `config_local.json` for machine-specific paths
- Don't commit machine-specific paths

### 3. Experimentation
- Start with simple workflow
- Add complexity incrementally
- Save successful configurations

### 4. Documentation
- Use `notes` field in nodes
- Update `usage_notes` in metadata
- Keep README updated

## File Structure

```
examples/AVTE/config/
├── README.md                        # This file
├── simple_training_workflow.json   # Minimal testing workflow
├── training_workflow.json          # Full training pipeline
└── [your_custom_workflows].json    # Your custom configurations
```

## Integration with GUI

These workflows integrate with the GUI launcher:

```bash
# Launch GUI
python examples/launch_gui.py

# Workflows appear in File → Open Workflow menu
# Or drag-and-drop JSON files into GUI
```

## Related Documentation

- **AVTE Module**: `../docs/README.md`
- **GUI Node Guide**: `../docs/GUI_NODE_GUIDE.md`
- **Launch Guide**: `../../LAUNCH_GUI_GUIDE.md`
- **Preprocessing**: `../preprocess_2d_slices.py`

## Example Session

```bash
# 1. Preprocess data
cd examples/AVTE
python preprocess_2d_slices.py --num_workers 8

# 2. Launch GUI
cd ../..
python examples/launch_gui.py

# 3. In GUI:
#    - Open simple_training_workflow.json
#    - Update data_dir
#    - Execute to validate
#    - Open training_workflow.json
#    - Configure and run training

# 4. Monitor training
tensorboard --logdir=/home/jin.park/Code_Hendrix/logs/avte_2d
```

---

**Last Updated**: 2026-02-08
**Workflow Version**: 1.0.0
