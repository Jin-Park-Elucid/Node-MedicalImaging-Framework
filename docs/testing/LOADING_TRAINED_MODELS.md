# Loading Trained Models for Testing

## Overview

When creating a testing/inference workflow, you need to load trained model weights from checkpoints saved during training. Without loading trained weights, the model will use random initialization and produce meaningless results.

---

## The Problem: Random vs Trained Models

### ❌ Wrong: Testing with Random Weights

```json
{
  "nodes": [
    {"type": "UNet2D", "name": "unet_model"},
    {"type": "BatchPredictor", "name": "predictor"}
  ],
  "links": [
    {
      "source_node": "unet_model",
      "source_port": "model",
      "target_node": "predictor",
      "target_port": "model"
    }
  ]
}
```

**Issue**: This creates a fresh UNet2D with random weights. Test results will be meaningless (≈random guessing).

### ✓ Correct: Loading Trained Weights

```json
{
  "nodes": [
    {"type": "UNet2D", "name": "unet_model"},
    {"type": "CheckpointLoader", "name": "checkpoint_loader"},
    {"type": "BatchPredictor", "name": "predictor"}
  ],
  "links": [
    {
      "source_node": "unet_model",
      "source_port": "model",
      "target_node": "checkpoint_loader",
      "target_port": "model"
    },
    {
      "source_node": "checkpoint_loader",
      "source_port": "model",
      "target_node": "predictor",
      "target_port": "model"
    }
  ]
}
```

**Result**: Loads trained weights from checkpoint. Test results are meaningful.

---

## Using CheckpointLoader Node

### What it Does

The **CheckpointLoader** node:
1. Takes an initialized model (with random weights)
2. Loads saved weights from a checkpoint file
3. Outputs the model with trained weights loaded

### Configuration

**Parameter**: `checkpoint_path`
- **Type**: Text field
- **Description**: Path to the checkpoint file to load
- **Examples**:
  - `checkpoints/best_model.pt` - Best model from training
  - `checkpoints/final_model.pt` - Final model after all epochs
  - `checkpoints/checkpoint_epoch_10.pt` - Specific epoch

### Port Connections

**Inputs**:
- `model` (MODEL): The model to load weights into

**Outputs**:
- `model` (MODEL): The same model with loaded weights
- `checkpoint_info` (METRICS): Information about the loaded checkpoint (epoch, loss, path)

---

## Complete Testing Workflow

### Architecture

```
┌──────────────┐
│ data_loader  │  ← Test data
└──────┬───────┘
       │ test_loader (BATCH)
       │
       │    ┌──────────────┐
       │    │  unet_model  │  ← Create model architecture
       │    └──────┬───────┘
       │           │ model (MODEL)
       │           ▼
       │    ┌─────────────────────┐
       │    │ checkpoint_loader   │  ← Load trained weights
       │    └──────┬──────────────┘
       │           │ model (MODEL) - with trained weights
       │           ▼
       ▼    ┌─────────────┐
       ├────▶  predictor  │  ← Run inference
       │    └──────┬──────┘
       │           │ predictions (TENSOR)
       │           ▼
       │    ┌─────────────┐
       │    │   metrics   │  ← Evaluate results
       │    └─────────────┘
```

### Step-by-Step Setup in GUI

1. **Add nodes**:
   - MedicalSegmentationLoader
   - UNet2D (or your network)
   - CheckpointLoader
   - BatchPredictor
   - MetricsCalculator

2. **Configure CheckpointLoader**:
   - Double-click the CheckpointLoader node
   - Set **Checkpoint Path**: `examples/medical_segmentation_pipeline/checkpoints/best_model.pt`
   - Click OK

3. **Create connections**:
   - data_loader.test_loader → predictor.dataloader
   - unet_model.model → checkpoint_loader.model
   - checkpoint_loader.model → predictor.model
   - predictor.all_predictions → metrics.predictions
   - predictor.all_labels → metrics.labels

4. **Execute workflow** (Ctrl+E)

---

## Example Workflows

### Example 1: Basic Testing

**File**: `examples/medical_segmentation_pipeline/testing_workflow_with_checkpoint.json`

This is the complete, correct testing workflow that:
- Loads test data
- Creates UNet2D architecture
- **Loads trained weights from `checkpoints/best_model.pt`**
- Runs predictions on test set
- Calculates metrics (accuracy, dice, etc.)

### Example 2: Testing Multiple Checkpoints

You can create multiple workflows to test different checkpoints:

**test_best_model.json**:
```json
{
  "type": "CheckpointLoader",
  "config": {
    "checkpoint_path": "checkpoints/best_model.pt"
  }
}
```

**test_epoch_10.json**:
```json
{
  "type": "CheckpointLoader",
  "config": {
    "checkpoint_path": "checkpoints/checkpoint_epoch_10.pt"
  }
}
```

Compare which checkpoint performs best on your test set!

---

## Common Mistakes

### Mistake 1: No Checkpoint Loader

❌ **Wrong**:
```
unet_model → predictor
```

✓ **Correct**:
```
unet_model → checkpoint_loader → predictor
```

**Why**: Without CheckpointLoader, the model has random weights.

### Mistake 2: Wrong Port Connection

❌ **Wrong**:
```json
{
  "source_port": "output",  // TENSOR type
  "target_port": "model"    // Expects MODEL type
}
```

✓ **Correct**:
```json
{
  "source_port": "model",   // MODEL type
  "target_port": "model"    // MODEL type
}
```

**Why**: Port types must match. Use `model` output, not `output`.

### Mistake 3: Wrong Checkpoint Path

❌ **Wrong**:
```
checkpoint_path: "best_model.pt"  // Relative to current directory
```

✓ **Correct**:
```
checkpoint_path: "examples/medical_segmentation_pipeline/checkpoints/best_model.pt"
```

**Why**: Checkpoint must exist at the specified path. Use full relative or absolute paths.

### Mistake 4: Checkpoint Doesn't Exist

**Error**: `CheckpointLoader: Checkpoint file not found`

**Solution**:
1. Run training workflow first to create checkpoints
2. Verify checkpoint path is correct
3. Check that training actually saved checkpoints (checkpoint_dir was set)

---

## Training → Testing Pipeline

### Complete Workflow

1. **Train the model**:
   ```bash
   # Load training_workflow.json in GUI
   # Configure trainer with checkpoint_dir
   # Execute workflow (Ctrl+E)
   ```

   **Output**: Checkpoints saved to `checkpoints/` folder

2. **Test the model**:
   ```bash
   # Load testing_workflow_with_checkpoint.json
   # Verify checkpoint_path points to saved checkpoint
   # Execute workflow (Ctrl+E)
   ```

   **Output**: Test metrics (accuracy, dice, loss, etc.)

### Directory Structure

```
examples/medical_segmentation_pipeline/
├── checkpoints/                    ← Created by training
│   ├── best_model.pt              ← Best performing model
│   ├── final_model.pt             ← Model after last epoch
│   ├── checkpoint_epoch_5.pt      ← Periodic checkpoints
│   ├── checkpoint_epoch_10.pt
│   └── ...
├── data/                          ← Dataset
│   ├── train/
│   └── test/
├── training_workflow.json         ← Training workflow
└── testing_workflow_with_checkpoint.json  ← Testing workflow
```

---

## Verifying Loaded Weights

The CheckpointLoader outputs `checkpoint_info` with details about the loaded checkpoint:

```python
checkpoint_info = {
    'epoch': 10,
    'loss': 0.1234,
    'checkpoint_path': 'checkpoints/best_model.pt'
}
```

When executing, you'll see:
```
Loading checkpoint from: checkpoints/best_model.pt
✓ Loaded checkpoint from epoch 10
  Loss at checkpoint: 0.1234
```

This confirms the weights were loaded successfully.

---

## Advanced: Multiple Model Comparison

### Testing Different Architectures

```json
{
  "nodes": [
    {"type": "UNet2D", "name": "unet"},
    {"type": "CheckpointLoader", "name": "unet_loader"},

    {"type": "VNet", "name": "vnet"},
    {"type": "CheckpointLoader", "name": "vnet_loader"},

    {"type": "BatchPredictor", "name": "unet_predictor"},
    {"type": "BatchPredictor", "name": "vnet_predictor"},

    {"type": "MetricsCalculator", "name": "unet_metrics"},
    {"type": "MetricsCalculator", "name": "vnet_metrics"}
  ]
}
```

Compare UNet vs VNet performance on the same test set!

---

## Troubleshooting

### Issue: Random Predictions

**Symptoms**:
- Test accuracy ≈ 50% (random guessing)
- Predictions don't make sense

**Solution**:
- Add CheckpointLoader node between model and predictor
- Verify checkpoint_path is correct
- Ensure checkpoint file exists

### Issue: Checkpoint Not Found

**Error**: `CheckpointLoader: Checkpoint file not found: checkpoints/best_model.pt`

**Solutions**:
1. Run training workflow first to generate checkpoints
2. Check spelling of checkpoint path
3. Use absolute path if relative path doesn't work
4. Verify checkpoint_dir was set in training workflow

### Issue: Model Architecture Mismatch

**Error**: `Error loading state_dict: size mismatch for conv.weight`

**Cause**: The checkpoint was saved from a different model architecture

**Solutions**:
1. Ensure UNet2D configuration matches training (in_channels, out_channels, depth, etc.)
2. Use the same network type (UNet2D/UNet3D/VNet/etc.)
3. Check checkpoint was saved correctly during training

---

## Code Example

### Loading Checkpoint Manually (Python)

If you want to load checkpoints outside the GUI:

```python
import torch
from medical_imaging_framework.core import NodeRegistry
import medical_imaging_framework.nodes

# Create model
model = NodeRegistry.create_node('UNet2D', 'test_model')
model.config['in_channels'] = 1
model.config['out_channels'] = 2
model.execute()  # Initialize the model

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.module.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Training loss at checkpoint: {checkpoint['loss']:.4f}")

# Now use model for inference
model.module.eval()
```

---

## Summary

**Problem**: Original `testing_workflow.json` used random model weights

**Solution**: Use CheckpointLoader node to load trained weights

**Workflow**:
1. Create model (UNet2D, etc.)
2. Load weights with CheckpointLoader
3. Pass loaded model to predictor
4. Evaluate on test data

**Key Connections**:
```
unet_model.model → checkpoint_loader.model
checkpoint_loader.model → predictor.model
```

**Always**:
- Train model first (creates checkpoints)
- Load checkpoint in testing workflow
- Verify checkpoint path is correct
