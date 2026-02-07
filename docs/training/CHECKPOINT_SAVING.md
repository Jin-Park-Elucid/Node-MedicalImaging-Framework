# Checkpoint Saving in Training

## Overview

The TrainerNode now supports automatic checkpoint saving during training. This allows you to:
- Save model checkpoints at regular intervals
- Keep the best performing model
- Resume training from saved checkpoints
- Track training progress over epochs

---

## Quick Start

### 1. Configure Checkpoint Settings

**In the GUI:**
1. Double-click the Trainer node
2. Set **Checkpoint Directory** (e.g., `checkpoints` or `./my_experiment/checkpoints`)
3. Set **Save Every N Epochs** (e.g., `5` to save every 5 epochs)
4. Click OK

**In code:**
```python
trainer.config['checkpoint_dir'] = 'checkpoints'
trainer.config['save_every_n_epochs'] = 5
```

### 2. Run Training

When training executes, checkpoints are automatically saved to the specified directory.

---

## Saved Checkpoint Files

The trainer saves three types of checkpoints:

### 1. Periodic Checkpoints
- **Filename**: `checkpoint_epoch_N.pt` (e.g., `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`)
- **When saved**: Every N epochs (based on `save_every_n_epochs` parameter)
- **Purpose**: Track progress at regular intervals

### 2. Best Model
- **Filename**: `best_model.pt`
- **When saved**: Whenever the validation/training loss improves
- **Purpose**: Keep the best performing model during training

### 3. Final Model
- **Filename**: `final_model.pt`
- **When saved**: At the end of training (after last epoch)
- **Purpose**: Save the model from the final training state

---

## Checkpoint Contents

Each checkpoint file contains:
- `epoch`: Epoch number when saved
- `model_state_dict`: Model weights and parameters
- `optimizer_state_dict`: Optimizer state (momentum, learning rates, etc.)
- `loss`: Loss value at this checkpoint
- `epoch_losses`: List of losses for all epochs up to this point

---

## Parameters

### checkpoint_dir
- **Type**: Text field
- **Default**: `checkpoints`
- **Description**: Directory where checkpoints will be saved
- **Examples**:
  - `checkpoints` - relative path from current directory
  - `./experiments/run1/checkpoints` - nested directory
  - `/absolute/path/to/checkpoints` - absolute path
- **Note**: Directory is created automatically if it doesn't exist

### save_every_n_epochs
- **Type**: Text field
- **Default**: `5`
- **Description**: How often to save periodic checkpoints
- **Examples**:
  - `1` - save every epoch (lots of disk space)
  - `5` - save every 5 epochs (balanced)
  - `10` - save every 10 epochs (minimal disk usage)

---

## Examples

### Example 1: Training for 20 Epochs

**Configuration:**
```
num_epochs: 20
checkpoint_dir: "checkpoints"
save_every_n_epochs: 5
```

**Saved files:**
```
checkpoints/
  ├── checkpoint_epoch_5.pt
  ├── checkpoint_epoch_10.pt
  ├── checkpoint_epoch_15.pt
  ├── checkpoint_epoch_20.pt  (same as final_model.pt)
  ├── best_model.pt
  └── final_model.pt
```

### Example 2: No Checkpoints (Default)

**Configuration:**
```
num_epochs: 10
checkpoint_dir: ""  (empty string)
```

**Result:** No checkpoints saved, training completes normally.

### Example 3: Save Every Epoch

**Configuration:**
```
num_epochs: 10
checkpoint_dir: "frequent_saves"
save_every_n_epochs: 1
```

**Saved files:**
```
frequent_saves/
  ├── checkpoint_epoch_1.pt
  ├── checkpoint_epoch_2.pt
  ├── ...
  ├── checkpoint_epoch_10.pt
  ├── best_model.pt
  └── final_model.pt
```

---

## Loading Checkpoints

### Load a Saved Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Extract information
epoch = checkpoint['epoch']
loss = checkpoint['loss']
epoch_losses = checkpoint['epoch_losses']

# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Load optimizer state (for resuming training)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
```

### Resume Training from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')

# Restore model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Get the epoch to resume from
start_epoch = checkpoint['epoch']

# Continue training from start_epoch
for epoch in range(start_epoch, num_epochs):
    # Training loop...
    pass
```

---

## Best Practices

### Disk Space Management

✓ **Use reasonable intervals**: `save_every_n_epochs = 5` is good for most cases

✓ **Monitor disk usage**: Each checkpoint is ~1-100 MB depending on model size

✓ **Clean up old checkpoints**: Manually delete old periodic checkpoints you don't need

✓ **Keep best_model.pt**: Always keep the best model for final evaluation

### Directory Organization

```
my_project/
  ├── experiments/
  │   ├── experiment_1/
  │   │   └── checkpoints/
  │   ├── experiment_2/
  │   │   └── checkpoints/
  │   └── experiment_3/
  │       └── checkpoints/
  └── final_models/
      └── production_model.pt
```

### Checkpoint Strategy

**For short training runs (< 10 epochs):**
```
save_every_n_epochs: 2-3
checkpoint_dir: "quick_test_checkpoints"
```

**For medium training runs (10-50 epochs):**
```
save_every_n_epochs: 5-10
checkpoint_dir: "checkpoints"
```

**For long training runs (> 50 epochs):**
```
save_every_n_epochs: 10-20
checkpoint_dir: "long_training_checkpoints"
```

---

## Training Output

When checkpoints are enabled, you'll see output like:

```
✓ Checkpoints will be saved to: checkpoints

Epoch [1/20], Loss: 0.6542
  ✓ Saved best model (loss: 0.6542)
Epoch [2/20], Loss: 0.5123
  ✓ Saved best model (loss: 0.5123)
Epoch [3/20], Loss: 0.4876
  ✓ Saved best model (loss: 0.4876)
Epoch [4/20], Loss: 0.4654
  ✓ Saved best model (loss: 0.4654)
Epoch [5/20], Loss: 0.4521
  ✓ Saved checkpoint: checkpoint_epoch_5.pt
  ✓ Saved best model (loss: 0.4521)
...
Epoch [20/20], Loss: 0.2341
  ✓ Saved checkpoint: checkpoint_epoch_20.pt

✓ Training complete. Final model saved to: checkpoints/final_model.pt
```

---

## Troubleshooting

### Issue: Permission Denied

**Error**: Cannot create checkpoint directory

**Solution**:
- Ensure you have write permissions in the directory
- Use absolute paths if relative paths aren't working
- Run the script with appropriate permissions

### Issue: Disk Full

**Error**: No space left on device

**Solution**:
- Increase `save_every_n_epochs` to save less frequently
- Delete old checkpoints you don't need
- Use a different directory with more space

### Issue: Checkpoints Not Saving

**Possible causes**:
- `checkpoint_dir` is empty string (checkpoints disabled)
- Invalid directory path
- Disk space or permission issues

**Check**:
```python
print(f"Checkpoint dir: '{trainer.config.get('checkpoint_dir')}'")
```

If empty or None, checkpoints won't be saved.

---

## Advanced: Custom Checkpoint Loading

### Evaluate Best Model

```python
import torch
from medical_imaging_framework.core import NodeRegistry

# Load best model checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create model and load weights
model = create_your_model()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(test_images)
```

### Compare Multiple Checkpoints

```python
from pathlib import Path
import torch

checkpoint_dir = Path("checkpoints")

# Load all checkpoints
checkpoints = {}
for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
    ckpt = torch.load(ckpt_file)
    checkpoints[ckpt['epoch']] = ckpt['loss']

# Find best epoch
best_epoch = min(checkpoints, key=checkpoints.get)
print(f"Best epoch: {best_epoch}, Loss: {checkpoints[best_epoch]:.4f}")
```

---

## Related Documentation

- [Editing Node Parameters](../gui/EDITING_PARAMETERS.md) - How to configure checkpoint settings in GUI
- [Training Guide](../segmentation/TRAINING.md) - General training information
- [Training vs Inference](../gui/TRAINING_VS_INFERENCE.md) - Workflow patterns

---

## Summary

**To enable checkpoint saving:**
1. Set `checkpoint_dir` parameter in Trainer node (e.g., `"checkpoints"`)
2. Optionally adjust `save_every_n_epochs` (default: 5)
3. Run training - checkpoints saved automatically

**Checkpoint types:**
- **Periodic**: `checkpoint_epoch_N.pt` - every N epochs
- **Best**: `best_model.pt` - best loss so far
- **Final**: `final_model.pt` - end of training

**To load:**
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```
