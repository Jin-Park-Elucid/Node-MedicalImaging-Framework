# AVTE Workflow Guide

## Quick Start

### Step 1: Preprocess Your Data
```bash
cd examples/AVTE
python preprocess_2d_slices.py --num_workers 8
```

### Step 2: Launch GUI with Workflow
```bash
cd ../..
python examples/launch_gui.py
```

### Step 3: Load Workflow
In the GUI:
- File → Open Workflow
- Navigate to `examples/AVTE/config/`
- Start with `simple_training_workflow.json` to validate data
- Then use `training_workflow.json` for actual training

## Workflow Visualization

### Simple Training Workflow
```
┌─────────────────────┐
│  AVTE2DLoader       │
│  (Data)             │
│                     │
│  Outputs:           │
│  • train_loader     │
│  • val_loader       │
│  • test_loader      │
│  • num_train        │───┐
│  • num_val          │───┼─┐
│  • num_test         │───┼─┼─┐
│  • num_channels     │───┼─┼─┼──┐
└─────────────────────┘   │ │ │  │
                          │ │ │  │
                          ▼ ▼ ▼  ▼
                    ┌─────────────────┐
                    │   PrintInfo     │
                    │   (Utils)       │
                    │                 │
                    │  Displays:      │
                    │  • Dataset size │
                    │  • Splits       │
                    │  • Channels     │
                    └─────────────────┘
```

**Purpose**: Validate data loading and check dataset statistics

### Training Workflow
```
┌─────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  AVTE2DLoader       │     │  ModelConfig    │     │ TrainingConfig   │
│  (Data)             │     │  (Model)        │     │  (Training)      │
│                     │     │                 │     │                  │
│  • Load 2D slices   │     │  • U-Net setup  │     │  • Epochs: 100   │
│  • Multi-slice      │     │  • 5 → 64 → 2   │     │  • LR: 0.001     │
│  • Train/Val/Test   │     │  • Dropout: 0.1 │     │  • Adam opt      │
└──────────┬──────────┘     └────────┬────────┘     └────────┬─────────┘
           │                         │                       │
           │ train_loader            │ model                 │ config
           │ val_loader              │                       │
           │ num_channels ──────────▶│                       │
           │                         │                       │
           └─────────────┬───────────┴───────────────────────┘
                         ▼
                  ┌──────────────────┐
                  │     Trainer      │
                  │   (Training)     │
                  │                  │
                  │  • Training loop │
                  │  • Validation    │
                  │  • Checkpointing │
                  │  • Early stop    │
                  └────────┬─────────┘
                           │
                           │ metrics
                           ▼
                  ┌──────────────────┐
                  │  MetricsLogger   │
                  │    (Utils)       │
                  │                  │
                  │  • TensorBoard   │
                  │  • Loss tracking │
                  │  • Dice score    │
                  └──────────────────┘
```

**Purpose**: Complete training pipeline with logging and checkpointing

## Node Configuration Details

### AVTE2DLoader Node

**Location**: Data category
**Outputs**: 7 ports
**Configuration**: 9 fields

```
┌─────────────────────────────────────────────┐
│           AVTE2DLoader Config               │
├─────────────────────────────────────────────┤
│ data_dir:       /path/to/2D_data            │ ← UPDATE THIS
│ batch_size:     16                          │ ← Adjust for GPU
│ train_ratio:    0.8                         │
│ val_ratio:      0.1                         │
│ random_seed:    42                          │
│ num_workers:    4                           │
│ shuffle_train:  True                        │
│ pin_memory:     True                        │
│ load_to_memory: False                       │
└─────────────────────────────────────────────┘
```

**Critical Settings**:
- `data_dir`: Must point to preprocessed data directory
- `batch_size`: Depends on GPU memory (8-32 typical)
- `num_workers`: Set to 4-8 for optimal performance

### Model Configuration

```
┌─────────────────────────────────────────────┐
│            U-Net Configuration              │
├─────────────────────────────────────────────┤
│ in_channels:  5  (from AVTE2DLoader)        │ ← Auto-set
│ out_channels: 2  (background + vessel)      │
│ features:     [64, 128, 256, 512]           │
│ dropout:      0.1                           │
└─────────────────────────────────────────────┘
```

**Architecture**:
```
Input (5 channels)
    ↓
Encoder Block 1 (64 features)
    ↓ MaxPool
Encoder Block 2 (128 features)
    ↓ MaxPool
Encoder Block 3 (256 features)
    ↓ MaxPool
Encoder Block 4 (512 features)
    ↓ Bottleneck
Decoder Block 4 (256 features) ← Skip connection
    ↑ UpConv
Decoder Block 3 (128 features) ← Skip connection
    ↑ UpConv
Decoder Block 2 (64 features)  ← Skip connection
    ↑ UpConv
Decoder Block 1
    ↓
Output (2 classes)
```

### Training Configuration

```
┌─────────────────────────────────────────────┐
│         Training Configuration              │
├─────────────────────────────────────────────┤
│ num_epochs:      100                        │
│ learning_rate:   0.001                      │
│ optimizer:       Adam                       │
│ loss_function:   CrossEntropyLoss           │
│ device:          cuda                       │
│ checkpoint_dir:  /path/to/checkpoints       │ ← UPDATE THIS
│ log_interval:    10                         │
│ save_interval:   5                          │
└─────────────────────────────────────────────┘
```

## Data Flow

### 1. Preprocessing Phase
```
Raw NIfTI (3D)
    ↓ preprocess_2d_slices.py
2D Slices (.npz files)
    ↓
Stored on disk
```

### 2. Loading Phase
```
.npz files
    ↓ AVTE2DLoader
PyTorch DataLoader
    ↓
Batched tensors
```

### 3. Training Phase
```
Batch of images (B, 5, H, W)
    ↓ Model forward
Predictions (B, 2, H, W)
    ↓ Loss computation
Gradients
    ↓ Backpropagation
Updated weights
```

### 4. Validation Phase
```
Validation batch
    ↓ Model forward (no grad)
Predictions
    ↓ Metrics computation
Dice score, IoU, Accuracy
    ↓ Early stopping check
Continue or stop training
```

## Execution Flow

### Simple Workflow Execution

1. **Load Workflow**
   - Open `simple_training_workflow.json`
   - GUI displays 2 nodes connected

2. **Configure**
   - Update `data_dir` in AVTE2DLoader

3. **Execute**
   - Click Execute button
   - Nodes run in topological order

4. **View Results**
   - Console shows dataset statistics
   - Verify numbers are correct

### Training Workflow Execution

1. **Load Workflow**
   - Open `training_workflow.json`
   - GUI displays 5 nodes connected

2. **Configure Paths**
   - AVTE2DLoader: `data_dir`
   - TrainingConfig: `checkpoint_dir`
   - MetricsLogger: `log_dir`

3. **Configure Hyperparameters**
   - Batch size (based on GPU)
   - Learning rate
   - Number of epochs

4. **Execute**
   - Click Execute button
   - Training starts
   - Progress in console

5. **Monitor**
   - Open TensorBoard
   - Watch metrics in real-time
   - Check checkpoints

6. **Results**
   - Best model saved
   - Metrics logged
   - Training curves available

## Common Workflow Modifications

### Change Batch Size
```json
{
  "config": {
    "batch_size": "32"  // Increase from 16
  }
}
```
**When**: You have more GPU memory

### Enable Memory Loading
```json
{
  "config": {
    "load_to_memory": "True"  // Change from False
  }
}
```
**When**: You have sufficient RAM (~100GB+) and want faster training

### Adjust Learning Rate
```json
{
  "config": {
    "learning_rate": "0.0001"  // Decrease from 0.001
  }
}
```
**When**: Training is unstable or overshooting

### Change Split Ratios
```json
{
  "config": {
    "train_ratio": "0.7",  // More validation data
    "val_ratio": "0.2"
  }
}
```
**When**: You want more rigorous validation

## Troubleshooting

### "Data directory not found"
**Fix**: Update `data_dir` in AVTE2DLoader config
```json
"data_dir": "/your/actual/path/to/2D_data"
```

### "CUDA out of memory"
**Fix**: Reduce batch size
```json
"batch_size": "8"  // Reduce from 16
```

### "No improvement in validation"
**Checks**:
1. Verify data is loading correctly
2. Check loss is decreasing
3. Try lower learning rate
4. Increase training time

### "Training too slow"
**Optimizations**:
1. Increase `num_workers` (4 → 8)
2. Enable `pin_memory` (True)
3. Increase `batch_size` (if GPU allows)
4. Consider `load_to_memory` (if RAM allows)

## Performance Expectations

### Data Loading
- **First epoch**: 10-30 seconds (loading from disk)
- **Subsequent epochs**: <1 second (cached in memory)
- **With load_to_memory**: Instant after first load

### Training Speed (per epoch)
- **CPU**: 30-60 minutes
- **GPU (8GB)**: 5-10 minutes
- **GPU (16GB+)**: 2-5 minutes

### Convergence
- **Epochs to convergence**: 50-80
- **Total training time**: 4-8 hours (GPU)
- **Expected Dice score**: >0.85

## Next Steps

After successful training:

1. **Evaluate on Test Set**
   - Use test_loader output
   - Compute final metrics

2. **Visualize Results**
   - Plot predictions
   - Compare with ground truth

3. **Export Model**
   - Save best checkpoint
   - Convert to inference format

4. **Deploy**
   - Create inference pipeline
   - Test on new data

## Related Files

- `README.md` - Configuration options
- `simple_training_workflow.json` - Testing workflow
- `training_workflow.json` - Full training workflow
- `../docs/GUI_NODE_GUIDE.md` - Node documentation

---

**Last Updated**: 2026-02-08
