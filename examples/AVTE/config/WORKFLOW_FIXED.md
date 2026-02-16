# AVTE Training Workflow - Fixed

## What Was Wrong

### Issue 1: Empty Links Array
```json
"links": []
```
The connections weren't saved or created properly. All nodes were present but not connected.

### Issue 2: Wrong Node Type Names
- Used: `"AVTE2DLoaderNode"`
- Should be: `"AVTE2DLoader"`
- Used: `"UNet2DNode"`, `"TrainerNode"`, etc.
- Should be: `"UNet2D"`, `"Trainer"`, etc.

### Issue 3: Missing Required Connections
The error message showed:
```
Required input 'model' on node 'Trainer_2' is not connected
Required input 'dataloader' on node 'Trainer_2' is not connected
Required input 'model' on node 'Optimizer_4' is not connected
```

## What Was Fixed

### Fixed Workflow Structure

**5 Nodes**:
1. `avte_dataloader` (AVTE2DLoader)
2. `unet_model` (UNet2D)
3. `loss_function` (LossFunction)
4. `optimizer` (Optimizer)
5. `trainer` (Trainer)

**6 Connections**:

1. **Data → Trainer**
   ```json
   {
     "source_node": "avte_dataloader",
     "source_port": "train_loader",
     "target_node": "trainer",
     "target_port": "dataloader"
   }
   ```

2. **Validation Data → Trainer**
   ```json
   {
     "source_node": "avte_dataloader",
     "source_port": "val_loader",
     "target_node": "trainer",
     "target_port": "val_dataloader"
   }
   ```

3. **Model → Optimizer**
   ```json
   {
     "source_node": "unet_model",
     "source_port": "model",
     "target_node": "optimizer",
     "target_port": "model"
   }
   ```

4. **Model → Trainer**
   ```json
   {
     "source_node": "unet_model",
     "source_port": "model",
     "target_node": "trainer",
     "target_port": "model"
   }
   ```

5. **Loss Function → Trainer**
   ```json
   {
     "source_node": "loss_function",
     "source_port": "loss_fn",
     "target_node": "trainer",
     "target_port": "loss_fn"
   }
   ```

6. **Optimizer → Trainer**
   ```json
   {
     "source_node": "optimizer",
     "source_port": "optimizer",
     "target_node": "trainer",
     "target_port": "optimizer"
   }
   ```

## Visual Workflow

```
┌──────────────────────┐
│  avte_dataloader     │
│  (AVTE2DLoader)      │
│                      │
│  train_loader ●──────┼─────┐
│  val_loader ●────────┼───┐ │
└──────────────────────┘   │ │
                           │ │
┌──────────────────────┐   │ │
│  unet_model          │   │ │
│  (UNet2D)            │   │ │
│                      │   │ │
│  model ●─────────────┼─┐ │ │
└──────────────────────┘ │ │ │
                         │ │ │
┌──────────────────────┐ │ │ │
│  loss_function       │ │ │ │
│  (LossFunction)      │ │ │ │
│                      │ │ │ │
│  loss_fn ●───────────┼─┼─┼─┼─┐
└──────────────────────┘ │ │ │ │
                         │ │ │ │
┌──────────────────────┐ │ │ │ │
│  optimizer           │ │ │ │ │
│  (Optimizer)         │ │ │ │ │
│                      │ │ │ │ │
│  ● model ◄───────────┼─┘ │ │ │
│  optimizer ●─────────┼───┼─┼─┼─┐
└──────────────────────┘   │ │ │ │
                           │ │ │ │
                           ▼ ▼ ▼ ▼ ▼
                    ┌──────────────────────┐
                    │      trainer         │
                    │      (Trainer)       │
                    ├──────────────────────┤
                    │  ● model             │
                    │  ● dataloader        │
                    │  ● val_dataloader    │
                    │  ● loss_fn           │
                    │  ● optimizer         │
                    └──────────────────────┘
```

## Configuration Summary

### AVTE2DLoader
- **data_dir**: `/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data`
- **batch_size**: 16
- **train_ratio**: 0.8
- **val_ratio**: 0.1
- **num_workers**: 4

### UNet2D
- **in_channels**: 5 (multi-slice context)
- **out_channels**: 2 (background + vessel)
- **base_channels**: 64
- **depth**: 4

### LossFunction
- **loss_type**: dice

### Optimizer
- **optimizer_type**: Adam
- **learning_rate**: 0.001

### Trainer
- **num_epochs**: 10
- **checkpoint_dir**: checkpoints/avte_2d
- **save_every_n_epochs**: 5

## How to Use

### 1. Reload the Workflow

In the GUI:
```
File → Open Workflow → examples/AVTE/config/training_workflow.json
```

You should now see:
```
Nodes: 5
Connections: 6

Loaded Nodes:
  • avte_dataloader (AVTE2DLoader)
  • unet_model (UNet2D)
  • loss_function (LossFunction)
  • optimizer (Optimizer)
  • trainer (Trainer)
```

### 2. Verify Connections

Check that all nodes show connections (lines between them):
- Yellow/orange lines = connections
- All nodes should be connected to the trainer

### 3. Update Data Path (if needed)

Click on `avte_dataloader` node and verify/update:
```
data_dir: /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data
```

### 4. Execute Workflow

Click **"Execute Workflow"** button.

Expected behavior:
1. AVTE2DLoader loads and splits data
2. UNet2D model is created with 5 input channels
3. Loss function (Dice) is initialized
4. Optimizer (Adam) is created
5. Trainer starts training for 10 epochs
6. Checkpoints saved every 5 epochs to `checkpoints/avte_2d/`

### 5. Monitor Training

Watch the console for:
```
Epoch 1/10
  Train Loss: X.XXX
  Val Loss: X.XXX
  Val Dice: X.XXX

Epoch 2/10
  Train Loss: X.XXX
  ...
```

## Troubleshooting

### Still Getting Connection Errors?

If you still see:
```
Required input 'model' on node 'trainer' is not connected
```

**In GUI**:
1. Delete all connections (right-click on lines)
2. Manually recreate connections by dragging:
   - Drag from output port (right side, orange circle)
   - To input port (left side, blue circle)
3. Follow the 6 connections listed above
4. Hover over ports to verify types match

### Can't Connect Ports?

If you get "Cannot connect X to Y":
- Check port types match (hover to see types)
- Use `unet_model.model` not `unet_model.output`
- Ensure dragging from output (right) to input (left)

### Data Path Error?

If you see "Data directory not found":
```bash
# Verify path exists
ls /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data/*.npz | head -5

# If missing, preprocess data
cd examples/AVTE
python preprocess_2d_slices.py --num_workers 8
```

## Next Steps

After successful training:

1. **Check Checkpoints**
   ```bash
   ls checkpoints/avte_2d/
   ```

2. **Increase Epochs**
   - Edit trainer node
   - Change `num_epochs` to 50 or 100
   - Re-execute workflow

3. **Add TensorBoard Logging**
   - Add a `TensorBoardLogger` node
   - Connect trainer outputs to logger

4. **Test the Model**
   - Create a new workflow for testing
   - Use `BatchPredictor` node
   - Load checkpoint and test on test_loader

5. **Visualize Results**
   - Add `SegmentationOverlay` node
   - Generate prediction visualizations

## Connection Checklist

Before executing, verify all 6 connections exist:

- [ ] avte_dataloader.train_loader → trainer.dataloader
- [ ] avte_dataloader.val_loader → trainer.val_dataloader
- [ ] unet_model.model → optimizer.model
- [ ] unet_model.model → trainer.model
- [ ] loss_function.loss_fn → trainer.loss_fn
- [ ] optimizer.optimizer → trainer.optimizer

If all 6 are checked, you should be able to execute successfully!

---

**Last Updated**: 2026-02-08
**Status**: ✅ Fixed and Validated
