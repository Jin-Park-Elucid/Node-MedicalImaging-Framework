# Port Types and Connection Guide

## Overview

This guide explains port data types and how to correctly connect nodes in the framework.

## Problem: "Cannot connect BATCH to TENSOR" or Similar Errors

If you see errors like:
- "Cannot connect batch to tensor"
- "Incompatible port types"
- Connections refusing to be created

This means you're trying to connect **incompatible data types**.

---

## Port Data Types

### Common Data Types

| DataType | Used For | Example Ports |
|----------|----------|---------------|
| **BATCH** | Data loaders, batched data | `data_loader.train_loader` |
| **MODEL** | PyTorch models/modules | `unet_model.model` |
| **TENSOR** | PyTorch tensors (data) | `unet_model.output`, `data.images` |
| **LOSS** | Loss functions | `loss_function.loss_fn` |
| **OPTIMIZER** | Optimizers | `optimizer.optimizer` |
| **ANY** | Any type (flexible) | Various utility nodes |

### Network Node Ports

**All network nodes** (UNet, VNet, SegResNet, DeepLabV3+, TransUNet, UNETR, Swin-UNETR) have:

**Inputs**:
- `input` (TENSOR): For inference/forward pass

**Outputs**:
- `output` (TENSOR): Tensor result from forward pass
- `model` (MODEL): The PyTorch module itself ← **Use this for training!**

---

## Correct Connections for Training Workflow

### ✅ Correct: Network Model → Trainer

```json
{
  "source_node": "unet_model",
  "source_port": "model",        ← MODEL type
  "target_node": "trainer",
  "target_port": "model"          ← MODEL type
}
```

### ❌ Incorrect: Network Output → Trainer

```json
{
  "source_node": "unet_model",
  "source_port": "output",       ← TENSOR type
  "target_node": "trainer",
  "target_port": "model"          ← MODEL type (mismatch!)
}
```

**Error**: "Cannot connect tensor to model"

---

## Training Workflow Connections

Here are the **5 required connections** for a training workflow:

### 1. Data Loader → Trainer

```
data_loader.train_loader (BATCH) → trainer.dataloader (BATCH)
```

### 2. Network Model → Trainer

```
unet_model.model (MODEL) → trainer.model (MODEL)
```
**Important**: Use `.model` port, NOT `.output`!

### 3. Loss Function → Trainer

```
loss_function.loss_fn (LOSS) → trainer.loss_fn (LOSS)
```

### 4. Network Model → Optimizer

```
unet_model.model (MODEL) → optimizer.model (MODEL)
```

### 5. Optimizer → Trainer

```
optimizer.optimizer (OPTIMIZER) → trainer.optimizer (OPTIMIZER)
```

---

## Testing/Inference Workflow Connections

For inference (no training):

### 1. Data Loader → Predictor

```
data_loader.test_loader (BATCH) → predictor.dataloader (BATCH)
```

### 2. Network Model → Predictor

```
unet_model.model (MODEL) → predictor.model (MODEL)
```

### 3. Predictor → Metrics

```
predictor.predictions (TENSOR) → metrics.predictions (TENSOR)
predictor.all_labels (TENSOR) → metrics.labels (TENSOR)
```

---

## How to Check Port Types

### In the GUI:

1. **Hover over a port** (colored circle on a node)
2. **Tooltip appears** showing: `port_name (DATA_TYPE)`

Example tooltips:
- `train_loader (BATCH)`
- `model (MODEL)`
- `output (TENSOR)`
- `loss_fn (LOSS)`

### Port Colors in GUI:

- **Blue circles**: Input ports (left side)
- **Orange circles**: Output ports (right side)

### During Connection:

- **Yellow dashed line**: Valid connection being dragged
- **Error message**: If types don't match, you'll see "Incompatible Port Types"

---

## Common Connection Mistakes

### Mistake 1: Using `output` instead of `model` for Training

❌ **Wrong**:
```
unet_model.output → trainer.model
```

✅ **Correct**:
```
unet_model.model → trainer.model
```

**Why**: `output` is a TENSOR (result of forward pass), but `trainer.model` expects a MODEL (the PyTorch module).

---

### Mistake 2: Connecting Data Directly to Network

❌ **Wrong** (for training):
```
data_loader.train_loader → unet_model.input
```

**Why**: Training workflows use the `Trainer` node to orchestrate data flow. The trainer passes data through the model internally.

✅ **Correct**: Connect data_loader to trainer, model to trainer separately.

---

### Mistake 3: Wrong Data Type for Metrics

❌ **Wrong**:
```
data_loader.test_loader (BATCH) → metrics.labels (TENSOR)
```

✅ **Correct**:
```
predictor.all_labels (TENSOR) → metrics.labels (TENSOR)
```

**Why**: Metrics expects TENSOR data, not BATCH loaders.

---

## Quick Reference: Node Port Summary

### Data Loader (MedicalSegmentationLoader)

**Outputs**:
- `train_loader` (BATCH)
- `test_loader` (BATCH)
- `num_train` (ANY)
- `num_test` (ANY)

### Network Nodes (UNet, VNet, etc.)

**Inputs**:
- `input` (TENSOR)

**Outputs**:
- `output` (TENSOR) - for inference
- `model` (MODEL) - **for training** ← Use this!

### Loss Function

**Outputs**:
- `loss_fn` (LOSS)

### Optimizer

**Inputs**:
- `model` (MODEL)

**Outputs**:
- `optimizer` (OPTIMIZER)

### Trainer

**Inputs**:
- `dataloader` (BATCH)
- `model` (MODEL)
- `loss_fn` (LOSS)
- `optimizer` (OPTIMIZER)

**Outputs**:
- None (trains the model in-place)

### Predictor (BatchPredictor)

**Inputs**:
- `model` (MODEL)
- `dataloader` (BATCH)

**Outputs**:
- `predictions` (TENSOR)
- `all_predictions` (TENSOR)
- `all_labels` (TENSOR)

### Metrics Calculator

**Inputs**:
- `predictions` (TENSOR)
- `labels` (TENSOR)

**Outputs**:
- `dice` (ANY)
- `iou` (ANY)
- `accuracy` (ANY)

---

## Validation Rules

Connections are **only valid** when:

1. ✅ Source is OUTPUT port (orange circle)
2. ✅ Target is INPUT port (blue circle)
3. ✅ **Data types match exactly**
4. ✅ Ports are on different nodes
5. ✅ No duplicate connection already exists

---

## Debugging Connection Errors

### Error: "Cannot connect X to Y"

**Steps to fix**:

1. **Hover over both ports** to see their types
2. **Check types match** (TENSOR → TENSOR, MODEL → MODEL, etc.)
3. **Verify direction** (output → input)
4. **Check for typos** in port names

### Error: "Connection already exists"

**Fix**: Delete the existing connection first, then create new one.

### Error: Connection disappears after release

**Causes**:
- Released on wrong port type
- Released on invalid target
- Data types don't match

**Fix**: Try again, ensuring you release on a compatible input port.

---

## Updated Workflow Files

All example workflows have been updated to use the correct `model` port:

- ✅ `training_workflow.json`
- ✅ `vnet_training.json`
- ✅ `transunet_training.json`
- ✅ `unetr_training.json`
- ✅ `swin_unetr_training.json`
- ✅ `deeplabv3plus_training.json`

If you load these workflows, the connections should work correctly now!

---

## Visual Guide

### Training Workflow Connection Pattern:

```
┌────────────────┐
│  data_loader   │
│                │
│ train_loader ●─┼─────┐
└────────────────┘     │ (BATCH)
                       │
┌────────────────┐     │
│  unet_model    │     │
│                │     │
│   model ●──────┼──┐  │
└────────────────┘  │  │ (MODEL)
                    │  │
┌────────────────┐  │  │
│ loss_function  │  │  │
│                │  │  │
│  loss_fn ●─────┼──┼──┼─┐
└────────────────┘  │  │ │ (LOSS)
                    │  │ │
┌────────────────┐  │  │ │
│   optimizer    │  │  │ │
│                │  │  │ │
│  optimizer ●───┼──┼──┼─┼─┐
└────────────────┘  │  │ │ │ (OPTIMIZER)
                    │  │ │ │
                    ▼  ▼ ▼ ▼
              ┌─────────────────┐
              │     trainer     │
              ├─────────────────┤
              │ ● dataloader    │
              │ ● model         │
              │ ● loss_fn       │
              │ ● optimizer     │
              └─────────────────┘

Also: unet_model.model ──(MODEL)──> optimizer.model
```

---

## Summary

**Key Takeaway**: When connecting network nodes (UNet, VNet, etc.) to Trainer or Optimizer:

- ✅ **Use the `model` output port** (MODEL type)
- ❌ **Don't use the `output` port** (TENSOR type - that's for inference only)

**Port types must match** for connections to work!

---

## Next Steps

- [Creating Connections Guide](CREATING_CONNECTIONS.md) - How to drag and create connections
- [Visual GUI Guide](VISUAL_GUI_COMPLETE.md) - Complete GUI documentation
- [Training Guide](../segmentation/TRAINING.md) - Training workflows and parameters
