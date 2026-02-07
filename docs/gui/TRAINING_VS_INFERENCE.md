# Training vs Inference Workflows

## Overview

This guide explains the difference between **training** and **inference** workflows, and how to connect nodes correctly for each use case.

---

## Key Concept: Two Types of Workflows

### 1. Training Workflow

**Purpose**: Train a model on data

**Key characteristics**:
- Model is **not** directly connected to data
- Trainer orchestrates everything
- Data flows through the Trainer node
- Network nodes only need their `model` output connected

### 2. Inference/Testing Workflow

**Purpose**: Use a trained model to make predictions

**Key characteristics**:
- Data **is** directly connected to model's `input`
- Model processes data and outputs predictions
- No training occurs

---

## Why Network Input is Optional

Network nodes (UNet, VNet, etc.) have an **optional** `input` port because:

✅ **For Training**: Input is NOT needed
   - Trainer handles passing data through model
   - Only `model` output needs to be connected

✅ **For Inference**: Input IS needed
   - Data flows directly through the network
   - Both `input` and `output` are used

---

## Training Workflow

### Architecture

```
┌──────────────┐
│ data_loader  │  ← Loads batches
└──────┬───────┘
       │ train_loader (BATCH)
       │
       │         ┌──────────────┐
       │         │ unet_model   │  ← Defines model architecture
       │         └──────┬───────┘
       │                │ model (MODEL)
       │                │
       │         ┌──────┴───────┐
       │         │              │
       ▼         ▼              ▼
   ┌────────────────────────────────┐
   │         TRAINER                │  ← Orchestrates training
   │                                │
   │ 1. Get batch from data_loader  │
   │ 2. Pass through model          │
   │ 3. Compute loss                │
   │ 4. Update with optimizer       │
   └────────────────────────────────┘
```

### Required Connections

1. **data_loader.train_loader** → **trainer.dataloader**
   - Type: BATCH → BATCH ✓

2. **unet_model.model** → **trainer.model**
   - Type: MODEL → MODEL ✓
   - ⚠️ **NOT** unet_model.output!

3. **loss_function.loss_fn** → **trainer.loss_fn**
   - Type: LOSS → LOSS ✓

4. **unet_model.model** → **optimizer.model**
   - Type: MODEL → MODEL ✓
   - Same port as #2 (one output, multiple connections)

5. **optimizer.optimizer** → **trainer.optimizer**
   - Type: OPTIMIZER → OPTIMIZER ✓

### What NOT to Connect

❌ **DON'T** connect: `data_loader.train_loader` → `unet_model.input`
   - Wrong! Trainer handles this internally

❌ **DON'T** connect: `unet_model.output` → `trainer.model`
   - Wrong type! output is TENSOR, trainer.model expects MODEL

❌ **DON'T** connect: `unet_model.input` to anything
   - Leave it unconnected for training!

---

## Inference Workflow

### Architecture

```
┌──────────────┐
│ data_loader  │
└──────┬───────┘
       │ test_loader (BATCH)
       ▼
   ┌─────────────┐
   │  predictor  │  ← Runs inference
   └──────┬──────┘
          │ predictions (TENSOR)
          ▼
      ┌──────────┐
      │ metrics  │  ← Evaluates results
      └──────────┘
```

OR direct connection:

```
┌──────────────┐
│    images    │  ← Single image or batch
└──────┬───────┘
       │ (TENSOR)
       ▼
   ┌─────────────┐
   │ unet_model  │  ← Input connected!
   │             │
   │ input ●     │  ← Used for inference
   │     output ●┼──→ predictions (TENSOR)
   └─────────────┘
```

### Required Connections (using Predictor)

1. **unet_model.model** → **predictor.model**
   - Type: MODEL → MODEL ✓

2. **data_loader.test_loader** → **predictor.dataloader**
   - Type: BATCH → BATCH ✓

3. **predictor.predictions** → **metrics.predictions**
   - Type: TENSOR → TENSOR ✓

4. **predictor.all_labels** → **metrics.labels**
   - Type: TENSOR → TENSOR ✓

### What's Different

✓ **Model input is NOT connected** (Predictor handles it)
✓ **No Trainer, Loss, or Optimizer**
✓ **Predictions flow to Metrics for evaluation**

---

## Network Node Ports Explained

### All Network Nodes Have:

**Input**:
- `input` (TENSOR) - **Optional**
  - Use for: Direct inference
  - Don't use for: Training workflows

**Outputs**:
- `output` (TENSOR)
  - Use for: Getting predictions in inference
  - Don't use for: Connecting to Trainer

- `model` (MODEL)
  - Use for: Connecting to Trainer/Optimizer
  - Use for: Connecting to Predictor
  - This is the PyTorch nn.Module itself

---

## Common Mistakes

### Mistake 1: Connecting Data to Model in Training

❌ **Wrong**:
```
data_loader.train_loader → unet_model.input
```

✓ **Correct**:
```
data_loader.train_loader → trainer.dataloader
unet_model.model → trainer.model
```

**Why**: Trainer orchestrates the training loop. It gets batches from the dataloader and passes them through the model internally.

---

### Mistake 2: Using `output` Instead of `model`

❌ **Wrong**:
```
unet_model.output → trainer.model
```
Error: "Cannot connect tensor to model"

✓ **Correct**:
```
unet_model.model → trainer.model
```

**Why**:
- `output` is a TENSOR (prediction result)
- `model` is a MODEL (the PyTorch module)
- Trainer expects the MODEL, not a tensor

---

### Mistake 3: Connecting Input in Training Workflow

❌ **Wrong**:
```
Error: "Required input 'input' on node 'unet_model' is not connected"
```
Then trying to connect something to unet_model.input

✓ **Correct**:
- **Don't connect anything** to unet_model.input in training
- The `input` port is optional (only needed for inference)
- Trainer handles passing data through the model

---

## Port Type Quick Reference

| Port | Type | Use In Training? | Use In Inference? |
|------|------|------------------|-------------------|
| `unet_model.input` | TENSOR | ❌ No | ✓ Yes |
| `unet_model.output` | TENSOR | ❌ No | ✓ Yes |
| `unet_model.model` | MODEL | ✓ Yes | ✓ Yes |
| `data_loader.train_loader` | BATCH | ✓ Yes | ❌ No |
| `data_loader.test_loader` | BATCH | ❌ No | ✓ Yes |
| `trainer.dataloader` | BATCH | ✓ Yes | ❌ No |
| `trainer.model` | MODEL | ✓ Yes | ❌ No |
| `predictor.model` | MODEL | ❌ No | ✓ Yes |

---

## Workflow Comparison

### Training Workflow Checklist

- [ ] Data loader → Trainer.dataloader
- [ ] Network.model → Trainer.model
- [ ] Network.model → Optimizer.model
- [ ] Loss → Trainer.loss_fn
- [ ] Optimizer → Trainer.optimizer
- [ ] Network.input is **NOT** connected ✓

### Inference Workflow Checklist

- [ ] Data loader → Predictor.dataloader
- [ ] Network.model → Predictor.model
- [ ] Predictor → Metrics
- [ ] Network.input is **NOT** connected (Predictor handles it) ✓

---

## How Trainer Works Internally

When you execute a training workflow:

```python
# Pseudocode of what Trainer does:

for epoch in range(num_epochs):
    for batch in dataloader:  # From data_loader.train_loader
        images, labels = batch

        # Forward pass through model
        predictions = model(images)  # Using unet_model.model

        # Compute loss
        loss_value = loss_fn(predictions, labels)  # Using loss_function.loss_fn

        # Backward pass
        optimizer.zero_grad()  # Using optimizer.optimizer
        loss_value.backward()
        optimizer.step()
```

Notice:
- Trainer gets batches from dataloader
- Trainer passes them through model
- You **don't** connect data to model directly

---

## Summary

### Training
- **Model input**: Leave unconnected
- **Model.model**: Connect to Trainer and Optimizer
- **Data**: Connect to Trainer.dataloader
- **Trainer orchestrates everything**

### Inference
- **Model input**: Can be connected for direct inference
- **Model.model**: Connect to Predictor
- **Data**: Connect to Predictor.dataloader or directly to model.input
- **Get predictions from model.output**

---

## Quick Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "Required input 'input' on node 'unet_model' is not connected" | Old code (before fix) | Update to latest code |
| "Cannot connect batch to tensor" | Trying data_loader → model.input | Connect to Trainer instead |
| "Cannot connect tensor to model" | Using model.output instead of model.model | Use model.model port |
| Connection types don't match | Wrong ports selected | Check tooltip to verify types |

---

## Example Workflows

### ✓ Correct Training Workflow

```json
{
  "links": [
    {"source": "data_loader", "source_port": "train_loader",
     "target": "trainer", "target_port": "dataloader"},

    {"source": "unet_model", "source_port": "model",
     "target": "trainer", "target_port": "model"},

    {"source": "loss_function", "source_port": "loss_fn",
     "target": "trainer", "target_port": "loss_fn"},

    {"source": "unet_model", "source_port": "model",
     "target": "optimizer", "target_port": "model"},

    {"source": "optimizer", "source_port": "optimizer",
     "target": "trainer", "target_port": "optimizer"}
  ]
}
```

Note: `unet_model.input` is **not** in the links!

---

## Next Steps

- [Port Types Guide](PORT_TYPES_GUIDE.md) - Understanding data types
- [Creating Connections](CREATING_CONNECTIONS.md) - How to drag connections
- [Training Guide](../segmentation/TRAINING.md) - Training parameters and strategies
