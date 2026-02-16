# AVTE Workflow JSON Format

## Overview

Workflow files define node configurations and connections for the GUI pipeline editor. They follow a specific JSON schema required by the Medical Imaging Framework.

## Required Format

### Basic Structure

```json
{
  "name": "Workflow Name",
  "nodes": [
    {
      "type": "NodeTypeName",
      "name": "unique_node_identifier",
      "position": [x, y],
      "config": {
        "param1": "value1",
        "param2": "value2"
      },
      "is_composite": false,
      "collapsed": false
    }
  ],
  "links": [
    {
      "source": "source_node_name",
      "source_port": "output_port_name",
      "target": "target_node_name",
      "target_port": "input_port_name"
    }
  ]
}
```

## Field Descriptions

### Top Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Display name for the workflow |
| `nodes` | array | Yes | List of node definitions |
| `links` | array | Yes | List of connections between nodes |

### Node Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Node class name (e.g., "AVTE2DLoader") |
| `name` | string | Yes | Unique identifier for this node instance |
| `position` | array[2] | Yes | [x, y] coordinates on canvas |
| `config` | object | Yes | Node-specific configuration parameters |
| `is_composite` | boolean | Yes | Whether node contains sub-nodes (usually false) |
| `collapsed` | boolean | Yes | Whether node is collapsed in UI (usually false) |

### Link Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Name of source node |
| `source_port` | string | Yes | Output port name on source node |
| `target` | string | Yes | Name of target node |
| `target_port` | string | Yes | Input port name on target node |

## AVTE2DLoader Node Configuration

### Complete Example

```json
{
  "name": "AVTE 2D Training Workflow",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "avte_dataloader",
      "position": [100, 200],
      "config": {
        "data_dir": "/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data",
        "batch_size": "16",
        "train_ratio": "0.8",
        "val_ratio": "0.1",
        "random_seed": "42",
        "num_workers": "4",
        "shuffle_train": "True",
        "pin_memory": "True",
        "load_to_memory": "False"
      },
      "is_composite": false,
      "collapsed": false
    }
  ],
  "links": []
}
```

### Configuration Parameters

All config values must be strings (even numbers and booleans):

| Parameter | Type | Example | Notes |
|-----------|------|---------|-------|
| `data_dir` | string | "/path/to/data" | Path to preprocessed 2D data |
| `batch_size` | string | "16" | Number as string |
| `train_ratio` | string | "0.8" | Float as string (0.0-1.0) |
| `val_ratio` | string | "0.1" | Float as string (0.0-1.0) |
| `random_seed` | string | "42" | Integer as string |
| `num_workers` | string | "4" | Integer as string |
| `shuffle_train` | string | "True" | Boolean as string |
| `pin_memory` | string | "True" | Boolean as string |
| `load_to_memory` | string | "False" | Boolean as string |

### Available Output Ports

When connecting to other nodes, AVTE2DLoader provides:

| Port Name | Type | Description |
|-----------|------|-------------|
| `train_loader` | BATCH | PyTorch DataLoader for training |
| `val_loader` | BATCH | PyTorch DataLoader for validation |
| `test_loader` | BATCH | PyTorch DataLoader for testing |
| `num_train` | ANY | Number of training samples |
| `num_val` | ANY | Number of validation samples |
| `num_test` | ANY | Number of test samples |
| `num_channels` | ANY | Number of input channels |

## Adding Links/Connections

### Example: Connecting to a Model

```json
{
  "name": "AVTE Training with Model",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "data_loader",
      "position": [100, 200],
      "config": { ... }
    },
    {
      "type": "UNet2DNode",
      "name": "unet_model",
      "position": [400, 200],
      "config": {
        "in_channels": 5,
        "out_channels": 2,
        "base_channels": 64
      }
    }
  ],
  "links": [
    {
      "source": "data_loader",
      "source_port": "train_loader",
      "target": "unet_model",
      "target_port": "data"
    },
    {
      "source": "data_loader",
      "source_port": "num_channels",
      "target": "unet_model",
      "target_port": "in_channels"
    }
  ]
}
```

## Common Patterns

### Pattern 1: Single Node (Testing)

```json
{
  "name": "Test Data Loading",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "loader",
      "position": [100, 200],
      "config": { ... },
      "is_composite": false,
      "collapsed": false
    }
  ],
  "links": []
}
```

**Use**: Validate data loading and configuration

### Pattern 2: Data + Model

```json
{
  "name": "Data and Model Setup",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "loader",
      "position": [100, 200],
      "config": { ... }
    },
    {
      "type": "UNet2DNode",
      "name": "model",
      "position": [400, 200],
      "config": { ... }
    }
  ],
  "links": [
    {
      "source": "loader",
      "source_port": "train_loader",
      "target": "model",
      "target_port": "data"
    }
  ]
}
```

**Use**: Set up data pipeline and model

### Pattern 3: Complete Training Pipeline

```json
{
  "name": "Full Training",
  "nodes": [
    { "type": "AVTE2DLoader", "name": "loader", ... },
    { "type": "UNet2DNode", "name": "model", ... },
    { "type": "LossFunctionNode", "name": "loss", ... },
    { "type": "OptimizerNode", "name": "optimizer", ... },
    { "type": "TrainerNode", "name": "trainer", ... }
  ],
  "links": [
    { "source": "loader", "source_port": "train_loader", "target": "trainer", "target_port": "train_data" },
    { "source": "loader", "source_port": "val_loader", "target": "trainer", "target_port": "val_data" },
    { "source": "model", "source_port": "model", "target": "trainer", "target_port": "model" },
    { "source": "loss", "source_port": "loss_fn", "target": "trainer", "target_port": "loss_function" },
    { "source": "optimizer", "source_port": "optimizer", "target": "trainer", "target_port": "optimizer" }
  ]
}
```

**Use**: Production training setup

## Creating Custom Workflows

### Step 1: Start with Template

```bash
cp simple_training_workflow.json my_workflow.json
```

### Step 2: Edit Node Configuration

Update the config values:
```json
"config": {
  "batch_size": "32",  // Your value
  "num_workers": "8"   // Your value
}
```

### Step 3: Add More Nodes

Copy node structure and change names:
```json
{
  "type": "NewNodeType",
  "name": "unique_name",  // Must be unique!
  "position": [x, y],     // Choose position
  "config": { ... },
  "is_composite": false,
  "collapsed": false
}
```

### Step 4: Add Connections

Link nodes together:
```json
{
  "source": "node1_name",
  "source_port": "output_port",
  "target": "node2_name",
  "target_port": "input_port"
}
```

### Step 5: Validate

```bash
python3 -m json.tool my_workflow.json
```

## Common Mistakes

### ❌ Wrong Position Format
```json
"position": {"x": 100, "y": 200}  // Wrong!
```
✅ Correct:
```json
"position": [100, 200]  // Right!
```

### ❌ Missing Required Fields
```json
{
  "type": "AVTE2DLoader",
  "position": [100, 200]
  // Missing: name, config, is_composite, collapsed
}
```
✅ Correct:
```json
{
  "type": "AVTE2DLoader",
  "name": "loader",
  "position": [100, 200],
  "config": {},
  "is_composite": false,
  "collapsed": false
}
```

### ❌ Config Values as Numbers
```json
"config": {
  "batch_size": 16  // Wrong! Should be string
}
```
✅ Correct:
```json
"config": {
  "batch_size": "16"  // Right! All values are strings
}
```

### ❌ Using "id" Instead of "name"
```json
{
  "type": "AVTE2DLoader",
  "id": "node_1"  // Wrong! Use "name"
}
```
✅ Correct:
```json
{
  "type": "AVTE2DLoader",
  "name": "node_1"  // Right!
}
```

### ❌ Using "connections" Instead of "links"
```json
{
  "nodes": [...],
  "connections": [...]  // Wrong!
}
```
✅ Correct:
```json
{
  "nodes": [...],
  "links": [...]  // Right!
}
```

## Troubleshooting

### Error: KeyError: 'name'
**Cause**: Node is missing the "name" field
**Fix**: Add unique "name" to each node

### Error: KeyError: 'type'
**Cause**: Node is missing the "type" field
**Fix**: Add "type" field with correct node class name

### Error: Invalid JSON
**Cause**: Syntax error in JSON
**Fix**: Validate with `python3 -m json.tool file.json`

### Error: Node not found
**Cause**: Node type doesn't exist in registry
**Fix**: Verify node is registered, check spelling

### Error: Port not found
**Cause**: Port name doesn't exist on node
**Fix**: Check available ports for that node type

## GUI Integration

### Loading a Workflow

1. Launch GUI: `python examples/launch_gui.py`
2. File → Open Workflow
3. Select your workflow JSON file
4. Nodes appear on canvas

### Saving a Workflow

1. Create/modify workflow in GUI
2. File → Save Workflow
3. Choose location and filename
4. JSON file is created

### Modifying in GUI

- Drag nodes to reposition
- Click node to edit config
- Right-click for more options
- Connect ports by dragging

## Reference

### Existing Example Workflows

Look at these for reference:
- `examples/medical_segmentation_pipeline/training_workflow_simple.json`
- `examples/medical_segmentation_pipeline/testing_workflow.json`

### Schema Summary

```
Workflow
├── name (string)
├── nodes (array)
│   └── Node
│       ├── type (string)
│       ├── name (string)
│       ├── position (array[2])
│       ├── config (object)
│       ├── is_composite (boolean)
│       └── collapsed (boolean)
└── links (array)
    └── Link
        ├── source (string)
        ├── source_port (string)
        ├── target (string)
        └── target_port (string)
```

---

**Last Updated**: 2026-02-08
**Format Version**: 1.0
