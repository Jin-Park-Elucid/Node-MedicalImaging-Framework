# Workflow Format Corrections

## Issue Fixed

The initial workflow JSON files used an incorrect format that caused this error:
```
Failed to load workflow: 'name'
KeyError: 'name'
```

## What Was Wrong

### Original Format (Incorrect)
```json
{
  "name": "Workflow",
  "description": "...",
  "version": "1.0.0",
  "created": "2026-02-08",
  "nodes": [
    {
      "id": "node_1",           // ❌ Wrong: should be "name"
      "type": "AVTE2DLoader",
      "category": "data",        // ❌ Wrong: not needed
      "position": {              // ❌ Wrong: should be array
        "x": 100,
        "y": 150
      },
      "config": { ... },
      "notes": "..."            // ❌ Wrong: not supported
    }
  ],
  "connections": [...]          // ❌ Wrong: should be "links"
}
```

### Corrected Format
```json
{
  "name": "Workflow",          // ✅ Only "name" at top level
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "node_1",          // ✅ "name" not "id"
      "position": [100, 150],    // ✅ Array format
      "config": { ... },
      "is_composite": false,     // ✅ Required field
      "collapsed": false         // ✅ Required field
    }
  ],
  "links": []                    // ✅ "links" not "connections"
}
```

## Key Changes

### 1. Node Identifier
- **Before**: `"id": "node_1"`
- **After**: `"name": "node_1"`

### 2. Position Format
- **Before**: `"position": {"x": 100, "y": 150}`
- **After**: `"position": [100, 150]`

### 3. Top-Level Fields
- **Before**: `name`, `description`, `version`, `created`, `metadata`, etc.
- **After**: Only `name`, `nodes`, `links`

### 4. Required Node Fields
Added:
- `"is_composite": false`
- `"collapsed": false`

Removed:
- `"id"`
- `"category"`
- `"notes"`

### 5. Connection Field Name
- **Before**: `"connections"`
- **After**: `"links"`

### 6. Link Structure
- **Before**: `"source_node"`, `"source_port"`, `"target_node"`, `"target_port"`
- **After**: `"source"`, `"source_port"`, `"target"`, `"target_port"`

## Current State

Both workflow files are now in the correct format:

### simple_training_workflow.json
```json
{
  "name": "AVTE 2D Simple Test",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "avte_dataloader",
      "position": [100, 200],
      "config": {
        "data_dir": "/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data",
        "batch_size": "8",
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

### training_workflow.json
Same structure, different batch_size (16 vs 8).

## How to Use

### 1. Load in GUI
```bash
python examples/launch_gui.py
```

In GUI:
- File → Open Workflow
- Select `examples/AVTE/config/training_workflow.json` or `simple_training_workflow.json`
- Workflow loads successfully ✓

### 2. Modify Configuration
Click on the node to edit:
- Update `data_dir` to your path
- Adjust `batch_size` as needed
- Change other parameters

### 3. Save Changes
- File → Save Workflow
- Saves in correct format

## Adding More Nodes

When you add more nodes in the GUI or manually:

```json
{
  "name": "My Workflow",
  "nodes": [
    {
      "type": "AVTE2DLoader",
      "name": "data_loader",
      "position": [100, 200],
      "config": { ... },
      "is_composite": false,
      "collapsed": false
    },
    {
      "type": "UNet2DNode",
      "name": "model",
      "position": [400, 200],
      "config": {
        "in_channels": 5,
        "out_channels": 2
      },
      "is_composite": false,
      "collapsed": false
    }
  ],
  "links": [
    {
      "source": "data_loader",
      "source_port": "train_loader",
      "target": "model",
      "target_port": "data"
    }
  ]
}
```

## Documentation

See `WORKFLOW_FORMAT.md` for complete format specification including:
- Field descriptions
- Common patterns
- Troubleshooting
- Examples

## Migration Notes

If you created custom workflows with the old format:

1. Change `"id"` to `"name"`
2. Change `"position": {"x": X, "y": Y}` to `"position": [X, Y]`
3. Remove extra top-level fields (keep only `name`, `nodes`, `links`)
4. Remove `"category"` from nodes
5. Add `"is_composite": false` and `"collapsed": false` to each node
6. Change `"connections"` to `"links"`
7. In links, change `"source_node"` to `"source"` and `"target_node"` to `"target"`

---

**Fixed**: 2026-02-08
**Status**: ✅ Working
