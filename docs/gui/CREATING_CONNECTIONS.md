# Creating Connections in the GUI

## Overview

This guide explains how to create connections between nodes in the visual GUI.

## What Changed

✅ **FIXED**: Port circles now respond to mouse clicks and drags
✅ **IMPROVED**: Larger port circles (radius 8px instead of 6px) for easier clicking
✅ **ENHANCED**: Bidirectional dragging - drag from either output or input ports

---

## How to Create Connections

### Method 1: Drag from Output Port (Recommended)

This is the standard way to create connections:

1. **Locate the Output Port** (orange circle on the right side of a node)
2. **Click and Hold** on the output port circle
3. **Drag** toward the target input port (blue circle on the left side of another node)
4. **Release** when hovering over the target input port

**Visual Feedback**:
- Yellow dashed line appears while dragging
- Target port highlights when hovering
- Connection appears as a curved line when released

### Method 2: Drag from Input Port

You can also drag from input ports for convenience:

1. **Click and Hold** on an input port (blue circle on the left)
2. **Drag** toward an output port (orange circle on the right)
3. **Release** on the target output port

The connection will automatically be created in the correct direction (output → input).

---

## Visual Guide

```
┌─────────────────┐                    ┌─────────────────┐
│  data_loader    │                    │     trainer     │
├─────────────────┤                    ├─────────────────┤
│                 │                    │  ● dataloader   │◄──┐
│                 │                    │  ● model        │   │
│  train_loader ●─┼────────────────────┼──┘ loss_fn      │   │
│                 │   (Curved line)    │  ● optimizer    │   │
└─────────────────┘                    └─────────────────┘   │
                                                              │
  1. Click here ──────────────────────────────────────────────┘
     (orange output port)

  2. Drag to here
     (blue input port)

  3. Release
```

---

## Port Colors

- **Blue circles** (●): Input ports (left side of nodes)
- **Orange circles** (●): Output ports (right side of nodes)
- **Yellow** highlight: Port while hovering during drag
- **Yellow dashed line**: Temporary connection while dragging

---

## Connection Rules

### Valid Connections

✅ Output port → Input port
✅ Same data type (TENSOR to TENSOR, MODEL to MODEL, etc.)
✅ Across different nodes
✅ One output can connect to multiple inputs

### Invalid Connections

❌ Output → Output
❌ Input → Input
❌ Incompatible data types (TENSOR to STRING)
❌ Port to itself
❌ Duplicate connections between same ports

---

## Tips for Success

### If Dragging Moves the Whole Node

This was the original problem! Here's how to avoid it:

1. **Click Precisely on the Port Circle**
   - The port circles are now larger (16px diameter)
   - Aim for the center of the colored circle
   - Don't click on the node body

2. **Click and Hold**
   - Don't just click quickly
   - Click, hold down the mouse button, then drag

3. **Look for Visual Feedback**
   - Port should highlight yellow when you click it
   - Yellow dashed line should appear when dragging
   - If the whole node moves, you clicked outside the port

### If Connection Doesn't Create

**Check these:**

1. **Port Types Match**
   - Hover over ports to see data types in tooltip
   - Example: "train_loader (DATALOADER)" → "dataloader (DATALOADER)"

2. **Correct Direction**
   - Connection must be output (orange) → input (blue)
   - Even if you drag from input, it creates output → input

3. **Target Port is Valid**
   - Must be on a different node
   - Must be opposite type (output → input or input → output)

4. **No Duplicate**
   - Connection doesn't already exist between these ports

### Making It Easier

**Zoom In**: Use mouse wheel to zoom in for more precise clicking

**Pan the Canvas**:
- Middle mouse drag to pan
- Or use scroll bars

**Port Tooltips**:
- Hover over a port to see its name and data type
- Example: "train_loader (DATALOADER)"

---

## Example: Creating Training Workflow Connections

Let's create all connections for the training workflow:

### Connection 1: data_loader → trainer

```
Source: data_loader.train_loader (output)
Target: trainer.dataloader (input)

Steps:
1. Click on data_loader's "train_loader" port (orange, right side)
2. Drag to trainer's "dataloader" port (blue, left side)
3. Release
```

### Connection 2: unet_model → trainer

```
Source: unet_model.output (output)
Target: trainer.model (input)

Steps:
1. Click on unet_model's "output" port
2. Drag to trainer's "model" port
3. Release
```

### Connection 3: loss_function → trainer

```
Source: loss_function.loss_fn (output)
Target: trainer.loss_fn (input)

Steps:
1. Click on loss_function's "loss_fn" port
2. Drag to trainer's "loss_fn" port
3. Release
```

### Connection 4: unet_model → optimizer

```
Source: unet_model.output (output)
Target: optimizer.model (input)

Steps:
1. Click on unet_model's "output" port (can reuse same output)
2. Drag to optimizer's "model" port
3. Release
```

### Connection 5: optimizer → trainer

```
Source: optimizer.optimizer (output)
Target: trainer.optimizer (input)

Steps:
1. Click on optimizer's "optimizer" port
2. Drag to trainer's "optimizer" port
3. Release
```

---

## Deleting Connections

### Method 1: Click and Delete

1. Click on the connection line (it should highlight)
2. Press **Delete** or **Backspace** key

### Method 2: Right-Click Menu

1. Right-click on the connection line
2. Select "Delete Connection" from menu

---

## Troubleshooting

### Problem: "Whole node moves when I try to drag from port"

**Solution**:
- The port circles are now larger (16px diameter)
- Click more precisely on the center of the colored circle
- Make sure you see the yellow highlight when clicking
- Try zooming in for better precision

### Problem: "Connection appears then disappears"

**Cause**: Invalid connection attempt

**Solutions**:
- Check port types match (hover for tooltip)
- Ensure connecting output to input (or input to output)
- Verify ports are on different nodes
- Check for duplicate connections

### Problem: "Port is too small to click"

**Solution**:
- Ports are now 16px diameter (doubled from original 12px)
- Zoom in using mouse wheel
- Hover to see tooltip confirming you're over the port

### Problem: "Error message appears"

**"Incompatible Port Types"**:
- Data types don't match
- Example: Can't connect TENSOR to STRING
- Check tooltips to verify types

**"Connection already exists"**:
- This connection is already created
- Check existing connections (curved lines)

---

## Keyboard Shortcuts

While creating connections:

- **ESC**: Cancel connection drag
- **Delete/Backspace**: Delete selected connection
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse Drag**: Pan canvas

---

## Technical Details

### What Happens When You Create a Connection

1. **Visual Connection**: A `ConnectionGraphicsItem` is created (curved line)
2. **Logical Connection**: A `Link` object connects the ports in the workflow
3. **Data Flow**: When workflow executes, data flows from output to input

### Port Graphics

- **Type**: `PortGraphicsItem` (QGraphicsEllipseItem)
- **Size**: 16px diameter circle
- **Z-Value**: 10 (drawn on top of nodes)
- **Interactive**: Accepts mouse events, hover events

### Connection Graphics

- **Type**: `ConnectionGraphicsItem` (custom painted curve)
- **Style**: Curved bezier path with arrowhead
- **Z-Value**: -1 (drawn behind nodes)
- **Updates**: Automatically updates when nodes move

---

## Best Practices

### Workflow Organization

1. **Arrange nodes left to right** (data flow direction)
2. **Space nodes adequately** for clear connections
3. **Avoid crossing connections** when possible
4. **Group related nodes** vertically

### Connection Strategy

1. **Create data flow connections first** (loader → trainer)
2. **Then model connections** (model → trainer, optimizer)
3. **Finally support connections** (loss, metrics)

### Visual Clarity

1. **Use zoom** to work on dense areas
2. **Reorganize nodes** if connections become tangled
3. **Delete and recreate** connections if needed
4. **Save frequently** after creating connections

---

## Next Steps

- [Visual GUI Guide](VISUAL_GUI_COMPLETE.md) - Complete GUI documentation
- [Network Architectures](../segmentation/NETWORK_ARCHITECTURES.md) - Understanding network nodes
- [Example Workflows](../../examples/medical_segmentation_pipeline/workflows/) - Pre-configured workflows

---

## Summary

**To create a connection:**
1. ✅ Click and hold on a port circle (orange or blue)
2. ✅ Drag to target port (opposite type)
3. ✅ Release when over target
4. ✅ Connection created!

**Common issues fixed:**
- ✅ Ports now larger and easier to click
- ✅ Mouse events properly captured
- ✅ Visual feedback during drag
- ✅ Bidirectional dragging support

Happy connecting! 🔗
