# GUI Save Bug Analysis

## The Problem

When you create connections in the GUI and click "Save Workflow", the connections are not saved to the JSON file. The saved file has `"links": []` even though you see connections on the canvas.

## Root Cause

### Location
File: `medical_imaging_framework/gui/node_graphics.py`
Method: `PortGraphicsItem._create_connection()` (lines 428-477)

### The Bug

When you drag and connect two ports in the GUI, the `_create_connection()` method:

1. ✅ Creates a visual connection (line 449)
2. ✅ Creates a Link object (lines 459-462)
3. ❌ **Does NOT call `graph.connect()`** to add the link to the graph
4. ❌ Stores link in `editor.workflow_links` instead (lines 465-468)

```python
# Current buggy code (lines 459-468)
link = Link(
    source=source_node.outputs[source_port.port_name],
    target=target_node.inputs[target_port.port_name]
)

# Store link reference (if workflow/editor needs it)
if hasattr(self.scene(), 'editor'):
    if not hasattr(self.scene().editor, 'workflow_links'):
        self.scene().editor.workflow_links = []
    self.scene().editor.workflow_links.append(link)  # ← Stored in wrong place!
```

### Why This Breaks Saving

When you click "Save Workflow", it calls:
```python
# editor.py line 370
self.graph.save_to_file(filename)
```

Which iterates over:
```python
# graph.py lines 267-275
'links': [
    {
        'source_node': link.source.node.name,
        'source_port': link.source.name,
        'target_node': link.target.node.name,
        'target_port': link.target.name
    }
    for link in self.links  # ← This list is empty!
]
```

The links are stored in `editor.workflow_links`, not `graph.links`, so they're never saved.

## The Fix

### Solution: Call `graph.connect()` Instead

Replace the manual Link creation with a call to `graph.connect()`, which properly adds the link to `graph.links`.

### Fixed Code

```python
def _create_connection(self, source_port, target_port):
    """Create a connection between source and target ports."""
    # Check if connection already exists
    for conn in source_port.connections:
        if (conn.source_port == source_port and conn.target_port == target_port) or \
           (conn.source_port == target_port and conn.target_port == source_port):
            print(f"Connection already exists between {source_port.port_name} and {target_port.port_name}")
            return

    # Check if data types are compatible
    if source_port.port.data_type != target_port.port.data_type:
        print(f"Cannot connect: incompatible types {source_port.port.data_type} -> {target_port.port.data_type}")
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(
            None,
            "Incompatible Port Types",
            f"Cannot connect {source_port.port.data_type.value} to {target_port.port.data_type.value}"
        )
        return

    # Create visual connection
    connection = ConnectionGraphicsItem(source_port, target_port)
    self.scene().addItem(connection)

    # Get node references
    source_node = source_port.parentItem().node
    target_node = target_port.parentItem().node

    try:
        # Get the graph from the editor
        editor = None
        if hasattr(self.scene(), 'editor'):
            editor = self.scene().editor
        elif hasattr(self.scene(), 'parent') and hasattr(self.scene().parent(), 'editor'):
            editor = self.scene().parent().editor

        if editor and hasattr(editor, 'graph'):
            # Use graph.connect() to properly add link to graph.links
            link = editor.graph.connect(
                source_node.name,
                source_port.port_name,
                target_node.name,
                target_port.port_name
            )

            # Store visual connection reference in the link
            if hasattr(link, 'graphics_item'):
                link.graphics_item = connection

            print(f"Connected: {source_node.name}.{source_port.port_name} -> {target_node.name}.{target_port.port_name}")
        else:
            print("Warning: Could not find graph to register connection")
            # Fallback to old behavior
            from ..core import Link
            link = Link(
                source=source_node.outputs[source_port.port_name],
                target=target_node.inputs[target_port.port_name]
            )

    except Exception as e:
        print(f"Error creating connection: {e}")
        import traceback
        traceback.print_exc()
        # Remove visual connection if logical connection failed
        self.scene().removeItem(connection)
        return

    # Track connections on ports
    source_port.connections.append(connection)
    target_port.connections.append(connection)
    connection.source_port = source_port
    connection.target_port = target_port
```

### Key Changes

1. **Get editor reference** to access `editor.graph`
2. **Call `editor.graph.connect()`** instead of creating Link manually
3. **Pass node names and port names** (as strings) to `connect()`
4. **Remove `editor.workflow_links`** storage (no longer needed)

## Implementation

### File to Modify
`medical_imaging_framework/gui/node_graphics.py`

### Lines to Change
Lines 428-477 (the entire `_create_connection` method)

### Testing the Fix

After applying the fix:

1. **Open GUI**: `python examples/launch_gui.py`
2. **Create workflow**: Add nodes and connect them
3. **Save**: File → Save Workflow
4. **Check JSON**: Open the saved file and verify `"links": [...]` is not empty
5. **Reload**: Open the saved workflow - connections should appear

## Workaround (Until Fixed)

Since the bug prevents saving connections, you have two options:

### Option 1: Use Pre-made Workflow
Load the corrected workflow I created:
```bash
# In GUI
File → Open Workflow → examples/AVTE/config/training_workflow.json
```

This file has the connections already in the JSON.

### Option 2: Manual JSON Editing
1. Create nodes in GUI
2. Save workflow (saves nodes but not connections)
3. Manually edit the JSON file to add connections

Example connections to add:
```json
"links": [
  {
    "source_node": "avte_dataloader",
    "source_port": "train_loader",
    "target_node": "trainer",
    "target_port": "dataloader"
  },
  {
    "source_node": "avte_dataloader",
    "source_port": "val_loader",
    "target_node": "trainer",
    "target_port": "val_dataloader"
  },
  {
    "source_node": "unet_model",
    "source_port": "model",
    "target_node": "optimizer",
    "target_port": "model"
  },
  {
    "source_node": "unet_model",
    "source_port": "model",
    "target_node": "trainer",
    "target_port": "model"
  },
  {
    "source_node": "loss_function",
    "source_port": "loss_fn",
    "target_node": "trainer",
    "target_port": "loss_fn"
  },
  {
    "source_node": "optimizer",
    "source_port": "optimizer",
    "target_node": "trainer",
    "target_port": "optimizer"
  }
]
```

## Summary

- ❌ **Bug**: GUI doesn't save connections created by dragging between ports
- 🔍 **Cause**: `_create_connection()` doesn't call `graph.connect()`
- ✅ **Fix**: Call `editor.graph.connect()` to properly register links
- 🔧 **Workaround**: Use pre-made workflow or manually edit JSON

---

**File**: `medical_imaging_framework/gui/node_graphics.py`
**Method**: `PortGraphicsItem._create_connection()`
**Lines**: 428-477
**Status**: Bug confirmed, fix provided above
