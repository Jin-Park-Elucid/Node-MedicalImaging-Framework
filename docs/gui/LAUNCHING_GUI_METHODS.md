# GUI Launching Methods - Complete Guide

This document explains the different methods for launching the Medical Imaging Framework GUI and how they work internally.

## Table of Contents

1. [Quick Comparison](#quick-comparison)
2. [Method 1: Generic GUI Editor](#method-1-generic-gui-editor)
3. [Method 2: Example-Specific GUI](#method-2-example-specific-gui)
4. [Understanding `python -m`](#understanding-python--m)
5. [Execution Flow Details](#execution-flow-details)
6. [The Import Chain](#the-import-chain)
7. [Custom Node Registration](#custom-node-registration)
8. [When to Use Each Method](#when-to-use-each-method)
9. [Remote GUI Access (SSH X11 Forwarding)](#remote-gui-access-ssh-x11-forwarding)

---

## Quick Comparison

| Aspect | `python -m medical_imaging_framework.gui.editor` | `python examples/.../launch_gui.py` |
|--------|--------------------------------------------------|-------------------------------------|
| **What it launches** | Generic GUI editor | Example-specific GUI with custom nodes |
| **Nodes available** | 23 built-in nodes only | 23 built-in + custom nodes |
| **Pre-loaded workflows** | None | Example workflows ready to load |
| **Use case** | General-purpose workflow creation | Medical segmentation pipeline demo |
| **Working directory** | Can run from anywhere | Must run from project root |
| **Dependencies** | Core framework only | Requires example-specific files |
| **Startup message** | Minimal | Detailed info about nodes and workflows |

---

## Method 1: Generic GUI Editor

### Command

```bash
python -m medical_imaging_framework.gui.editor
```

### What It Does

Launches the core GUI workflow editor with only the built-in framework nodes.

### Features

- **23 Built-in Nodes:**
  - Data: ImagePathLoader, DataLoader, BatchExtractor, RandomFlip, RandomRotation, Normalize
  - Networks: UNet2D/3D, AttentionUNet2D, ResNet variants, Transformers, VisionTransformer
  - Training: Trainer, LossFunction, Optimizer
  - Inference: Predictor, BatchPredictor, MetricsCalculator
  - Visualization: ImageViewer, MetricsPlotter, Print

- **Clean Slate:**
  - Empty workflow on startup
  - No pre-loaded configurations
  - General-purpose use

- **Can Run From Anywhere:**
  - Works from any directory (as long as package is installed)
  - No dependency on example files

### Use Cases

- Creating workflows from scratch
- General medical imaging tasks
- Working with different datasets
- Learning the framework basics
- Building custom pipelines
- Production deployments

### Example Session

```bash
# From any directory
cd /home/user/my_project
python -m medical_imaging_framework.gui.editor
```

**Result:** GUI opens with 23 nodes available in the node palette.

---

## Method 2: Example-Specific GUI

### Command

```bash
python examples/medical_segmentation_pipeline/launch_gui.py
```

### What It Does

Launches the GUI with additional custom nodes specific to the medical segmentation example.

### Features

- **24 Nodes Total:**
  - All 23 built-in nodes
  - **PLUS:** `MedicalSegmentationLoader` (custom node)

- **Helpful Startup Message:**
  ```
  ================================================================================
  MEDICAL IMAGING FRAMEWORK - GUI EDITOR
  ================================================================================

  Custom nodes registered:
    ✓ MedicalSegmentationLoader

  Total nodes available: 24

  Workflow files available:
    • training_workflow.json
    • testing_workflow.json
    • training_workflow_simple.json

  ================================================================================
  ```

- **Pre-Made Workflows:**
  - Ready-to-load workflow files in the same directory
  - Configured for the example dataset
  - Demonstrate complete training and testing pipelines

- **Custom Data Loader:**
  - `MedicalSegmentationLoaderNode` knows the exact data structure
  - Automatically finds images and masks
  - Handles specific dataset format

### Use Cases

- Running the medical segmentation demo
- Loading and studying pre-built workflows
- Testing with example dataset
- Learning from working examples
- Tutorial walkthroughs
- Quick prototyping with example data

### Example Session

```bash
# Must run from project root
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
python examples/medical_segmentation_pipeline/launch_gui.py
```

**Result:** GUI opens with 24 nodes, startup message shows available workflows.

---

## Understanding `python -m`

### What Does the `-m` Flag Do?

The `-m` flag tells Python to run a **module** as a script.

### Syntax

```bash
python -m <package.module.submodule>
```

### How It Works

1. **Module Resolution:**
   - Python searches for the module in `sys.path`
   - Finds the module within the package structure
   - Example: `medical_imaging_framework.gui.editor` → `medical_imaging_framework/gui/editor.py`

2. **Import as Package:**
   - Imports the module with full package context
   - Enables relative imports like `from ..core import ...`
   - Maintains proper namespace

3. **Execute `if __name__ == "__main__":` Block:**
   - Sets `__name__` to `"__main__"`
   - Runs the main execution block

### Why Use `-m` Instead of Direct Script Execution?

**Method A: Running as module (Recommended)**
```bash
python -m medical_imaging_framework.gui.editor
```

**Method B: Running as script (Not recommended)**
```bash
python /path/to/medical_imaging_framework/gui/editor.py
```

**Key Differences:**

| Aspect | `-m` (Module) | Direct Script |
|--------|---------------|---------------|
| **Import path** | ✅ Correctly set up | ❌ May have import errors |
| **Relative imports** | ✅ `from ..core import ...` works | ❌ May fail |
| **Package context** | ✅ Runs within package namespace | ❌ Runs as standalone |
| **Working directory** | ✅ Can be anywhere | ⚠️ Must manage paths |
| **Professional** | ✅ Standard practice | ❌ Hacky approach |

---

## Execution Flow Details

### Method 1: Generic GUI (`python -m medical_imaging_framework.gui.editor`)

```
Step 1: Python locates module
├─> Searches sys.path for medical_imaging_framework
└─> Finds: medical_imaging_framework/gui/editor.py

Step 2: Module imports execute
├─> import sys, PyQt5 widgets
├─> from ..core import NodeRegistry, ComputationalGraph, GraphExecutor
└─> from .node_graphics import NodeGraphicsItem, ConnectionGraphicsItem

Step 3: Class definitions loaded
└─> class NodeGraphEditor(QMainWindow): ...

Step 4: if __name__ == "__main__": block executes
└─> Calls main()

Step 5: main() function runs
├─> app = QApplication(sys.argv)
├─> Set dark theme palette
├─> import medical_imaging_framework.nodes  # Registers 23 nodes
├─> editor = NodeGraphEditor()
├─> editor.show()
└─> sys.exit(app.exec_())  # Start Qt event loop
```

### Method 2: Example GUI (`python examples/.../launch_gui.py`)

```
Step 1: Python executes launch_gui.py as script
└─> Sets __name__ to "__main__"

Step 2: Path manipulation
├─> sys.path.insert(0, parent.parent.parent)  # Framework root
└─> sys.path.insert(0, parent)                # Example directory

Step 3: Import framework
├─> from medical_imaging_framework.core import NodeRegistry
└─> import medical_imaging_framework.nodes  # Registers 23 nodes

Step 4: Import custom node
└─> from custom_dataloader import MedicalSegmentationLoaderNode
    └─> @NodeRegistry.register() decorator executes
        └─> Registers MedicalSegmentationLoader (node #24)

Step 5: Print helpful information
├─> Print custom nodes registered
├─> Print total node count (24)
└─> List available workflow files

Step 6: Launch GUI
├─> from medical_imaging_framework.gui.editor import main as gui_main
└─> gui_main()  # Same as Method 1, but with 24 nodes
```

---

## The Import Chain

### Complete Import Hierarchy

```
python -m medical_imaging_framework.gui.editor
│
├─> Imports medical_imaging_framework.gui.editor
│   │
│   ├─> Imports PyQt5 components
│   │   ├─> QApplication, QMainWindow, QGraphicsView, QGraphicsScene
│   │   ├─> QDockWidget, QListWidget, QPushButton, QFileDialog
│   │   └─> Qt, QPointF, QPainter, QBrush, QColor
│   │
│   ├─> Imports medical_imaging_framework.core
│   │   ├─> NodeRegistry (decorator system)
│   │   ├─> ComputationalGraph (DAG manager)
│   │   ├─> GraphExecutor (execution engine)
│   │   ├─> BaseNode (node base class)
│   │   ├─> CompositeNode (hierarchical nodes)
│   │   └─> DataType, PortType (type system)
│   │
│   ├─> Imports medical_imaging_framework.gui.node_graphics
│   │   ├─> NodeGraphicsItem (visual node representation)
│   │   └─> ConnectionGraphicsItem (visual connection lines)
│   │
│   └─> Defines NodeGraphEditor class
│       ├─> __init__() - Initialize graph, scene, view
│       ├─> init_ui() - Set up GUI components
│       ├─> create_node_palette() - Build node library
│       ├─> create_menu_bar() - File, View, Execute menus
│       ├─> add_node() - Add node to graph and scene
│       ├─> execute_graph() - Run the workflow
│       └─> save_workflow() / load_workflow() - Serialization
│
└─> Runs main() function
    │
    ├─> Creates QApplication
    ├─> Sets dark theme palette
    │
    ├─> import medical_imaging_framework.nodes
    │   │
    │   ├─> nodes/__init__.py imports all submodules
    │   │
    │   ├─> nodes/data/__init__.py
    │   │   ├─> ImagePathLoader (registers)
    │   │   ├─> DataLoader (registers)
    │   │   ├─> BatchExtractor (registers)
    │   │   ├─> RandomFlip (registers)
    │   │   ├─> RandomRotation (registers)
    │   │   └─> Normalize (registers)
    │   │
    │   ├─> nodes/networks/ (various files)
    │   │   ├─> UNet2D, UNet3D (register)
    │   │   ├─> AttentionUNet2D (registers)
    │   │   ├─> ResNetEncoder2D, ResNetEncoder3D (register)
    │   │   ├─> ResNetDecoder2D (registers)
    │   │   ├─> TransformerEncoder (registers)
    │   │   └─> VisionTransformer2D (registers)
    │   │
    │   ├─> nodes/training/ (various files)
    │   │   ├─> Trainer (registers)
    │   │   ├─> LossFunction (registers)
    │   │   └─> Optimizer (registers)
    │   │
    │   ├─> nodes/inference/ (various files)
    │   │   ├─> Predictor (registers)
    │   │   ├─> BatchPredictor (registers)
    │   │   └─> MetricsCalculator (registers)
    │   │
    │   └─> nodes/visualization/ (various files)
    │       ├─> ImageViewer (registers)
    │       ├─> MetricsPlotter (registers)
    │       └─> Print (registers)
    │
    ├─> Creates NodeGraphEditor instance
    ├─> Shows the editor window
    └─> Starts Qt event loop (app.exec_())
```

---

## Custom Node Registration

### How Node Registration Works

Nodes use the `@NodeRegistry.register()` decorator:

```python
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register(
    category='data',           # Category for organization
    name='UNet2D',            # Display name
    description='2D U-Net for segmentation',  # Tooltip
    icon='network'            # Icon identifier
)
class UNet2DNode(PyTorchModuleNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def execute(self) -> bool:
        # Node logic here
        pass
```

### Registration Process

1. **Decorator Executes During Import:**
   - When Python imports the module, decorators run
   - `@NodeRegistry.register()` adds node to registry

2. **Registry Storage:**
   ```python
   NodeRegistry._registry = {
       'UNet2D': {
           'class': UNet2DNode,
           'category': 'data',
           'description': '2D U-Net for segmentation',
           'icon': 'network',
           'name': 'UNet2D'
       },
       # ... more nodes
   }
   ```

3. **GUI Discovers Nodes:**
   - GUI queries `NodeRegistry.get_categories()`
   - For each category, calls `NodeRegistry.get_nodes_by_category(category)`
   - Displays nodes in palette organized by category

### Custom Node Example: MedicalSegmentationLoader

In `examples/medical_segmentation_pipeline/custom_dataloader.py`:

```python
@NodeRegistry.register(
    'data',
    'MedicalSegmentationLoader',
    description='Load medical segmentation dataset with images and masks'
)
class MedicalSegmentationLoaderNode(BaseNode):
    """
    Custom dataloader for medical segmentation example.

    Knows the exact structure:
      data/train/images/*.png
      data/train/masks/*.png
      data/test/images/*.png
      data/test/masks/*.png
    """

    def _setup_ports(self):
        self.add_output('train_loader', DataType.DATALOADER)
        self.add_output('test_loader', DataType.DATALOADER)

    def execute(self) -> bool:
        # Load dataset from specific structure
        train_dataset = MedicalSegmentationDataset(
            data_dir='./data',
            split='train'
        )
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.set_output_value('train_loader', train_loader)
        return True
```

**Why It Matters:**
- Built-in `DataLoader` node is generic
- `MedicalSegmentationLoader` knows the example's specific data structure
- Makes the example workflows work without configuration

---

## When to Use Each Method

### Use Generic GUI: `python -m medical_imaging_framework.gui.editor`

**Recommended for:**

- ✅ **Creating workflows from scratch**
  - You know your data structure
  - Building custom pipelines
  - Production deployments

- ✅ **General medical imaging tasks**
  - Not using the example dataset
  - Different data formats or structures
  - Custom requirements

- ✅ **Working with different datasets**
  - Your own medical images
  - Different file organizations
  - Custom preprocessing needs

- ✅ **Learning the framework basics**
  - Understanding core concepts
  - Exploring available nodes
  - Building foundational knowledge

- ✅ **No example dependencies**
  - Don't need example-specific files
  - Cleaner environment
  - Minimal dependencies

**Example scenarios:**
```bash
# Scenario 1: New project with custom data
cd /home/user/my_medical_project
python -m medical_imaging_framework.gui.editor
# Create workflow, add nodes, configure for your data

# Scenario 2: Production deployment
cd /opt/medical_imaging_app
python -m medical_imaging_framework.gui.editor workflow.json
# Load production workflow and execute

# Scenario 3: Learning the framework
cd ~/learning
python -m medical_imaging_framework.gui.editor
# Explore nodes, experiment, learn
```

---

### Use Example GUI: `python examples/.../launch_gui.py`

**Recommended for:**

- ✅ **Running the medical segmentation demo**
  - Trying out the framework
  - Seeing it in action
  - Quick demonstration

- ✅ **Loading pre-built workflows**
  - training_workflow.json
  - testing_workflow.json
  - Pre-configured pipelines

- ✅ **Testing with example dataset**
  - Using the provided sample data
  - Verifying installation
  - Reproducing tutorial results

- ✅ **Learning from working examples**
  - Understanding workflow design
  - Seeing best practices
  - Template for your own workflows

- ✅ **Using MedicalSegmentationLoader**
  - Need the custom data loader
  - Working with example data structure
  - Following tutorials

- ✅ **Quick start tutorials**
  - First time users
  - Guided learning
  - Step-by-step examples

**Example scenarios:**
```bash
# Scenario 1: First time user
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
python examples/medical_segmentation_pipeline/launch_gui.py
# See available workflows, load training_workflow.json, execute

# Scenario 2: Testing installation
python examples/medical_segmentation_pipeline/launch_gui.py
# Verify all nodes load, check example data, test execution

# Scenario 3: Learning workflow design
python examples/medical_segmentation_pipeline/launch_gui.py
# Load testing_workflow.json, examine node connections, understand pipeline
```

---

## Remote GUI Access (SSH X11 Forwarding)

If you want to run the GUI on a remote server but display it on your local machine:

### Prerequisites

**On Local Machine (Ubuntu):**
- X11 server running (usually automatic)
- SSH client installed

**On Remote Server:**
- X11 forwarding enabled in SSH config
- Framework installed

### Method: X11 Forwarding over SSH

#### Step 1: Connect with X11 Forwarding

```bash
# On your local Ubuntu laptop
ssh -X username@remote-server
# or for trusted connections (faster):
ssh -Y username@remote-server
```

#### Step 2: Verify X11 Setup

```bash
# On remote server (after SSH connection)
echo $DISPLAY
# Should show: localhost:10.0 or similar

# Test with simple app
xeyes
# Should display on your local screen
```

#### Step 3: Launch GUI

```bash
# Navigate to framework
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging

# Launch GUI (either method)
python -m medical_imaging_framework.gui.editor
# OR
python examples/medical_segmentation_pipeline/launch_gui.py
```

**Result:** GUI window appears on your local Ubuntu laptop!

### Troubleshooting X11 Forwarding

**If X11 doesn't work:**

1. **Check X11 forwarding is enabled on server:**
   ```bash
   # On server
   grep X11Forwarding /etc/ssh/sshd_config
   # Should show: X11Forwarding yes
   ```

2. **Enable X11 forwarding if needed:**
   ```bash
   # On server (requires sudo)
   sudo nano /etc/ssh/sshd_config
   # Add or change to: X11Forwarding yes
   sudo systemctl restart sshd
   ```

3. **Check DISPLAY variable:**
   ```bash
   # After SSH connection
   echo $DISPLAY
   # If empty, X11 forwarding isn't working
   ```

4. **Test with simple app:**
   ```bash
   xclock  # Should show a clock on your local screen
   ```

### Performance Tips

For better X11 performance over slow connections:

```bash
# Use compression
ssh -X -C username@remote-server

# Or specific compression level
ssh -X -C -o CompressionLevel=9 username@remote-server
```

### SSH Config for Convenience

Create `~/.ssh/config` on your local machine:

```
Host medical-server
    HostName remote-server-address
    User username
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes
```

Then simply:
```bash
ssh medical-server
# X11 forwarding automatically enabled!
```

---

## Alternative Launch Methods

### From Python Code

```python
# Method 1: Generic GUI
from medical_imaging_framework.gui import main
main()

# Method 2: Import and call directly
from medical_imaging_framework.gui.editor import NodeGraphEditor
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
editor = NodeGraphEditor()
editor.show()
sys.exit(app.exec_())
```

### From Jupyter Notebook

**Note:** PyQt5 GUI doesn't work well in Jupyter. Use the Python API instead:

```python
# In Jupyter
from medical_imaging_framework import (
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor
)
import medical_imaging_framework.nodes

# Create workflow programmatically
graph = ComputationalGraph("My Workflow")
# Add nodes, connect, execute...
```

### With Workflow File Argument

```bash
# Load workflow on startup (if implemented)
python -m medical_imaging_framework.gui.editor my_workflow.json
```

---

## Summary

### Quick Reference

**Generic GUI (General Purpose):**
```bash
python -m medical_imaging_framework.gui.editor
```
- 23 nodes
- Empty start
- Any dataset
- Production use

**Example GUI (Demo/Learning):**
```bash
python examples/medical_segmentation_pipeline/launch_gui.py
```
- 24 nodes (23 + custom)
- Pre-made workflows
- Example dataset
- Tutorial/demo use

**Remote Display:**
```bash
ssh -X user@server
python -m medical_imaging_framework.gui.editor
```
- GUI displays on local machine
- Runs on remote server
- Requires X11 forwarding

### Key Takeaways

1. **`python -m` is the proper way** to run Python modules
2. Both methods launch **the same GUI editor**
3. The difference is **node availability** and **context**
4. Example launcher **registers custom nodes** before starting GUI
5. Generic launcher is **cleaner** for production use
6. Example launcher is **better** for learning and demos
7. X11 forwarding allows **remote GUI** on local display

---

## Related Documentation

- [Main README](../README.md) - Framework overview
- [Getting Started](../getting-started/GETTING_STARTED.md) - Quick start guide
- [GUI Complete Guide](./VISUAL_GUI_COMPLETE.md) - All GUI features
- [GUI Quick Reference](./VISUAL_GUI_QUICK_REFERENCE.md) - Keyboard shortcuts
- [Example README](../examples/medical-segmentation/README.md) - Medical segmentation example

---

**Last Updated:** February 7, 2026
