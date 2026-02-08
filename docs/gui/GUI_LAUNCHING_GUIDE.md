# GUI Launching Guide

**Last Updated:** February 7, 2026
**Status:** ✅ Verified Working

This guide explains how to properly launch the GUI with different node configurations.

---

## 🎯 Quick Reference

### For Medical Segmentation Example

```bash
# ✅ CORRECT - Use this to see all nodes including MedicalSegmentationLoader
python examples/medical_segmentation_pipeline/launch_gui.py
```

### For Generic Framework Usage

```bash
# Use this if you only need the standard framework nodes
python -m medical_imaging_framework.gui.editor
```

---

## 📋 Understanding the Two Launch Methods

### Method 1: Generic GUI (26 nodes)

**Command:**
```bash
python -m medical_imaging_framework.gui.editor
```

**Available Nodes:**
- ✅ All framework nodes (25 nodes)
- ❌ Custom example-specific nodes NOT loaded

**Use When:**
- Building general workflows
- Not using medical segmentation example
- Creating custom pipelines from scratch

**Registered Nodes:**
- Networks (14): UNet2D, UNet3D, ResNet, Transformers, etc.
- Training (4): Trainer, Optimizer, Loss, CheckpointLoader
- Inference (3): Predictor, BatchPredictor, MetricsCalculator
- Visualization (4): ImageViewer, MetricsPlotter, Print, SegmentationOverlay

---

### Method 2: Medical Segmentation GUI (26 nodes)

**Command:**
```bash
python examples/medical_segmentation_pipeline/launch_gui.py
```

**Available Nodes:**
- ✅ All framework nodes (25 nodes)
- ✅ Custom MedicalSegmentationLoader node
- **Total: 26 nodes**

**Use When:**
- Loading medical segmentation workflows
- Using `training_workflow.json` or `testing_workflow.json`
- Working with medical segmentation example

**Additional Nodes:**
- **MedicalSegmentationLoader** - Custom dataloader for medical segmentation dataset
  - Loads images and masks from train/test directories
  - Creates PyTorch DataLoaders
  - Supports batch configuration

---

## 🚨 Important: Loading Workflows

### Problem: Nodes Don't Appear

**Symptom:**
- You load `training_workflow.json` in the GUI
- No nodes are displayed
- Workflow appears empty

**Cause:**
- You're using the generic GUI launcher
- The workflow requires `MedicalSegmentationLoader` node
- This node is only available with the special launcher

**Solution:**
```bash
# Use the medical segmentation launcher
python examples/medical_segmentation_pipeline/launch_gui.py
```

---

## 📖 How It Works

### Generic Launcher

The generic launcher (`-m medical_imaging_framework.gui.editor`) imports:
```python
import medical_imaging_framework.nodes
```

This loads only the nodes in the main framework:
- `nodes/networks/`
- `nodes/training/`
- `nodes/inference/`
- `nodes/visualization/`

### Medical Segmentation Launcher

The special launcher (`launch_gui.py`) does additional imports:
```python
import medical_imaging_framework.nodes  # Load framework nodes
from custom_dataloader import MedicalSegmentationLoaderNode  # Load custom node
```

This registers the custom node before starting the GUI.

---

## 🔧 Creating Custom Launchers

If you create your own custom nodes, you'll need a custom launcher:

### Example: Custom Launcher Template

```python
"""Launch GUI with custom nodes registered."""
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import framework
from medical_imaging_framework.core import NodeRegistry
import medical_imaging_framework.nodes

# Import your custom nodes HERE
from my_custom_node import MyCustomNode
from another_custom_node import AnotherCustomNode

# Launch GUI
from medical_imaging_framework.gui.editor import main as gui_main

if __name__ == "__main__":
    print("="*80)
    print("CUSTOM GUI LAUNCHER")
    print("="*80)
    print()
    print("Custom nodes registered:")
    print("  ✓ MyCustomNode")
    print("  ✓ AnotherCustomNode")
    print()
    print(f"Total nodes available: {len(NodeRegistry.get_all_nodes())}")
    print()
    print("="*80)
    print()

    gui_main()
```

Save as `my_launcher.py` and run:
```bash
python my_launcher.py
```

---

## 🎨 SSH X11 Forwarding (Remote GUI)

### From Local Machine

```bash
# Connect with X11 forwarding
ssh -X your-server

# On the server
cd Node-MedicalImaging-Framework
source activate.sh

# Launch appropriate GUI
python examples/medical_segmentation_pipeline/launch_gui.py
```

The GUI will appear on your local machine! ✅

### Troubleshooting X11

If GUI doesn't appear:

1. **Check X11 forwarding is enabled:**
   ```bash
   echo $DISPLAY
   # Should show something like: localhost:10.0
   ```

2. **Test X11 with simple app:**
   ```bash
   xclock
   # A clock should appear on your local machine
   ```

3. **Update SSH config** (`~/.ssh/config` on local machine):
   ```
   Host your-server
       ForwardX11 yes
       ForwardX11Trusted yes
   ```

4. **Server X11 dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install xauth
   ```

See `docs/gui/SSH_X11_FORWARDING_GUIDE.md` for complete guide.

---

## 📝 Workflow Files

### Medical Segmentation Workflows

Located in: `examples/medical_segmentation_pipeline/`

**training_workflow.json** - Training pipeline
- Nodes: MedicalSegmentationLoader, UNet2D, Trainer, Optimizer, Loss
- Purpose: Train segmentation model
- Run with: Medical Segmentation GUI

**testing_workflow.json** - Testing pipeline
- Nodes: MedicalSegmentationLoader, UNet2D, CheckpointLoader, Predictor, Metrics
- Purpose: Test trained model and visualize results
- Run with: Medical Segmentation GUI

### Loading Workflows

1. Launch appropriate GUI:
   ```bash
   python examples/medical_segmentation_pipeline/launch_gui.py
   ```

2. In GUI:
   - Click "📂 Load Workflow" button
   - Navigate to workflow file
   - Select `training_workflow.json` or `testing_workflow.json`
   - Click "Open"

3. Verify nodes appear:
   - You should see all nodes positioned on the canvas
   - Connections between nodes should be visible
   - Node count should match workflow

---

## ✅ Verification

### Check Available Nodes

**Generic GUI:**
```bash
python -c "from medical_imaging_framework.core import NodeRegistry; import medical_imaging_framework.nodes; print(f'{len(NodeRegistry.get_all_nodes())} nodes')"
# Expected: 25 nodes
```

**With Custom Nodes:**
```bash
cd examples/medical_segmentation_pipeline
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent.parent)); sys.path.insert(0, str(Path.cwd())); from medical_imaging_framework.core import NodeRegistry; import medical_imaging_framework.nodes; from custom_dataloader import MedicalSegmentationLoaderNode; print(f'{len(NodeRegistry.get_all_nodes())} nodes')"
# Expected: 26 nodes
```

---

## 🐛 Troubleshooting

### Issue: Nodes Still Don't Appear

**Check 1: Using correct launcher?**
```bash
# Should see "Custom nodes registered:"
python examples/medical_segmentation_pipeline/launch_gui.py
```

**Check 2: Custom node imported?**
```bash
# From examples/medical_segmentation_pipeline directory
python -c "from custom_dataloader import MedicalSegmentationLoaderNode; print('✓ Custom node imports')"
```

**Check 3: Node is registered?**
```bash
cd examples/medical_segmentation_pipeline
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent.parent)); sys.path.insert(0, str(Path.cwd())); from medical_imaging_framework.core import NodeRegistry; import medical_imaging_framework.nodes; from custom_dataloader import MedicalSegmentationLoaderNode; print('MedicalSegmentationLoader' in NodeRegistry.get_all_nodes())"
# Expected: True
```

### Issue: GUI Doesn't Launch

**Check Python and dependencies:**
```bash
python --version
python -c "import PyQt5; print('PyQt5 OK')"
```

**Check X11 (if remote):**
```bash
echo $DISPLAY
xclock  # Test app should appear
```

---

## 📚 Related Documentation

- **Main GUI Guide:** `docs/gui/VISUAL_GUI_COMPLETE.md`
- **SSH X11 Guide:** `docs/gui/SSH_X11_FORWARDING_GUIDE.md`
- **Launch Methods:** `docs/gui/LAUNCHING_GUI_METHODS.md`
- **Medical Segmentation Example:** `docs/examples/medical-segmentation/README.md`

---

## 💡 Best Practices

### For Development

1. **Use custom launchers** for project-specific workflows
2. **Keep custom nodes** in example directories
3. **Document required nodes** in workflow README
4. **Test both launchers** to ensure compatibility

### For Deployment

1. **Include launcher scripts** in deployment
2. **Document which launcher** to use for each workflow
3. **Test GUI launches** after deployment
4. **Verify X11** works if deploying remotely

---

## 🚀 Summary

| Launcher | Nodes | Use Case |
|----------|-------|----------|
| `python -m medical_imaging_framework.gui.editor` | 25 framework nodes | General workflows |
| `python examples/medical_segmentation_pipeline/launch_gui.py` | 26 nodes (+ MedicalSegmentationLoader) | Medical segmentation workflows |

**Key Takeaway:**
Always use the appropriate launcher for your workflow. If a workflow requires custom nodes, use the custom launcher that imports them first.

---

**Last Updated:** February 7, 2026
**Verified:** GUI launches correctly with both methods
**Status:** ✅ Working
