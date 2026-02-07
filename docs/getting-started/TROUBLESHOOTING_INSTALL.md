# Installation Troubleshooting Guide

## Common Import Error: Circular Import

### Error Message
```
❌ Import failed: cannot import name 'data' from partially initialized module
'medical_imaging_framework.nodes' (most likely due to a circular import)
```

### Quick Fix

**Option 1: Clean and Reinstall**
```bash
cd Node-MedicalImaging-Framework

# 1. Remove any cached files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# 2. Reinstall the package
source venv/bin/activate
pip uninstall -y medical-imaging-framework
pip install -e .

# 3. Test the import
python -c "import medical_imaging_framework.nodes; print('✅ Import successful!')"
```

**Option 2: Manual Activation and Test**
```bash
cd Node-MedicalImaging-Framework
source activate.sh

# Test import
python -c "import medical_imaging_framework; print('✅ Framework imported!')"
python examples/simple_test.py
```

---

## After Successful Installation

### 1. Verify Installation
```bash
# Should show the framework installed
pip list | grep medical-imaging-framework

# Test basic import
python -c "import medical_imaging_framework; print(medical_imaging_framework.__version__)"

# Run full test suite
python examples/simple_test.py
```

### 2. Quick Start Options

#### Option A: Run GUI (Recommended for First-Time Users)
```bash
# Activate environment
source activate.sh

# Launch GUI
python -m medical_imaging_framework.gui.editor

# Or use the example GUI
cd examples/medical_segmentation_pipeline
python launch_gui.py
```

**What you'll see:**
- Visual workflow editor
- Node library on the left
- Canvas in the center
- Load pre-built workflows or create your own

#### Option B: Run Example Workflow (Command Line)
```bash
# Run medical segmentation example
cd examples/medical_segmentation_pipeline

# 1. Generate synthetic data (optional, for testing)
python download_dataset.py

# 2. Train a model
python train_pipeline.py

# 3. Test the model
python test_pipeline.py

# 4. View results
ls results/visualizations/
```

#### Option C: Quick Python API Test
```bash
python3 << 'EOF'
from medical_imaging_framework import NodeRegistry, ComputationalGraph, GraphExecutor
import medical_imaging_framework.nodes

# Create a simple graph
graph = ComputationalGraph("Test Pipeline")

# Create a UNet node
unet = NodeRegistry.create_node('UNet2D', 'test_unet', config={
    'in_channels': 1,
    'out_channels': 2,
    'base_channels': 32
})
graph.add_node(unet)

# Validate
is_valid, errors = graph.validate()
print(f"✅ Graph valid: {is_valid}")
print(f"✅ UNet node created successfully!")
print(f"✅ Framework working correctly!")
EOF
```

---

## Next Steps by Use Case

### For Learning the Framework
1. **Read Documentation**
   ```bash
   cat docs/getting-started/GETTING_STARTED.md
   cat docs/getting-started/QUICK_REFERENCE.md
   ```

2. **Try the GUI**
   ```bash
   python -m medical_imaging_framework.gui.editor
   ```

3. **Run Examples**
   ```bash
   python examples/simple_test.py
   cd examples/medical_segmentation_pipeline
   python train_pipeline.py
   ```

### For Medical Segmentation
1. **Review Available Networks**
   ```bash
   cat docs/segmentation/NETWORK_ARCHITECTURES.md
   ```

2. **Prepare Your Data**
   - Organize in `data/train/images/` and `data/train/masks/`
   - See: `docs/segmentation/DATALOADER.md`

3. **Choose a Workflow**
   ```bash
   ls examples/medical_segmentation_pipeline/workflows/
   # Options: unet_training.json, vnet_training.json, etc.
   ```

4. **Train and Test**
   ```bash
   python train_pipeline.py
   python test_pipeline.py
   ```

### For Remote GUI Access (SSH X11)
1. **Setup X11 Forwarding**
   ```bash
   # On local machine
   ssh -X your-server

   # Or add to ~/.ssh/config
   cat >> ~/.ssh/config << 'EOF'
   Host your-server
       ForwardX11 yes
       ForwardX11Trusted yes
   EOF
   ```

2. **Test X11**
   ```bash
   xeyes  # Should open a window on your local machine
   ```

3. **Launch GUI**
   ```bash
   cd Node-MedicalImaging-Framework
   source activate.sh
   python -m medical_imaging_framework.gui.editor
   ```

   See: `docs/gui/SSH_X11_FORWARDING_GUIDE.md`

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'medical_imaging_framework'"
**Solution:**
```bash
source venv/bin/activate
pip install -e .
```

### Issue 2: "CUDA out of memory"
**Solution:**
- Reduce batch size in workflow configs
- Use CPU instead: set `"device": "cpu"` in trainer config

### Issue 3: GUI doesn't launch
**Solution:**
```bash
# Check PyQt5 is installed
pip install PyQt5

# For SSH, ensure X11 forwarding works
xeyes  # Test X11
```

### Issue 4: Import takes a long time
**Solution:**
- This is normal on first import (torch initialization)
- Subsequent imports will be faster

---

## Verification Checklist

After installation, verify these work:

- [ ] Virtual environment activates (`source activate.sh`)
- [ ] Framework imports (`python -c "import medical_imaging_framework"`)
- [ ] Nodes import (`python -c "import medical_imaging_framework.nodes"`)
- [ ] Simple test passes (`python examples/simple_test.py`)
- [ ] Can create nodes (`NodeRegistry.create_node(...)`)
- [ ] GUI launches (if using GUI)

---

## Quick Reference Card

```bash
# Activate environment
source activate.sh
# or (with direnv)
cd Node-MedicalImaging-Framework  # Auto-activates

# Test installation
python examples/simple_test.py

# Launch GUI
python -m medical_imaging_framework.gui.editor

# Run example
cd examples/medical_segmentation_pipeline
python train_pipeline.py

# Get help
cat docs/INDEX.md
cat docs/getting-started/QUICK_REFERENCE.md
```

---

## Getting Help

**Documentation:**
- Main docs: `docs/README.md`
- Getting started: `docs/getting-started/GETTING_STARTED.md`
- GUI guide: `docs/gui/VISUAL_GUI_COMPLETE.md`
- Segmentation: `docs/segmentation/README.md`

**Quick Navigation:**
```bash
cat docs/QUICK_NAVIGATION.md
cat docs/INDEX.md
```

---

**Installation complete! Choose your next step above based on your use case.** 🚀
