# 🚀 Next Steps After Installation

## If You Encountered the Import Error

Run this fix script on the server:

```bash
cd Node-MedicalImaging-Framework
./fix_import.sh
```

This will:
- Clean all cached Python files
- Reinstall the package
- Test the installation
- Verify everything works

---

## After Successful Installation

### 1️⃣ Quick Test (30 seconds)

```bash
# Activate environment
source activate.sh

# Run quick test
python examples/simple_test.py
```

**Expected output:**
```
✅ Node Registry: 23 nodes registered
✅ Node Creation: All nodes work
✅ Graph Building: Graphs work
✅ All tests passed!
```

---

### 2️⃣ Try the Visual GUI (Recommended)

```bash
# Launch the GUI
python -m medical_imaging_framework.gui.editor
```

**What you'll see:**
- Node library on the left (23 nodes in 5 categories)
- Canvas in the center (visual workflow editor)
- Menu bar with File, Workflow, Help options

**Try this:**
1. Click "File → Load Workflow"
2. Select `examples/medical_segmentation_pipeline/training_workflow.json`
3. See the visual nodes appear on canvas
4. Click "Workflow → Validate" to check it
5. Explore the nodes, drag them around

---

### 3️⃣ Run a Complete Example

```bash
cd examples/medical_segmentation_pipeline

# Step 1: Generate test data (optional)
python download_dataset.py
# Choose option 1, press Enter for defaults

# Step 2: Train a model (takes ~5-10 minutes)
python train_pipeline.py

# Step 3: Test the model
python test_pipeline.py

# Step 4: View results
ls results/visualizations/
# You'll see overlay_*.png files showing predictions
```

---

### 4️⃣ Quick API Test

Test the Python API directly:

```bash
python3 << 'EOF'
from medical_imaging_framework import NodeRegistry, ComputationalGraph
import medical_imaging_framework.nodes

# Show available nodes
print(f"Available nodes: {len(NodeRegistry.list_nodes())}")

# Create a simple graph
graph = ComputationalGraph("My Pipeline")

# Create a UNet model
unet = NodeRegistry.create_node('UNet2D', 'my_unet', config={
    'in_channels': 1,
    'out_channels': 2,
    'base_channels': 32
})
graph.add_node(unet)

print(f"✅ Created graph with UNet node")
print(f"✅ Framework working perfectly!")
EOF
```

---

## 🎯 What to Do Based on Your Goal

### Goal: Learn the Framework
**Start here:**
1. Read: `docs/getting-started/QUICK_REFERENCE.md`
2. Read: `docs/getting-started/GETTING_STARTED.md`
3. Try: Run `python examples/simple_test.py`
4. Explore: Launch the GUI and load example workflows

**Time needed:** 30 minutes

---

### Goal: Medical Image Segmentation
**Start here:**
1. Read: `docs/segmentation/README.md`
2. Review networks: `docs/segmentation/NETWORK_ARCHITECTURES.md`
3. Prepare your data: See `docs/segmentation/DATALOADER.md`
4. Choose workflow: `ls examples/medical_segmentation_pipeline/workflows/`
5. Train and test!

**Time needed:** 1-2 hours for first model

---

### Goal: Use the GUI
**Start here:**
1. Launch: `python -m medical_imaging_framework.gui.editor`
2. Read: `docs/gui/VISUAL_GUI_COMPLETE.md`
3. Load example: `training_workflow.json`
4. Learn shortcuts: Press `Ctrl+?` for help

**Time needed:** 15 minutes to get comfortable

---

### Goal: Deploy on Server
**You're already here!** ✅

**Next steps:**
1. Test GUI via SSH X11: See `docs/gui/SSH_X11_FORWARDING_GUIDE.md`
2. Run workflows: `cd examples/medical_segmentation_pipeline`
3. Monitor training: Use `screen` or `tmux` for long jobs

---

## 📚 Essential Documentation

Quick access to key docs:

```bash
# Navigation hub
cat docs/INDEX.md

# Quick commands
cat docs/getting-started/QUICK_REFERENCE.md

# Getting started tutorial
cat docs/getting-started/GETTING_STARTED.md

# Troubleshooting
cat docs/getting-started/TROUBLESHOOTING_INSTALL.md

# GUI guide
cat docs/gui/VISUAL_GUI_COMPLETE.md
```

---

## 🎓 Learning Path

### Beginner (Just Starting)
1. ✅ Complete installation (you're here!)
2. Run `python examples/simple_test.py`
3. Launch GUI: `python -m medical_imaging_framework.gui.editor`
4. Load and explore `training_workflow.json`
5. Read `GETTING_STARTED.md`

### Intermediate (Ready to Build)
1. Read available node types
2. Create a custom workflow in GUI
3. Run training on synthetic data
4. Understand the results
5. Try different architectures

### Advanced (Production Use)
1. Prepare real medical data
2. Choose appropriate architecture
3. Configure hyperparameters
4. Train and validate
5. Deploy and monitor

---

## 🛠️ Common Commands

```bash
# Activate environment (always first!)
source activate.sh

# Test installation
python examples/simple_test.py

# Launch GUI
python -m medical_imaging_framework.gui.editor

# List available nodes
python -c "from medical_imaging_framework import NodeRegistry; import medical_imaging_framework.nodes; print(f'{len(NodeRegistry.list_nodes())} nodes available')"

# Check version
python -c "import medical_imaging_framework; print(medical_imaging_framework.__version__)"

# Run example workflow
cd examples/medical_segmentation_pipeline
python train_pipeline.py
```

---

## 🆘 If Something Goes Wrong

### Import Error Still Persists
```bash
./fix_import.sh
```

### Can't Find Commands
```bash
source activate.sh  # Make sure environment is activated
```

### GUI Won't Launch
```bash
pip install PyQt5  # Install GUI dependencies
xeyes              # Test X11 forwarding (if on SSH)
```

### Need More Help
```bash
cat docs/getting-started/TROUBLESHOOTING_INSTALL.md
cat docs/INDEX.md  # Find relevant documentation
```

---

## ✅ Quick Verification Checklist

After installation, make sure these work:

- [ ] `source activate.sh` - Activates environment
- [ ] `python -c "import medical_imaging_framework"` - Basic import
- [ ] `python examples/simple_test.py` - Tests pass
- [ ] `python -m medical_imaging_framework.gui.editor` - GUI launches
- [ ] Can load and view example workflows

---

## 🎉 You're All Set!

Pick your next step from above and start exploring the framework!

**Recommended first action:**
```bash
source activate.sh
python examples/simple_test.py
python -m medical_imaging_framework.gui.editor
```

Good luck! 🚀
