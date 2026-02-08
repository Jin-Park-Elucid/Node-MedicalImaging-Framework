# Installation Guide for New Server

**Last Updated:** February 7, 2026
**Version:** 1.0.0
**Status:** ✅ Tested and Verified

This guide covers installing the Medical Imaging Framework on a fresh server, including all fixes applied during development.

---

## 📋 Prerequisites

- **Python:** 3.8 or higher (tested with 3.10.12)
- **Git:** For cloning the repository
- **direnv:** (optional) For automatic environment activation
- **CUDA:** (optional) For GPU support

---

## 🚀 Quick Installation (Automated)

### Step 1: Clone the Repository

```bash
cd ~/Codes  # or your preferred location
git clone <repository-url> Node-MedicalImaging-Framework
cd Node-MedicalImaging-Framework
```

### Step 2: Run Setup Script

```bash
chmod +x setup_server.sh
./setup_server.sh
```

This will automatically:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Install framework in editable mode
- ✅ Configure environment variables
- ✅ Run diagnostic tests

---

## 📝 Manual Installation (Step-by-Step)

If the automated setup fails or you prefer manual installation:

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Framework

```bash
pip install -e .
```

This installs the package in **editable mode**, so code changes are immediately reflected.

### Step 4: Verify Installation

```bash
python diagnose_import.py
```

You should see:
```
╔══════════════════════════════════════════════════════════════════╗
║                 ✅ ALL DIAGNOSTICS PASSED                        ║
╚══════════════════════════════════════════════════════════════════╝
```

### Step 5: Run Example Test

```bash
python examples/simple_test.py
```

Expected output:
```
🎉 All tests passed! Framework is working correctly.
```

---

## 🔧 Setting Up Automatic Environment Activation (direnv)

### Why Use direnv?

direnv automatically activates the virtual environment when you `cd` into the project directory. No more manual activation!

### Installation

#### Ubuntu/Debian
```bash
sudo apt-get install direnv
```

#### macOS
```bash
brew install direnv
```

### Configuration

Add the direnv hook to your shell configuration:

#### For Bash (most common)

Add to `~/.bashrc`:
```bash
eval "$(direnv hook bash)"
```

#### For Zsh

Add to `~/.zshrc`:
```bash
eval "$(direnv hook zsh)"
```

### Apply Configuration

```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc for zsh

# Or start a new shell
exec bash
```

### Allow direnv for This Project

```bash
cd Node-MedicalImaging-Framework
direnv allow
```

### Test Auto-Activation

```bash
cd ..
cd Node-MedicalImaging-Framework
```

You should see:
```
✅ Node-MedicalImaging-Framework environment activated
📁 Project root: /path/to/Node-MedicalImaging-Framework
🐍 Python: Python 3.10.12
📦 Virtual env: venv/
```

---

## 🐛 Known Issues & Fixes

### Issue 1: Circular Import Error ✅ FIXED

**Error:**
```
❌ Core import failed: cannot import name 'data' from partially initialized module
```

**Cause:** The `nodes/__init__.py` was trying to import a non-existent `data` module.

**Fix Applied:** Commented out the import in `medical_imaging_framework/nodes/__init__.py`:
```python
# from . import data  # TODO: Create data module with DataLoader nodes
```

**Status:** ✅ Fixed in current version

### Issue 2: direnv PS1 Export Error ✅ FIXED

**Error:**
```
direnv: PS1 cannot be exported
```

**Cause:** Virtual environment activation was trying to modify the shell prompt.

**Fix Applied:** Added to `.envrc`:
```bash
export VIRTUAL_ENV_DISABLE_PROMPT=1
```

**Status:** ✅ Fixed in current version

### Issue 3: Package Not Found

**Error:**
```
❌ Package not installed via pip
```

**Solution:**
```bash
pip install -e .
```

### Issue 4: ImagePathLoader Not Available

**Warning:**
```
⚠️  ImagePathLoader node not available (data module not yet implemented)
```

**Status:** Expected behavior - data loading nodes are planned for future implementation. Current workaround: use existing nodes or create custom data loaders as needed.

---

## ✅ Verification Checklist

Use this checklist to verify your installation:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Virtual environment created (`venv/` directory exists)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Framework installed (`pip list | grep Node-MedicalImaging-Framework`)
- [ ] Core imports work (`python -c "import medical_imaging_framework"`)
- [ ] Diagnostic tests pass (`python diagnose_import.py`)
- [ ] Example tests pass (`python examples/simple_test.py`)
- [ ] direnv installed (optional: `which direnv`)
- [ ] direnv hook configured (optional: `grep direnv ~/.bashrc`)
- [ ] Auto-activation works (optional: test by cd)

---

## 🎯 Quick Commands Reference

```bash
# Activate environment (manual)
source activate.sh

# Activate environment (standard venv)
source venv/bin/activate

# Run diagnostics
python diagnose_import.py

# Run example test
python examples/simple_test.py

# Test auto-activation setup
./test_auto_activation.sh

# Check installed nodes
python -c "from medical_imaging_framework.core import NodeRegistry; print(f'{len(NodeRegistry.get_all_nodes())} nodes registered')"

# Launch GUI
python -m medical_imaging_framework.gui.editor

# Reinstall if needed
pip install -e .
```

---

## 📊 Installation Verification

After installation, verify the framework is working:

### Check 1: Package Installed

```bash
pip show Node-MedicalImaging-Framework
```

Expected output:
```
Name: Node-MedicalImaging-Framework
Version: 1.0.0
Location: /path/to/Node-MedicalImaging-Framework
```

### Check 2: Imports Work

```bash
python -c "from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType; print('✅ Imports successful')"
```

### Check 3: Nodes Registered

```bash
python -c "from medical_imaging_framework.core import NodeRegistry; import medical_imaging_framework.nodes; print(f'✅ {len(NodeRegistry.get_all_nodes())} nodes registered')"
```

Expected: `✅ 25 nodes registered`

### Check 4: Can Create Nodes

```bash
python -c "from medical_imaging_framework import NodeRegistry; import medical_imaging_framework.nodes; node = NodeRegistry.create_node('UNet2D', 'test', {'in_channels': 1, 'out_channels': 2}); print(f'✅ Created node: {node.name}')"
```

---

## 🔍 Troubleshooting

### Python Version Issues

**Problem:** Python 3.8+ not available

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3-pip

# macOS
brew install python@3.10
```

### Virtual Environment Issues

**Problem:** Cannot create venv

**Solution:**
```bash
# Ubuntu/Debian - install venv module
sudo apt-get install python3.10-venv

# Then recreate
rm -rf venv
python3 -m venv venv
```

### Import Errors After Installation

**Problem:** `ModuleNotFoundError: No module named 'medical_imaging_framework'`

**Solution:**
```bash
source venv/bin/activate
pip install -e .
```

### CUDA Not Available

**Problem:** PyTorch not detecting GPU

**Solution:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### direnv Not Working

**Problem:** Environment not auto-activating

**Solutions:**

1. Check direnv is installed: `which direnv`
2. Check hook is configured: `grep direnv ~/.bashrc`
3. Reload shell: `source ~/.bashrc` or `exec bash`
4. Allow .envrc: `direnv allow`
5. Run test: `./test_auto_activation.sh`

**Fallback:** Use manual activation: `source activate.sh`

---

## 📦 What Gets Installed

### Framework Components

- **Core Framework** (25 nodes total)
  - 14 Network architectures (UNet, ResNet, Transformers, etc.)
  - 4 Training nodes (Trainer, Optimizer, Loss, Checkpoint)
  - 3 Inference nodes (Predictor, BatchPredictor, Metrics)
  - 4 Visualization nodes (Viewer, Plotter, Print, Overlay)

### Dependencies

Major packages installed:
- PyTorch 2.0+
- torchvision
- MONAI (medical imaging library)
- PyQt5 (GUI framework)
- nibabel (NIfTI support)
- pydicom (DICOM support)
- SimpleITK
- scikit-learn, scikit-image
- matplotlib, opencv-python
- albumentations (augmentation)
- tensorboard

---

## 🚢 Deploying to New Server

### Quick Deploy Checklist

1. **Copy repository** to new server
2. **Run setup script**: `./setup_server.sh`
3. **Test installation**: `python diagnose_import.py`
4. **Run example**: `python examples/simple_test.py`
5. **Configure direnv** (optional): Add hook to `~/.bashrc`
6. **Done!** ✅

### Files to Copy

Minimum files needed:
```
Node-MedicalImaging-Framework/
├── medical_imaging_framework/  # Source code
├── examples/                   # Example scripts
├── docs/                       # Documentation
├── requirements.txt            # Dependencies
├── setup.py                    # Installation config
├── setup_server.sh            # Setup script
├── diagnose_import.py         # Diagnostic tool
├── .envrc                     # direnv config
└── activate.sh                # Manual activation
```

### What NOT to Copy

Don't copy these (will be regenerated):
```
venv/                          # Virtual environment
*.pyc, __pycache__/           # Python cache
.git/                         # Git history (optional)
*.egg-info/                   # Package info
workflows/                    # Generated workflows
```

---

## 🔐 SSH X11 Forwarding (For GUI)

If you want to run the GUI on a remote server:

### On Your Local Machine

Edit `~/.ssh/config`:
```
Host your-server
    HostName server.address.com
    User your-username
    ForwardX11 yes
    ForwardX11Trusted yes
```

### Connect with X11

```bash
ssh -X your-server
```

### On the Server

```bash
cd Node-MedicalImaging-Framework
source activate.sh
python -m medical_imaging_framework.gui.editor
```

The GUI will appear on your local machine!

See `docs/gui/SSH_X11_FORWARDING_GUIDE.md` for detailed instructions.

---

## 📚 Next Steps After Installation

1. **Read Documentation**
   - `docs/getting-started/GETTING_STARTED.md` - Quick start guide
   - `docs/getting-started/QUICK_REFERENCE.md` - One-page reference
   - `docs/README.md` - Complete documentation

2. **Explore Examples**
   - `examples/simple_test.py` - Framework test
   - `docs/examples/medical-segmentation/` - Full example

3. **Launch GUI**
   ```bash
   python -m medical_imaging_framework.gui.editor
   ```

4. **Build Your First Pipeline**
   - Create nodes
   - Connect them in a graph
   - Execute the workflow

---

## 💡 Tips for Production Deployment

### Use System-Wide Installation (Optional)

Instead of venv, you can install system-wide:
```bash
sudo pip install -e .
```

### Set Up as Service (Optional)

Create systemd service for automated workflows:
```bash
sudo nano /etc/systemd/system/medical-imaging.service
```

### Use Docker (Future)

Docker deployment coming soon!

---

## 📞 Support & Resources

- **Documentation:** `docs/` folder
- **Troubleshooting:** `docs/getting-started/TROUBLESHOOTING_INSTALL.md`
- **Examples:** `examples/` folder
- **Diagnostic Tool:** `python diagnose_import.py`
- **Test Script:** `./test_auto_activation.sh`

---

## ✅ Installation Complete!

Your Medical Imaging Framework is now ready to use! 🎉

**Quick verification:**
```bash
python diagnose_import.py && python examples/simple_test.py
```

Both should show: **✅ ALL TESTS PASSED**

**Start building:**
```bash
python -m medical_imaging_framework.gui.editor
```

---

**Last Updated:** February 7, 2026
**Maintained By:** Medical Imaging Framework Team
