# Deployment Checklist for New Server

**Last Updated:** February 7, 2026
**Version:** 1.0.0
**Status:** ✅ Production Ready

Complete checklist for deploying the Medical Imaging Framework to a new server.

---

## 📋 Pre-Deployment Checklist

- [ ] Server has Python 3.8+ installed
- [ ] Server has Git installed
- [ ] You have SSH access to the server
- [ ] (Optional) X11 forwarding configured for GUI access
- [ ] (Optional) direnv installed for auto-activation

---

## 🚀 Deployment Steps

### Step 1: Clone Repository

```bash
# On the new server
cd ~/Codes  # or your preferred location
git clone <repository-url> Node-MedicalImaging-Framework
cd Node-MedicalImaging-Framework
```

**Verify:**
```bash
ls -la
# Should see: medical_imaging_framework/, examples/, docs/, setup.py, etc.
```

---

### Step 2: Run Automated Setup

```bash
chmod +x setup_server.sh
./setup_server.sh
```

**What this does:**
- ✅ Checks Python version
- ✅ Creates virtual environment
- ✅ Installs dependencies
- ✅ Installs framework in editable mode
- ✅ Runs diagnostic tests

**Expected output:**
```
✅ ALL DIAGNOSTICS PASSED
25 nodes registered
```

---

### Step 3: Configure direnv (Optional but Recommended)

#### 3a. Add direnv Hook

```bash
# For bash users
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# For zsh users
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
```

#### 3b. Allow direnv

```bash
direnv allow
```

#### 3c. Test Auto-Activation

```bash
./test_auto_activation.sh
```

**Expected:** All checks pass ✅

---

### Step 4: Verify Installation

#### 4a. Run Diagnostic Tool

```bash
python diagnose_import.py
```

**Expected output:**
```
╔══════════════════════════════════════════════════════════════════╗
║                 ✅ ALL DIAGNOSTICS PASSED                        ║
╚══════════════════════════════════════════════════════════════════╝

✅ Package installed: version 1.0.0
✅ Core imports successful
✅ Nodes package imported
✅ 25 nodes registered
✅ Can create nodes and graphs
```

#### 4b. Run Example Test

```bash
python examples/simple_test.py
```

**Expected output:**
```
🎉 All tests passed! Framework is working correctly.
```

---

### Step 5: Test GUI (If Using Remote GUI)

#### 5a. Connect with X11 Forwarding

From your local machine:
```bash
ssh -X server-name
```

#### 5b. Launch Generic GUI

```bash
cd Node-MedicalImaging-Framework
source activate.sh  # or let direnv activate
python -m medical_imaging_framework.gui.editor
```

**Expected:** GUI window appears on your local machine ✅

#### 5c. Launch Medical Segmentation GUI

```bash
python examples/medical_segmentation_pipeline/launch_gui.py
```

**Expected output:**
```
================================================================================
MEDICAL IMAGING FRAMEWORK - GUI EDITOR
================================================================================

Custom nodes registered:
  ✓ MedicalSegmentationLoader

Total nodes available: 26

Workflow files available:
  • training_workflow.json
  • testing_workflow.json

================================================================================
```

GUI opens with all 26 nodes available ✅

---

## ✅ Verification Checklist

After deployment, verify these items:

### Core Framework
- [ ] Python version 3.8+ (`python --version`)
- [ ] Virtual environment created (`ls venv/`)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Framework installed (`pip show Node-MedicalImaging-Framework`)
- [ ] Core imports work (`python -c "import medical_imaging_framework"`)

### Node System
- [ ] 25 framework nodes registered (`python diagnose_import.py`)
- [ ] Can create nodes (`python examples/simple_test.py`)
- [ ] Medical segmentation node available (with custom launcher)

### Environment Activation
- [ ] direnv installed (`which direnv`)
- [ ] direnv hook configured (`grep direnv ~/.bashrc`)
- [ ] .envrc allowed (`direnv status | grep "allowed true"`)
- [ ] Auto-activation works (`./test_auto_activation.sh`)
- [ ] Manual activation works (`source activate.sh`)

### GUI (If Applicable)
- [ ] X11 forwarding works (`echo $DISPLAY`)
- [ ] Generic GUI launches (`python -m medical_imaging_framework.gui.editor`)
- [ ] Custom GUI launches (`python examples/medical_segmentation_pipeline/launch_gui.py`)
- [ ] Can load workflows (load `training_workflow.json`)
- [ ] Nodes appear in loaded workflows

---

## 🎯 Quick Verification Commands

Run these on the new server to verify everything:

```bash
# 1. Test imports
python diagnose_import.py

# 2. Test examples
python examples/simple_test.py

# 3. Check node count (should be 25)
python -c "from medical_imaging_framework.core import NodeRegistry; import medical_imaging_framework.nodes; print(f'{len(NodeRegistry.get_all_nodes())} nodes')"

# 4. Test auto-activation (if using direnv)
./test_auto_activation.sh

# 5. Check package version
pip show Node-MedicalImaging-Framework | grep Version

# 6. Test GUI (if X11 available)
python -m medical_imaging_framework.gui.editor &
sleep 2
pkill -f gui.editor

# 7. Test custom launcher
python examples/medical_segmentation_pipeline/launch_gui.py &
sleep 2
pkill -f launch_gui
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'medical_imaging_framework'`

**Solution:**
```bash
source venv/bin/activate
pip install -e .
```

### Issue 2: direnv Not Working

**Symptom:** Environment doesn't auto-activate

**Solution:**
```bash
# Add hook
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# Allow directory
direnv allow

# Test
./test_auto_activation.sh
```

### Issue 3: GUI Doesn't Launch

**Symptom:** GUI window doesn't appear

**Solutions:**

**Check X11:**
```bash
echo $DISPLAY
# Should show something like: localhost:10.0
```

**Test X11:**
```bash
xclock
# Clock should appear on local machine
```

**Fix SSH config** (on local machine, `~/.ssh/config`):
```
Host your-server
    ForwardX11 yes
    ForwardX11Trusted yes
```

### Issue 4: Nodes Don't Appear in GUI

**Symptom:** Loaded workflow shows no nodes

**Cause:** Using generic GUI launcher instead of custom launcher

**Solution:**
```bash
# Use the medical segmentation launcher
python examples/medical_segmentation_pipeline/launch_gui.py

# NOT this (for medical segmentation workflows):
python -m medical_imaging_framework.gui.editor
```

### Issue 5: PS1 Export Error

**Symptom:** `direnv: PS1 cannot be exported`

**Status:** ✅ Fixed in current version

**Verification:** `.envrc` contains `export VIRTUAL_ENV_DISABLE_PROMPT=1`

### Issue 6: Circular Import Error

**Symptom:** `cannot import name 'data' from partially initialized module`

**Status:** ✅ Fixed in current version

**Verification:** `nodes/__init__.py` has data import commented out

---

## 📚 Documentation Reference

After deployment, users should reference:

- **Installation:** `INSTALLATION_GUIDE.md`
- **GUI Launching:** `GUI_LAUNCHING_GUIDE.md`
- **Environment Setup:** `docs/getting-started/ENVIRONMENT_SETUP.md`
- **Quick Start:** `docs/getting-started/GETTING_STARTED.md`
- **Troubleshooting:** `docs/getting-started/TROUBLESHOOTING_INSTALL.md`
- **Fixes Applied:** `FIXES_APPLIED.md`

---

## 🔄 Update Procedure (Pulling New Changes)

If you need to update an existing deployment:

```bash
cd Node-MedicalImaging-Framework

# Pull latest changes
git pull origin main

# Activate environment
source activate.sh

# Update dependencies (if requirements changed)
pip install -r requirements.txt

# Reinstall framework
pip install -e .

# Run diagnostics
python diagnose_import.py

# Test
python examples/simple_test.py
```

---

## 🗂️ Files to Include in Deployment

### Required Files
```
Node-MedicalImaging-Framework/
├── medical_imaging_framework/     # Core framework code
├── examples/                      # Example scripts and workflows
│   └── medical_segmentation_pipeline/
│       ├── launch_gui.py          # ⚠️ Important for GUI
│       ├── custom_dataloader.py   # ⚠️ Required for workflows
│       ├── training_workflow.json
│       └── testing_workflow.json
├── docs/                          # Documentation
├── requirements.txt               # Dependencies
├── setup.py                       # Installation config
├── setup_server.sh               # Automated setup
├── diagnose_import.py            # Diagnostic tool
├── test_auto_activation.sh       # direnv test
├── activate.sh                   # Manual activation
├── .envrc                        # direnv config
├── INSTALLATION_GUIDE.md         # Installation docs
├── GUI_LAUNCHING_GUIDE.md        # GUI launch docs
├── DEPLOYMENT_CHECKLIST.md       # This file
└── FIXES_APPLIED.md              # Applied fixes log
```

### Optional Files (Don't Copy)
```
venv/                             # Virtual environment (regenerate)
__pycache__/                      # Python cache
*.pyc                             # Compiled Python
.git/                             # Git history
*.egg-info/                       # Package info
workflows/                        # Generated workflows
checkpoints/                      # Model checkpoints
visualization_output/             # Output images
```

---

## 🎓 Training New Users

After deployment, new users should:

1. **Read Getting Started:**
   ```bash
   cat docs/getting-started/GETTING_STARTED.md
   ```

2. **Run Example Test:**
   ```bash
   python examples/simple_test.py
   ```

3. **Launch GUI:**
   ```bash
   # For general use:
   python -m medical_imaging_framework.gui.editor

   # For medical segmentation:
   python examples/medical_segmentation_pipeline/launch_gui.py
   ```

4. **Load and Run Workflow:**
   - Load `training_workflow.json` in GUI
   - Verify all nodes appear
   - Click "Execute Workflow"

---

## 📊 Deployment Success Criteria

Deployment is successful when:

✅ All diagnostic tests pass
✅ Example tests run successfully
✅ 25 framework nodes registered
✅ Environment activation works (manual or auto)
✅ GUI launches (if applicable)
✅ Workflows load with all nodes visible
✅ Custom launcher shows 26 nodes (25 + MedicalSegmentationLoader)

---

## 🚨 Critical Notes for Deployment

### 1. GUI Launcher Selection

**CRITICAL:** When working with medical segmentation workflows:
- ✅ **USE:** `python examples/medical_segmentation_pipeline/launch_gui.py`
- ❌ **DON'T USE:** `python -m medical_imaging_framework.gui.editor`

The custom launcher registers the `MedicalSegmentationLoader` node required by the workflows.

### 2. Environment Activation

**CRITICAL:** Always activate environment before running scripts:
```bash
source activate.sh
# OR let direnv auto-activate
```

### 3. X11 Forwarding

**CRITICAL:** For remote GUI access:
- Connect with `ssh -X` not just `ssh`
- Verify `$DISPLAY` is set
- Test with `xclock` before launching GUI

---

## 📞 Support

If deployment issues persist:

1. **Check diagnostic output:**
   ```bash
   python diagnose_import.py 2>&1 | tee diagnostic.log
   ```

2. **Check environment:**
   ```bash
   which python
   pip list | grep -E "(torch|medical-imaging)"
   echo $PYTHONPATH
   ```

3. **Review fixes log:**
   ```bash
   cat FIXES_APPLIED.md
   ```

4. **Consult documentation:**
   - `INSTALLATION_GUIDE.md`
   - `docs/getting-started/TROUBLESHOOTING_INSTALL.md`

---

## ✅ Deployment Complete!

Once all checks pass, your deployment is complete and ready for use! 🎉

**Next steps:**
- Train users on GUI usage
- Set up data directories if needed
- Configure workflows for your specific use case
- Review documentation in `docs/` folder

---

**Last Updated:** February 7, 2026
**Tested On:** Ubuntu Server with Python 3.10.12
**Status:** ✅ Production Ready
