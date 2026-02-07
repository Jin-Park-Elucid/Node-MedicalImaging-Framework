# Server Setup Guide for Node-MedicalImaging-Framework

## 🚀 Quick Setup (Automated)

### One-Command Setup

```bash
# On the server, after cloning the repository
cd Node-MedicalImaging-Framework
chmod +x setup_server.sh
./setup_server.sh
```

That's it! The script will:
- ✅ Check Python version (requires 3.8+)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Install the framework package
- ✅ Configure environment variables
- ✅ Set up direnv (if available)
- ✅ Test the installation

---

## 📋 Manual Setup (Step-by-Step)

If you prefer to set up manually or the automated script fails:

### Step 1: Clone the Repository

```bash
cd ~/Codes
git clone git@github.com:Jin-Park-Elucid/Node-MedicalImaging-Framework.git
cd Node-MedicalImaging-Framework
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 4: Test

```bash
python examples/simple_test.py
```

---

## 🔧 Environment Activation

### Option 1: Using activate.sh (Recommended)

```bash
cd Node-MedicalImaging-Framework
source activate.sh
```

### Option 2: Using direnv (Automatic)

If direnv is installed:

```bash
# Create .envrc
cat > .envrc << 'EOF'
source venv/bin/activate
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD:$PYTHONPATH"
EOF

direnv allow

# Now just cd into directory
cd Node-MedicalImaging-Framework  # Auto-activates!
```

---

## 🎨 Running GUI via SSH X11 Forwarding

```bash
# From local laptop
ssh hendrix

# On server
cd Node-MedicalImaging-Framework
source activate.sh
python -m medical_imaging_framework.gui.editor
```

GUI appears on your local laptop! 🎉

---

**For complete documentation, see the full guide above.**
