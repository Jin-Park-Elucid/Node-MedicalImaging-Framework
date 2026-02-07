# Environment Setup Guide

## ✅ Automatic Activation (Using direnv)

Your environment is configured to **automatically activate** when you enter this directory!

### How it Works

This project uses [direnv](https://direnv.net/) to automatically activate the virtual environment when you `cd` into the directory.

**Configuration files:**
- `.envrc` - Direnv configuration (auto-activates venv)
- `venv/` - Python virtual environment
- `activate.sh` - Manual activation script

### Current Setup Status

✅ Virtual environment created: `venv/`
✅ Direnv configuration: `.envrc`
✅ Direnv allowed for this directory
✅ Framework installed in development mode
✅ All dependencies installed

### Testing Automatic Activation

1. Exit this directory:
   ```bash
   cd ..
   ```

2. Re-enter the directory:
   ```bash
   cd medical_imaging_framework
   ```

3. You should see:
   ```
   ✅ Medical Imaging Framework environment activated
   📁 Project root: /path/to/medical_imaging_framework
   🐍 Python: Python 3.10.12
   📦 Virtual env: venv/
   ```

### Verify It's Working

Check that the virtual environment is active:

```bash
# Should show venv/bin/python
which python

# Should show the framework version
python -c "import medical_imaging_framework; print(medical_imaging_framework.__version__)"

# Run the test
python examples/simple_test.py
```

## Alternative Activation Methods

### Method 1: Manual Activation Script

If direnv is not working, use the manual activation script:

```bash
source activate.sh
```

This will:
- Activate the virtual environment
- Set environment variables
- Display helpful information

### Method 2: Standard venv Activation

Traditional Python virtual environment activation:

```bash
source venv/bin/activate
```

### Method 3: One-time Commands

Run commands directly without activation:

```bash
./venv/bin/python examples/simple_test.py
```

## Direnv Setup (If Not Installed)

If you don't have direnv installed:

### Ubuntu/Debian
```bash
sudo apt-get install direnv
```

### macOS
```bash
brew install direnv
```

### Configure Shell Hook

Add to your shell configuration:

**For Bash** (`~/.bashrc`):
```bash
eval "$(direnv hook bash)"
```

**For Zsh** (`~/.zshrc`):
```bash
eval "$(direnv hook zsh)"
```

**For Fish** (`~/.config/fish/config.fish`):
```fish
direnv hook fish | source
```

After adding the hook, restart your shell or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Environment Variables

The `.envrc` file sets these variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PROJECT_ROOT` | Current directory | Project root path |
| `PYTHONPATH` | `$PWD:$PYTHONPATH` | Python module search path |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDA memory allocation |
| `PYTHONUNBUFFERED` | `1` | Immediate output flushing |

## Troubleshooting

### Direnv not activating?

1. Check if direnv is installed:
   ```bash
   which direnv
   ```

2. Check if .envrc is allowed:
   ```bash
   direnv allow .
   ```

3. Check if shell hook is configured:
   ```bash
   echo $DIRENV_DIR
   ```

### Permission denied on .envrc?

Make sure the file has correct permissions:
```bash
chmod +x .envrc
```

### Virtual environment not found?

Recreate it:
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Framework not importing?

Make sure it's installed:
```bash
source venv/bin/activate
pip install -e .
```

## Deactivating the Environment

When using direnv, the environment **automatically deactivates** when you leave the directory!

```bash
cd ..  # Environment automatically deactivates
```

For manual activation:
```bash
deactivate
```

## Quick Commands

After activation, try these:

```bash
# Run tests
python examples/simple_test.py

# Launch GUI
python -m medical_imaging_framework.gui.editor

# Check installation
pip list | grep medical-imaging-framework

# Run workflow example
python examples/segmentation_workflow.py
```

## Summary

Your environment is configured for **automatic activation** via direnv:

1. ✅ Enter directory → Environment activates automatically
2. ✅ Exit directory → Environment deactivates automatically
3. ✅ No manual activation needed!

Alternatively, use:
- `source activate.sh` - Manual activation
- `source venv/bin/activate` - Standard Python venv

---

**Next Steps:**
- Read [GETTING_STARTED.md](GETTING_STARTED.md) for framework usage
- Check [README.md](README.md) for complete documentation
- Run `python examples/simple_test.py` to verify everything works
