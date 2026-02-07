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

## What the .envrc File Does

The `.envrc` file automatically executes when you enter the directory:

```bash
# Automatically activate virtual environment
source venv/bin/activate

# Set the project root
export PROJECT_ROOT="$PWD"

# Add the project to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Optional development environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONUNBUFFERED=1

# Display welcome message
echo "✅ Medical Imaging Framework environment activated"
echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python: $(python --version)"
echo "📦 Virtual env: venv/"
```

## Environment Variables

The following environment variables are automatically set:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VIRTUAL_ENV` | Path to venv | Virtual environment path |
| `PROJECT_ROOT` | Current directory | Project root path |
| `PYTHONPATH` | `$PWD:$PYTHONPATH` | Python module search path |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDA memory allocation |
| `PYTHONUNBUFFERED` | `1` | Immediate output flushing |
| `PATH` | Modified to use venv | Uses venv Python binaries |

## Benefits of Automatic Activation

Using direnv provides several advantages:

- ✅ **No manual activation needed** - Environment activates automatically
- ✅ **Consistent environment** - Same setup across all terminal sessions
- ✅ **Automatic deactivation** - Cleans up when you leave the directory
- ✅ **Project-specific Python** - Uses correct Python and packages
- ✅ **Proper PYTHONPATH** - Enables correct module imports
- ✅ **Development ready** - All variables set for immediate work

## Verification Checklist

Use this checklist to verify your environment is properly configured:

- [ ] direnv installed (`which direnv` shows path)
- [ ] direnv hook in `~/.bashrc` or `~/.zshrc`
- [ ] `.envrc` file exists at project root
- [ ] `.envrc` is allowed (`direnv allow` completed)
- [ ] `venv/` directory exists with Python packages
- [ ] Automatic activation works (see welcome message)
- [ ] Environment variables set correctly (`echo $PROJECT_ROOT`)
- [ ] Framework imports work (`python -c "import medical_imaging_framework"`)

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
   grep direnv ~/.bashrc  # For bash
   grep direnv ~/.zshrc   # For zsh
   ```
   Should show: `eval "$(direnv hook bash)"`

4. Check direnv status:
   ```bash
   direnv status
   ```

5. Reload your shell:
   ```bash
   exec bash
   # or
   source ~/.bashrc
   ```

### Permission denied or .envrc is blocked?

If you see "direnv: error .envrc is blocked":
```bash
direnv allow
```

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
4. ✅ All environment variables set correctly
5. ✅ Ready for development immediately

**Alternative Activation Methods:**
- `source activate.sh` - Manual activation script
- `source venv/bin/activate` - Standard Python venv activation
- `./venv/bin/python script.py` - One-time command execution

**Quick Verification:**
```bash
cd /path/to/medical_imaging_framework  # Auto-activates
echo $PROJECT_ROOT                      # Should show project path
which python                            # Should show venv/bin/python
python examples/simple_test.py          # Should pass all tests
```

---

**Next Steps:**
- Read [GETTING_STARTED.md](GETTING_STARTED.md) for framework usage
- Check [../README.md](../README.md) for complete documentation
- Run `python examples/simple_test.py` to verify everything works
- See [SERVER_SETUP.md](SERVER_SETUP.md) for server deployment

**Last Updated:** February 7, 2026
