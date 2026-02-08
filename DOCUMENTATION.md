# Documentation Guide

**All documentation has been organized in the `docs/` folder!**

---

## 🚀 Quick Links

### Getting Started
- **[Quick Reference](docs/getting-started/QUICK_REFERENCE.md)** - One-page cheat sheet ⚡
- **[Getting Started](docs/getting-started/GETTING_STARTED.md)** - 5-minute quick start
- **[Installation Guide](docs/getting-started/INSTALLATION_GUIDE.md)** - Complete installation

### GUI Usage
- **[GUI Launching Guide](docs/gui/GUI_LAUNCHING_GUIDE.md)** ⚡ **CRITICAL** - How to launch GUI correctly
- **[Visual GUI Complete](docs/gui/VISUAL_GUI_COMPLETE.md)** - Full GUI documentation

### Deployment
- **[Deployment Checklist](docs/getting-started/DEPLOYMENT_CHECKLIST.md)** - Step-by-step deployment
- **[Deployment Quick Start](docs/getting-started/README_DEPLOYMENT.md)** - Ready to deploy

### Project Info
- **[Fixes Applied](docs/project/FIXES_APPLIED.md)** - All bugs fixed today
- **[Project Status](docs/project/PROJECT_STATUS.md)** - Project overview

---

## 📚 Complete Documentation Index

**See [docs/INDEX.md](docs/INDEX.md) for complete navigation!**

---

## 📁 Documentation Structure

```
docs/
├── getting-started/     # Installation, setup, quick start (10 files)
├── gui/                 # GUI documentation (9 files)
├── project/             # Project info, contributing (5 files)
├── examples/            # Example workflows (9+ files)
├── segmentation/        # Segmentation guides (5 files)
├── testing/             # Testing documentation (2 files)
├── training/            # Training documentation (1 file)
└── visualization/       # Visualization guides (1 file)
```

**Total: 40+ documentation files, 7,000+ lines**

---

## 🎯 Most Important Documents

### If You're New
1. [docs/getting-started/QUICK_REFERENCE.md](docs/getting-started/QUICK_REFERENCE.md)
2. [docs/getting-started/GETTING_STARTED.md](docs/getting-started/GETTING_STARTED.md)

### If You're Deploying
1. [docs/getting-started/DEPLOYMENT_CHECKLIST.md](docs/getting-started/DEPLOYMENT_CHECKLIST.md)
2. [docs/getting-started/README_DEPLOYMENT.md](docs/getting-started/README_DEPLOYMENT.md)

### If You're Using GUI
1. [docs/gui/GUI_LAUNCHING_GUIDE.md](docs/gui/GUI_LAUNCHING_GUIDE.md) ⚡
2. [docs/gui/VISUAL_GUI_COMPLETE.md](docs/gui/VISUAL_GUI_COMPLETE.md)

### If You Have Problems
1. [docs/getting-started/TROUBLESHOOTING_INSTALL.md](docs/getting-started/TROUBLESHOOTING_INSTALL.md)
2. [docs/project/FIXES_APPLIED.md](docs/project/FIXES_APPLIED.md)

---

## ⚡ Critical: GUI Launching

**If nodes don't appear when loading workflows:**

Use the **custom launcher** not the generic one:

```bash
# ✅ CORRECT - Use this for medical segmentation workflows
python examples/medical_segmentation_pipeline/launch_gui.py

# ❌ WRONG - This won't load custom nodes
python -m medical_imaging_framework.gui.editor
```

See [docs/gui/GUI_LAUNCHING_GUIDE.md](docs/gui/GUI_LAUNCHING_GUIDE.md) for details.

---

## 📖 Full Documentation

For complete framework documentation, see:
- **Main Documentation:** [docs/README.md](docs/README.md)
- **Complete Index:** [docs/INDEX.md](docs/INDEX.md)

---

**Last Updated:** February 7, 2026
