# Medical Imaging Framework - Documentation Index

Welcome to the Medical Imaging Framework documentation!

**All documentation is now organized in the `docs/` folder with clear categories.**

---

## 📁 Documentation Structure

```
docs/
├── README.md                          # Main framework documentation
├── INDEX.md                           # This file - navigation hub
│
├── getting-started/                   # Quick start guides
│   ├── QUICK_REFERENCE.md            # One-page cheat sheet ⚡
│   ├── GETTING_STARTED.md            # Beginner's guide
│   └── ENVIRONMENT_SETUP.md          # Virtual environment setup
│
├── project/                           # Project information
│   ├── CONTRIBUTING.md               # How to contribute
│   └── PROJECT_STATUS.md             # Project overview & stats
│
├── gui/                               # GUI documentation
│   ├── VISUAL_GUI_COMPLETE.md        # Complete visual implementation
│   ├── VISUAL_GUI_QUICK_REFERENCE.md # Quick reference card
│   └── LAUNCHING_GUI_METHODS.md      # How to launch GUI (methods & SSH)
│
└── examples/
    └── medical-segmentation/          # Medical segmentation example
        ├── README.md                  # Example overview
        ├── WORKFLOWS_COMPLETE.md      # Workflow documentation
        └── gui/                       # GUI-specific docs
            ├── GUI_GUIDE.md
            ├── QUICKSTART_GUI.md
            ├── VISUAL_GUI_GUIDE.md
            └── ... (more GUI docs)
```

---

## 🎯 Quick Navigation

### For New Users

1. **[Quick Reference](getting-started/QUICK_REFERENCE.md)** ⚡ - One-page reference
   - Common tasks
   - Quick commands
   - Code snippets

2. **[Getting Started](getting-started/GETTING_STARTED.md)** - 5-minute quick start
   - Basic concepts
   - Simple examples
   - Creating custom nodes

3. **[Installation Guide](getting-started/INSTALLATION_GUIDE.md)** - Complete installation
   - Automated and manual installation
   - All fixes included
   - Troubleshooting

4. **[Environment Setup](getting-started/ENVIRONMENT_SETUP.md)** - Virtual environment guide
   - Automatic activation (direnv)
   - Manual activation methods
   - Troubleshooting

5. **[Main Documentation](README.md)** - Complete framework docs
   - Full feature overview
   - Architecture details
   - API reference

### For GUI Users

6. **[GUI Launching Guide](gui/GUI_LAUNCHING_GUIDE.md)** ⚡ - **CRITICAL** launcher guide
   - Generic vs custom launcher
   - Loading workflows correctly
   - Troubleshooting nodes not appearing

7. **[Visual GUI Complete](gui/VISUAL_GUI_COMPLETE.md)** - Full visual implementation
   - All visual features
   - Implementation summary
   - Complete guide

8. **[GUI Quick Reference](gui/VISUAL_GUI_QUICK_REFERENCE.md)** ⚡ - Quick reference card
   - Keyboard shortcuts
   - Mouse controls
   - Common actions

9. **[GUI Launching Methods](gui/LAUNCHING_GUI_METHODS.md)** - Complete launch guide
   - python -m vs script execution
   - Generic vs example-specific GUI
   - Remote GUI access (SSH X11)

### For Example Users

10. **[Medical Segmentation Example](examples/medical-segmentation/README.md)** - Complete example
    - Training pipeline
    - Testing pipeline
    - Dataset setup

11. **[Visual GUI Guide](examples/medical-segmentation/gui/VISUAL_GUI_GUIDE.md)** - Visual features
    - How to use GUI with example
    - Step-by-step tutorials
    - Troubleshooting

### For Deployment

12. **[Deployment Checklist](getting-started/DEPLOYMENT_CHECKLIST.md)** ⚡ - Step-by-step deployment
    - Complete deployment guide
    - Verification steps
    - Common issues

13. **[Deployment Quick Start](getting-started/README_DEPLOYMENT.md)** - Ready to deploy
    - Quick deployment overview
    - Success criteria
    - Critical reminders

### For Contributors

14. **[Contributing Guide](project/CONTRIBUTING.md)** - How to contribute
    - Adding nodes
    - Documentation standards
    - Pull request process

15. **[Project Status](project/PROJECT_STATUS.md)** - Project overview
    - What's included
    - Statistics
    - Future enhancements

16. **[Fixes Applied](project/FIXES_APPLIED.md)** - Issue resolution log
    - All bugs fixed
    - Solutions applied
    - Verification methods

---

## 📖 Documentation by Topic

### Getting Started
- [Quick Start (5 min)](getting-started/GETTING_STARTED.md#quick-start)
- [Installation](README.md#installation)
- [First Pipeline](getting-started/GETTING_STARTED.md#basic-example)
- [Running Examples](README.md#examples)

### Core Concepts
- [Node-Based Architecture](README.md#architecture-overview)
- [Ports and Links](getting-started/GETTING_STARTED.md#key-concepts)
- [Computational Graphs](README.md#computational-graph)
- [Graph Execution](README.md#graph-executor)

### GUI
- [Visual Node Rendering](gui/VISUAL_GUI_COMPLETE.md)
- [Keyboard Shortcuts](gui/VISUAL_GUI_QUICK_REFERENCE.md)
- [Launching GUI Methods](gui/LAUNCHING_GUI_METHODS.md)
- [Remote GUI via SSH](gui/LAUNCHING_GUI_METHODS.md#remote-gui-access-ssh-x11-forwarding)
- [Workflow Editor](README.md#gui-features)
- [Load/Save Workflows](examples/medical-segmentation/gui/GUI_GUIDE.md)

### Examples
- [Medical Segmentation Pipeline](examples/medical-segmentation/README.md)
- [Complete Workflows](examples/medical-segmentation/WORKFLOWS_COMPLETE.md)
- [GUI Usage for Examples](examples/medical-segmentation/gui/QUICKSTART_GUI.md)

### Development
- [Creating Custom Nodes](README.md#creating-custom-nodes)
- [Composite Nodes](README.md#creating-composite-nodes)
- [Contributing](project/CONTRIBUTING.md)

### Environment
- [Automatic Activation](getting-started/ENVIRONMENT_SETUP.md#automatic-activation-using-direnv)
- [Manual Activation](getting-started/ENVIRONMENT_SETUP.md#alternative-activation-methods)
- [Troubleshooting](getting-started/ENVIRONMENT_SETUP.md#troubleshooting)

---

## 🔍 Find What You Need

### I want to...

**Get started quickly**
→ [Getting Started](getting-started/GETTING_STARTED.md)

**See one-page reference**
→ [Quick Reference](getting-started/QUICK_REFERENCE.md)

**Install on a new server**
→ [Installation Guide](getting-started/INSTALLATION_GUIDE.md) | [Deployment Checklist](getting-started/DEPLOYMENT_CHECKLIST.md)

**Deploy to production**
→ [Deployment Quick Start](getting-started/README_DEPLOYMENT.md)

**Use the visual GUI**
→ [Visual GUI Complete](gui/VISUAL_GUI_COMPLETE.md)

**Launch GUI correctly** ⚡ **IMPORTANT**
→ [GUI Launching Guide](gui/GUI_LAUNCHING_GUIDE.md)

**Understand GUI launch methods**
→ [GUI Launching Methods](gui/LAUNCHING_GUI_METHODS.md)

**Fix "nodes not appearing in GUI"**
→ [GUI Launching Guide - Troubleshooting](gui/GUI_LAUNCHING_GUIDE.md#troubleshooting)

**Run GUI remotely via SSH**
→ [Remote GUI Access](gui/LAUNCHING_GUI_METHODS.md#remote-gui-access-ssh-x11-forwarding)

**Run the medical segmentation example**
→ [Medical Segmentation README](examples/medical-segmentation/README.md)

**Load workflows in GUI**
→ [GUI Quick Start](examples/medical-segmentation/gui/QUICKSTART_GUI.md)

**Understand the architecture**
→ [Main README - Architecture](README.md#architecture-overview)

**See all available nodes**
→ [Main README - Available Nodes](README.md#available-nodes)

**Create my own node**
→ [Main README - Custom Nodes](README.md#creating-custom-nodes)

**Fix environment issues**
→ [Environment Setup - Troubleshooting](getting-started/ENVIRONMENT_SETUP.md#troubleshooting)

**See what was fixed today**
→ [Fixes Applied](project/FIXES_APPLIED.md)

**Contribute to the project**
→ [Contributing Guide](project/CONTRIBUTING.md)

---

## 📋 Complete File List

### Core Documentation (docs/)

| File | Description | Best For |
|------|-------------|----------|
| **README.md** | Complete framework docs | Reference, API docs |
| **INDEX.md** | This file | Navigation |

### Getting Started (docs/getting-started/)

| File | Description | Best For |
|------|-------------|----------|
| **QUICK_REFERENCE.md** | One-page reference | Quick lookup ⚡ |
| **GETTING_STARTED.md** | Quick start guide | New users, tutorials |
| **ENVIRONMENT_SETUP.md** | Environment details | Setup, troubleshooting |
| **INSTALLATION_GUIDE.md** | Complete installation guide | New server deployment |
| **DEPLOYMENT_CHECKLIST.md** | Step-by-step deployment | Server deployment |
| **README_DEPLOYMENT.md** | Deployment quick start | Ready to deploy |
| **SERVER_SETUP.md** | Server installation | Remote server setup |
| **TROUBLESHOOTING_INSTALL.md** | Installation issues | Fixing problems |
| **NEXT_STEPS.md** | What to do next | After installation |

### Project Info (docs/project/)

| File | Description | Best For |
|------|-------------|----------|
| **CONTRIBUTING.md** | Contribution guide | Contributing code/docs |
| **PROJECT_STATUS.md** | Project overview | Contributors, overview |
| **FIXES_APPLIED.md** | All fixes applied | Issue resolution log |
| **CIRCULAR_IMPORT_FIX.md** | Import fix details | Technical reference |

### GUI Documentation (docs/gui/)

| File | Description | Best For |
|------|-------------|----------|
| **VISUAL_GUI_COMPLETE.md** | Complete visual implementation | Understanding GUI features |
| **VISUAL_GUI_QUICK_REFERENCE.md** | Quick reference card | Quick lookup ⚡ |
| **LAUNCHING_GUI_METHODS.md** | Launch methods & SSH guide | Running GUI locally/remotely |
| **GUI_LAUNCHING_GUIDE.md** | Complete launcher guide | Critical for workflows ⚡ |
| **SSH_X11_FORWARDING_GUIDE.md** | Remote GUI setup | X11 forwarding |
| **CREATING_CONNECTIONS.md** | Node connections | Building workflows |
| **EDITING_PARAMETERS.md** | Parameter editing | Node configuration |
| **PORT_TYPES_GUIDE.md** | Port types explained | Understanding ports |
| **TRAINING_VS_INFERENCE.md** | Workflow types | Training vs testing |

### Medical Segmentation Example (docs/examples/medical-segmentation/)

| File | Description | Best For |
|------|-------------|----------|
| **README.md** | Example overview | Understanding the example |
| **WORKFLOWS_COMPLETE.md** | Workflow documentation | Complete workflows |

### GUI Example Docs (docs/examples/medical-segmentation/gui/)

| File | Description | Best For |
|------|-------------|----------|
| **VISUAL_GUI_GUIDE.md** | Visual features guide | Complete GUI tutorial |
| **QUICKSTART_GUI.md** | Quick start | Fast start with GUI ⚡ |
| **GUI_GUIDE.md** | General GUI guide | GUI basics |
| **GUI_WHAT_TO_EXPECT.md** | What to expect | Understanding GUI |
| **FIXED_GUI_ISSUE.md** | GUI issues fixed | Troubleshooting |
| **VISUAL_FEATURES_IMPLEMENTED.md** | Implementation details | Technical details |
| **VISUAL_IMPLEMENTATION_SUMMARY.md** | Implementation summary | Quick overview |

---

## 🎓 Learning Path

### Beginner (Start Here!)

1. Read [Quick Reference](getting-started/QUICK_REFERENCE.md) for overview
2. Read [Getting Started](getting-started/GETTING_STARTED.md)
3. Run `python examples/simple_test.py`
4. Try the [medical segmentation example](examples/medical-segmentation/README.md)
5. Launch the GUI: `python examples/medical_segmentation_pipeline/launch_gui.py`

### Intermediate

1. Read [Main README](README.md) sections on nodes
2. Review all [Available Nodes](README.md#available-nodes)
3. Load workflows in GUI
4. Create your first custom node

### Advanced

1. Study [Composite Nodes](README.md#creating-composite-nodes)
2. Implement hierarchical pipelines
3. Extend with new architectures
4. [Contribute](project/CONTRIBUTING.md) to the framework

---

## 🔗 External Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **MONAI Documentation**: https://docs.monai.io/
- **NiBabel (NIfTI)**: https://nipy.org/nibabel/
- **PyDicom (DICOM)**: https://pydicom.github.io/

---

## 📊 Documentation Statistics

- **Total Documentation Files**: 40+
- **Documentation Categories**: 8
- **Quick Reference Pages**: 3
- **Installation Guides**: 6
- **GUI Guides**: 9
- **Example Guides**: 9
- **Project Documentation**: 5
- **Code Examples**: 30+
- **Total Documentation**: ~7,000+ lines
- **Last Updated**: February 7, 2026
- **Last Reorganized**: February 7, 2026

---

## 🆘 Need Help?

1. Check the [Quick Reference](getting-started/QUICK_REFERENCE.md)
2. Review [Troubleshooting](getting-started/ENVIRONMENT_SETUP.md#troubleshooting)
3. Check [Project Status](project/PROJECT_STATUS.md) for current status
4. Look at [examples/](../examples/) for working code
5. Read relevant documentation sections above

---

## 📝 Documentation Guidelines

When adding new documentation:

1. **Save in appropriate subfolder**:
   - Getting started docs → `getting-started/`
   - Project info → `project/`
   - GUI docs → `gui/`
   - Example docs → `examples/[example-name]/`

2. **Update this INDEX.md** with new files
3. **Link from root README.md** if appropriate
4. **Use clear section headers** for navigation
5. **Include code examples** where relevant
6. **Keep docs up to date** with code changes

---

**Happy coding!** 🚀

**Start with [Quick Reference](getting-started/QUICK_REFERENCE.md) or [Getting Started](getting-started/GETTING_STARTED.md)!**
