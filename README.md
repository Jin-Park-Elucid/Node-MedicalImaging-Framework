# Medical Imaging Framework

A comprehensive PyTorch-based node-based deep learning framework for 2D/3D medical image segmentation and classification.

## 🚀 Quick Start

```bash
# The virtual environment activates automatically when you enter this directory!
cd medical_imaging_framework

# Test the framework
python examples/simple_test.py

# Launch GUI editor
python -m medical_imaging_framework.gui.editor
```

## 📚 Documentation

**All documentation is now organized in the `docs/` folder!**

### 🎯 Quick Start
- **[📖 Documentation Hub](docs/INDEX.md)** - Complete navigation ⭐
- **[⚡ Quick Reference](docs/getting-started/QUICK_REFERENCE.md)** - One-page cheat sheet
- **[🚀 Getting Started](docs/getting-started/GETTING_STARTED.md)** - 5-minute quick start
- **[🎨 Visual GUI Guide](docs/gui/VISUAL_GUI_COMPLETE.md)** - Complete GUI documentation

### 📂 Documentation Structure
```
docs/
├── getting-started/      # Quick start guides (3 files)
├── project/              # Project info & contributing (2 files)
├── gui/                  # Visual GUI documentation (2 files)
├── examples/             # Example documentation
│   └── medical-segmentation/  # Complete example (9 files)
├── README.md             # Full framework documentation
└── INDEX.md              # Navigation hub
```

**Browse all docs:** [docs/INDEX.md](docs/INDEX.md)

## ✨ Features

- **Node-Based Architecture**: Modular, composable pipeline design
- **23 Built-in Nodes**: Data loading, networks, training, inference, visualization
- **Medical Imaging Support**: NIfTI, DICOM, 2D/3D processing
- **Network Architectures**: U-Net, ResNet, Transformers
- **PyTorch Integration**: Full training and inference pipelines
- **GUI Workflow Editor**: Visual pipeline design with PyQt5
- **Automatic Environment**: Virtual environment activates on directory entry

## 🎯 Key Components

| Component | Description |
|-----------|-------------|
| **Core Framework** | BaseNode, CompositeNode, Graph, Executor |
| **Data Nodes** | Loaders, augmentation, batching |
| **Network Nodes** | U-Net 2D/3D, ResNet, Transformers |
| **Training Nodes** | Trainer, optimizers, loss functions |
| **Inference Nodes** | Prediction, metrics calculation |
| **Visualization** | Image viewer, metrics plotting, GUI |

## 📦 Installation

The framework is already installed and ready to use!

**Environment activates automatically** via direnv when you enter this directory.

Alternative activation:
```bash
source activate.sh              # Manual activation script
source venv/bin/activate        # Standard Python venv
```

## 🔬 Examples

```bash
# Run quick test (23 nodes registered)
python examples/simple_test.py

# Run segmentation workflow example
python examples/segmentation_workflow.py

# Launch GUI editor
python -m medical_imaging_framework.gui.editor
```

## 📁 Project Structure

```
medical_imaging_framework/
├── medical_imaging_framework/     # Main package
│   ├── core/                     # Core framework
│   ├── nodes/                    # 23 implemented nodes
│   ├── gui/                      # PyQt5 workflow editor
│   └── utils/                    # Utilities
├── docs/                         # 📚 All documentation
│   ├── README.md                 # Complete documentation
│   ├── GETTING_STARTED.md        # Quick start guide
│   ├── ENVIRONMENT_SETUP.md      # Environment details
│   └── PROJECT_STATUS.md         # Project overview
├── examples/                      # Example workflows
├── tests/                        # Unit tests
├── venv/                         # Virtual environment
├── .envrc                        # Auto-activation config
└── activate.sh                   # Manual activation script
```

## 🛠️ Creating Custom Nodes

```python
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register('custom', 'MyNode', description='My custom node')
class MyNode(BaseNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def execute(self) -> bool:
        x = self.get_input_value('input')
        result = your_processing(x)
        self.set_output_value('output', result)
        return True
```

## 📊 Statistics

- **23 Registered Nodes** across 5 categories
- **8 Network Architectures** (U-Net, ResNet, Transformers)
- **50+ Dependencies** installed (PyTorch 2.10.0, MONAI, etc.)
- **5,000+ Lines** of code
- **Complete Pipeline** from data loading to visualization

## 🤝 Contributing

Contributions are welcome! See [docs/README.md](docs/README.md) for detailed information.

## 📄 License

MIT License

## 🔗 Links

- [Full Documentation](docs/README.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Environment Setup](docs/ENVIRONMENT_SETUP.md)
- [Project Status](docs/PROJECT_STATUS.md)

---

**Ready to use!** Just enter the directory and the environment activates automatically. 🎉

For detailed documentation, see the **[`docs/`](docs/)** folder.
