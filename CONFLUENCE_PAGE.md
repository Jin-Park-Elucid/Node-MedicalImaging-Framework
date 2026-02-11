# Medical Imaging Framework - Confluence Page

Copy and paste this content into your Confluence page editor.

---

# Medical Imaging Framework

## Overview

The **Medical Imaging Framework** is a comprehensive PyTorch-based node-based deep learning framework designed for 2D/3D medical image segmentation and classification. It provides a visual, modular approach to building and deploying medical imaging pipelines through an intuitive GUI workflow editor.

{panel:title=🎯 Key Value Proposition|borderStyle=solid|borderColor=#ccc|titleBGColor=#4A90E2|bgColor=#F5F8FA}
Transform complex medical imaging workflows into visual, reusable pipelines without writing code. Design, test, and deploy deep learning models through an intuitive node-based interface.
{panel}

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Node-Based Architecture** | Modular, composable pipeline design with drag-and-drop interface |
| **23 Built-in Nodes** | Complete toolkit covering data loading, networks, training, inference, and visualization |
| **Medical Imaging Support** | Native support for NIfTI and DICOM formats with 2D/3D processing |
| **Network Architectures** | 8 pre-built architectures including U-Net 2D/3D, ResNet, and Transformers |
| **PyTorch Integration** | Full training and inference pipelines built on PyTorch 2.10.0 |
| **Visual GUI Editor** | PyQt5-based workflow editor for visual pipeline design |
| **Production Ready** | Complete framework with 5,000+ lines of tested code |

---

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Workflow Editor                       │
│              (Visual Pipeline Design)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Node-Based Pipeline                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │  Data    │ → │ Network  │ → │ Training │ → │ Results │ │
│  │  Loader  │   │  Node    │   │   Node   │   │  Node   │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PyTorch Deep Learning Backend                   │
│            (U-Net, ResNet, Transformers, etc.)              │
└─────────────────────────────────────────────────────────────┘
```

### Node Categories

{expand:title=📦 Data Nodes (5 nodes)}
- **Loaders**: NIfTI, DICOM, custom medical datasets
- **Augmentation**: Spatial transforms, intensity adjustments
- **Batching**: Dynamic batch creation and management
{expand}

{expand:title=🧠 Network Nodes (8 architectures)}
- **U-Net**: 2D and 3D variants for segmentation
- **ResNet**: Classification and feature extraction
- **Transformers**: Attention-based architectures
- **Custom Networks**: Extensible architecture support
{expand}

{expand:title=🎓 Training Nodes (6 nodes)}
- **Trainer**: Complete training loop with validation
- **Optimizers**: Adam, SGD, AdamW
- **Loss Functions**: Dice, Cross-Entropy, Combined losses
- **Learning Rate Schedulers**: Step, Cosine, ReduceOnPlateau
{expand}

{expand:title=🔍 Inference & Visualization (4 nodes)}
- **Prediction**: Single and batch inference
- **Metrics**: Dice score, IoU, accuracy
- **Visualization**: Image overlays, metric plots
- **Export**: Model and results export
{expand}

---

## 🚀 Quick Start

### Installation

{code:bash}
# Clone the repository
git clone https://github.com/Jin-Park-Elucid/Node-MedicalImaging-Framework.git
cd Node-MedicalImaging-Framework

# Environment activates automatically via direnv
# Or manually activate:
source venv/bin/activate
{code}

### Launch GUI Editor

{code:bash}
# Start the visual workflow editor
python -m medical_imaging_framework.gui.editor

# Or use the launcher script
python examples/launch_gui.py
{code}

### Create Your First Pipeline

{info}
**In 3 minutes, you can:**
1. Launch the GUI editor
2. Drag and drop nodes to create a workflow
3. Connect nodes to define data flow
4. Configure parameters through the interface
5. Execute the pipeline and view results
{info}

---

## 💼 Use Cases

### ✅ Supported Applications

| Use Case | Description | Example Workflow |
|----------|-------------|------------------|
| **Medical Image Segmentation** | Segment organs, tumors, vessels from CT/MRI | Load → Preprocess → U-Net 3D → Visualize |
| **Classification** | Classify medical images by disease/condition | Load → Augment → ResNet → Metrics |
| **Multi-Modal Fusion** | Combine CT, MRI, PET data for analysis | Multi-Load → Fusion → Network → Results |
| **Research Prototyping** | Rapidly test different architectures | Visual Design → Quick Iteration → Compare |
| **Model Training** | Full training pipeline with checkpointing | Data → Network → Train → Validate → Save |
| **Inference Deployment** | Load trained models for prediction | Load Model → Predict → Export Results |

### 🎯 Target Users

- **Researchers**: Rapid prototyping of medical imaging algorithms
- **Data Scientists**: Build and test deep learning pipelines visually
- **Medical Imaging Teams**: Standardize processing workflows
- **ML Engineers**: Deploy and monitor medical imaging models

---

## 🛠️ Technical Specifications

### Technology Stack

{code:yaml}
Framework:
  - Core: Python 3.12
  - Deep Learning: PyTorch 2.10.0
  - Medical Imaging: MONAI, NiBabel, PyDICOM
  - GUI: PyQt5
  - Scientific: NumPy, SciPy, scikit-learn

Dependencies:
  - 50+ production-grade packages
  - Fully containerizable
  - GPU-accelerated (CUDA support)

Architecture:
  - Lines of Code: 5,000+
  - Node Count: 23 built-in nodes
  - Network Architectures: 8 pre-built
  - Test Coverage: Unit tests included
{code}

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux, macOS, Windows | Linux (Ubuntu 20.04+) |
| **Python** | 3.8+ | 3.12 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Optional | NVIDIA GPU with CUDA 11.8+ |
| **Storage** | 5 GB | 20 GB+ (for datasets) |

---

## 📊 Project Statistics

{panel:borderStyle=solid|borderColor=#4A90E2}
|| Metric || Count ||
| Registered Nodes | 23 |
| Network Architectures | 8 |
| Installed Dependencies | 50+ |
| Lines of Code | 5,000+ |
| Documentation Pages | 40+ |
| Example Workflows | Multiple |
{panel}

---

## 📚 Documentation

### Quick Links

- **[Complete Documentation](docs/README.md)** - Full framework documentation
- **[Getting Started Guide](docs/getting-started/GETTING_STARTED.md)** - 5-minute quick start
- **[Visual GUI Guide](docs/gui/VISUAL_GUI_COMPLETE.md)** - Complete GUI documentation
- **[Quick Reference](docs/getting-started/QUICK_REFERENCE.md)** - One-page cheat sheet

### Documentation Structure

{code}
docs/
├── getting-started/      # Quick start guides
├── gui/                  # Visual GUI documentation
├── examples/             # Example workflows
│   └── medical-segmentation/  # Complete segmentation example
├── project/              # Project info & contributing
└── INDEX.md              # Navigation hub
{code}

---

## 🔧 Creating Custom Nodes

The framework is fully extensible. Create custom nodes for your specific use cases:

{code:python}
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register('custom', 'MyCustomNode',
                       description='My custom processing node')
class MyCustomNode(BaseNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def execute(self) -> bool:
        # Your custom processing logic
        input_data = self.get_input_value('input')
        result = your_processing_function(input_data)
        self.set_output_value('output', result)
        return True
{code}

{tip}
Custom nodes automatically appear in the GUI and can be connected like built-in nodes!
{tip}

---

## 🎬 Getting Started Examples

### Example 1: Simple Segmentation Pipeline

{code:bash}
# Run the example segmentation workflow
python examples/segmentation_workflow.py
{code}

### Example 2: Visual Pipeline Design

{code:bash}
# Launch GUI and load example workflow
python examples/launch_gui.py
# File → Open → Select avte_2d_training_workflow.json
{code}

### Example 3: AVTE Dataset Processing

{code:bash}
# Process AVTE medical imaging dataset
cd examples/AVTE
python examples/launch_gui.py
{code}

---

## 🤝 Contributing & Support

### Repository

{panel:bgColor=#F0F4F8}
**GitHub Repository**: [Node-MedicalImaging-Framework](https://github.com/Jin-Park-Elucid/Node-MedicalImaging-Framework)

**License**: MIT License
{panel}

### Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/project/CONTRIBUTING.md) for guidelines.

### Getting Help

- Review the [documentation](docs/INDEX.md)
- Check [troubleshooting guides](docs/getting-started/TROUBLESHOOTING_INSTALL.md)
- Open an issue on GitHub

---

## 🎯 Next Steps

{info:title=Ready to get started?}
1. **Try it out**: `python -m medical_imaging_framework.gui.editor`
2. **Read the docs**: Start with [Getting Started](docs/getting-started/GETTING_STARTED.md)
3. **Run examples**: Explore the `examples/` directory
4. **Build your pipeline**: Create custom workflows for your datasets
5. **Contribute**: Share your custom nodes with the community
{info}

---

## 📞 Contact

For questions, feedback, or collaboration opportunities, please reach out through GitHub issues or the repository maintainer.

---

{panel:title=🚀 Project Status|borderStyle=solid|borderColor=#28A745|bgColor=#E8F5E9}
**Status**: Production Ready ✅

The framework is fully functional with comprehensive documentation and examples. Ready for research and development use.
{panel}

---

*Last Updated: February 2026*
*Version: 1.0*
