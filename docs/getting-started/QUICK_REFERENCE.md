# Medical Imaging Framework - Quick Reference

**One-page reference for common tasks**

## 📁 Project Locations

```
medical_imaging_framework/
├── README.md                      # Overview
├── docs/                          # 📚 ALL DOCUMENTATION
├── medical_imaging_framework/     # Source code
├── examples/                      # Examples
├── venv/                          # Virtual env (auto-activates)
└── activate.sh                    # Manual activation
```

## 🚀 Quick Commands

```bash
# Environment activates automatically when entering directory!
cd medical_imaging_framework

# Test framework
python examples/simple_test.py

# Launch GUI
python -m medical_imaging_framework.gui.editor

# Run example
python examples/segmentation_workflow.py

# Manual activation (if needed)
source activate.sh
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [INDEX.md](INDEX.md) | Documentation navigation |
| [README.md](README.md) | Complete documentation |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Quick start guide |
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | Environment help |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Project overview |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## 🔧 Common Tasks

### Create a Node

```python
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register('category', 'NodeName')
class MyNode(BaseNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)
    
    def execute(self) -> bool:
        x = self.get_input_value('input')
        result = process(x)
        self.set_output_value('output', result)
        return True
```

### Create a Pipeline

```python
from medical_imaging_framework import (
    NodeRegistry, ComputationalGraph, GraphExecutor
)
import medical_imaging_framework.nodes

graph = ComputationalGraph("My Pipeline")

# Add nodes
node1 = NodeRegistry.create_node('DataLoader', 'loader')
node2 = NodeRegistry.create_node('UNet2D', 'model')

graph.add_node(node1)
graph.add_node(node2)

# Connect
graph.connect('loader', 'batch', 'model', 'input')

# Execute
executor = GraphExecutor(graph)
result = executor.execute()
```

### Save/Load Workflow

```python
# Save
graph.save_to_file('my_workflow.json')

# Load
graph.load_from_file('my_workflow.json')
```

## 📊 Available Nodes (23 Total)

**Data (6)**: DataLoader, ImagePathLoader, BatchExtractor, RandomFlip, RandomRotation, Normalize

**Networks (8)**: UNet2D, UNet3D, AttentionUNet2D, ResNetEncoder2D, ResNetEncoder3D, ResNetDecoder2D, TransformerEncoder, VisionTransformer2D

**Training (3)**: Trainer, LossFunction, Optimizer

**Inference (3)**: Predictor, BatchPredictor, MetricsCalculator

**Visualization (3)**: ImageViewer, MetricsPlotter, Print

## 🔍 Troubleshooting

**Environment not activating?**
```bash
direnv allow .
source activate.sh
```

**Import error?**
```bash
source venv/bin/activate
pip install -e .
```

**Need help?**
```bash
cat docs/INDEX.md           # Documentation index
cat docs/GETTING_STARTED.md # Quick start
```

## 🎯 Next Steps

1. Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. Run `python examples/simple_test.py`
3. Explore [docs/README.md](README.md) for complete API
4. Build your first pipeline!

---

**For full documentation, see [docs/INDEX.md](INDEX.md)**
