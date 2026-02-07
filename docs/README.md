# Medical Imaging Framework

A comprehensive PyTorch-based node-based deep learning framework for 2D/3D medical image segmentation and classification.

## Features

### Core Architecture
- **Node-Based System**: Modular, composable nodes for building complex pipelines
- **Hierarchical Composition**: Nodes can contain sub-nodes (composite nodes)
- **Visual Workflow Editor**: PyQt5-based GUI for visual pipeline design
- **Graph Execution**: Topological sorting and efficient execution
- **Serialization**: Save and load workflows as JSON

### Medical Imaging Support
- **Formats**: NIfTI (.nii, .nii.gz), DICOM (.dcm), standard image formats
- **Dimensions**: Both 2D and 3D medical image processing
- **Data Pipeline**: Efficient loading, batching, and augmentation

### Network Architectures
- **U-Net**: 2D/3D U-Net and Attention U-Net for segmentation
- **ResNet**: 2D/3D ResNet encoders and decoders with skip connections
- **Transformers**: Vision Transformer (ViT) and transformer encoders/decoders
- **Modular Design**: Reusable building blocks for custom architectures

### Training & Inference
- **Training Pipeline**: Complete training loop with loss functions and optimizers
- **Multiple Loss Functions**: Cross-entropy, Dice loss, MSE, BCE
- **Optimizers**: Adam, SGD, AdamW
- **Metrics**: Accuracy, precision, recall, F1-score, Dice coefficient
- **GPU Support**: Automatic CUDA detection and utilization

### Data Augmentation
- Random flipping (horizontal/vertical)
- Random rotation
- Intensity normalization (z-score, min-max)
- Extensible augmentation system

### Visualization
- Medical image viewer with overlay support
- Training metrics plotting
- Prediction visualization
- Debug printing utilities

## Installation

```bash
# Clone the repository
cd medical_imaging_framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install framework in development mode
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- nibabel (for NIfTI support)
- pydicom (for DICOM support)
- matplotlib (for visualization)
- PyQt5 (for GUI)
- numpy, PIL

## Quick Start

### 1. Using the Python API

```python
from medical_imaging_framework import (
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor
)
import medical_imaging_framework.nodes

# Create a computational graph
graph = ComputationalGraph("Segmentation Pipeline")

# Add nodes
data_loader = NodeRegistry.create_node(
    'DataLoaderNode',
    'train_loader',
    config={'batch_size': 4, 'shuffle': True}
)
graph.add_node(data_loader)

unet = NodeRegistry.create_node(
    'UNet2DNode',
    'unet',
    config={
        'in_channels': 1,
        'out_channels': 2,
        'base_channels': 64
    }
)
graph.add_node(unet)

trainer = NodeRegistry.create_node(
    'TrainerNode',
    'trainer',
    config={'num_epochs': 10, 'learning_rate': 0.001}
)
graph.add_node(trainer)

# Connect nodes
graph.connect('train_loader', 'batch', 'trainer', 'dataloader')
graph.connect('unet', 'output', 'trainer', 'model')

# Execute
executor = GraphExecutor(graph)
result = executor.execute()

print(f"Execution status: {result.status.value}")
```

### 2. Using the GUI

```bash
# Launch the GUI editor
python -m medical_imaging_framework.gui.editor
```

Or from Python:

```python
from medical_imaging_framework.gui import main
main()
```

### 3. Running Examples

```bash
# Run segmentation workflow example
python examples/segmentation_workflow.py
```

## Project Structure

```
medical_imaging_framework/
├── core/                   # Core framework
│   ├── node.py            # Base node classes
│   ├── registry.py        # Node registry
│   ├── graph.py           # Computational graph
│   └── executor.py        # Graph executor
├── nodes/                 # Node implementations
│   ├── data/              # Data loading & augmentation
│   │   ├── loader.py      # DataLoader, ImagePathLoader
│   │   └── augmentation.py # RandomFlip, Rotation, Normalize
│   ├── networks/          # Network architectures
│   │   ├── unet.py        # U-Net variants
│   │   ├── resnet_blocks.py # ResNet components
│   │   └── transformers.py  # Transformer blocks
│   ├── training/          # Training nodes
│   │   └── trainer.py     # Trainer, Optimizer, Loss
│   ├── inference/         # Inference nodes
│   │   └── predictor.py   # Predictor, MetricsCalculator
│   └── visualization/     # Visualization nodes
│       └── viewer.py      # ImageViewer, MetricsPlotter
├── gui/                   # GUI components
│   └── editor.py          # PyQt5 workflow editor
├── utils/                 # Utility functions
└── examples/              # Example workflows
    └── segmentation_workflow.py
```

## Available Nodes

### Data Nodes
- **ImagePathLoader**: Load image file paths from directory
- **DataLoader**: Create batches from medical images
- **BatchExtractor**: Extract single batch for testing
- **RandomFlip**: Random horizontal/vertical flipping
- **RandomRotation**: Random rotation augmentation
- **Normalize**: Intensity normalization

### Network Nodes
- **UNet2D/3D**: U-Net for segmentation
- **AttentionUNet2D**: U-Net with attention gates
- **ResNetEncoder2D/3D**: ResNet encoder with skip connections
- **ResNetDecoder2D**: ResNet decoder
- **TransformerEncoder**: Transformer encoder blocks
- **VisionTransformer2D**: Vision Transformer for classification

### Training Nodes
- **Trainer**: Training loop with backpropagation
- **LossFunction**: Various loss functions
- **Optimizer**: Optimizer configuration

### Inference Nodes
- **Predictor**: Single batch prediction
- **BatchPredictor**: Batch inference over dataset
- **MetricsCalculator**: Calculate accuracy, precision, recall, F1

### Visualization Nodes
- **ImageViewer**: Display images and segmentation overlays
- **MetricsPlotter**: Plot training curves
- **Print**: Debug printing

## Creating Custom Nodes

```python
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register('custom', 'MyCustomNode',
                      description='My custom processing node')
class MyCustomNode(BaseNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def execute(self) -> bool:
        try:
            x = self.get_input_value('input')
            # Your processing logic here
            result = x * 2
            self.set_output_value('output', result)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def get_field_definitions(self):
        return {
            'scale': {'type': 'text', 'label': 'Scale Factor', 'default': '2.0'}
        }
```

## Creating Composite Nodes

```python
from medical_imaging_framework.core import CompositeNode

class SegmentationPipeline(CompositeNode):
    def __init__(self, name: str, config=None):
        super().__init__(name, config)

        # Add sub-nodes
        encoder = NodeRegistry.create_node('ResNetEncoder2DNode', 'encoder')
        decoder = NodeRegistry.create_node('ResNetDecoder2DNode', 'decoder')

        self.add_sub_node(encoder)
        self.add_sub_node(decoder)

        # Connect internally
        self.connect_internal('encoder', 'features', 'decoder', 'features')
        self.connect_internal('encoder', 'skip_connections', 'decoder', 'skip_connections')

        # Expose ports
        self.expose_input('encoder', 'input', 'image')
        self.expose_output('decoder', 'output', 'segmentation')
```

## Workflow Serialization

```python
# Save workflow
graph.save_to_file('my_workflow.json')

# Load workflow
graph.load_from_file('my_workflow.json')
```

## Advanced Usage

### Interactive Execution

```python
from medical_imaging_framework.core import InteractiveExecutor

executor = InteractiveExecutor(graph)
result = executor.start()

# Step through execution
has_more, node_name = executor.step()
while has_more:
    print(f"Executed: {node_name}")
    has_more, node_name = executor.step()

final_result = executor.get_result()
```

### Progress Monitoring

```python
def progress_callback(node_name, progress):
    print(f"Executing {node_name}: {progress:.1%}")

executor = GraphExecutor(graph, progress_callback=progress_callback)
result = executor.execute()
```

### Validation

```python
is_valid, errors = graph.validate()
if not is_valid:
    print("Errors:")
    for error in errors:
        print(f"  - {error}")
```

## Examples

See the `examples/` directory for complete workflows:
- `segmentation_workflow.py`: Full segmentation pipeline
- More examples coming soon!

## GUI Features

The PyQt5-based GUI provides:
- Visual node library browser
- Workflow validation
- Execution monitoring
- Save/load workflows
- Dark theme UI

**Note**: The current GUI is a prototype. Full visual node editing with drag-and-drop connections is planned for future releases. For now, use the Python API to create complex workflows.

## Future Enhancements

- [ ] Full drag-and-drop visual editor
- [ ] React-based web GUI
- [ ] More network architectures (YOLO, Mask R-CNN)
- [ ] Advanced augmentation techniques
- [ ] Multi-GPU training support
- [ ] TensorBoard integration
- [ ] ONNX export
- [ ] Model zoo with pre-trained weights

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{medical_imaging_framework,
  title = {Medical Imaging Framework},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/medical_imaging_framework}
}
```

## Acknowledgments

- Inspired by node-based visual programming paradigms
- Built with PyTorch, PyQt5, and the Python scientific stack
- Medical imaging support via nibabel and pydicom

## Contact

For questions and support, please open an issue on GitHub.
