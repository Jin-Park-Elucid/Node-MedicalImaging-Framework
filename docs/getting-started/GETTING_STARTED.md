# Getting Started with Medical Imaging Framework

## Quick Test

Run the quick test to verify the framework is working:

```bash
python examples/simple_test.py
```

This will test:
- Node registry (23 built-in nodes)
- Node creation
- Graph building and validation
- Workflow serialization

## Architecture Overview

The framework follows a node-based architecture inspired by visual programming:

```
┌─────────────────────────────────────────────────┐
│          Computational Graph                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │  Data   │───▶│ Network │───▶│ Trainer │    │
│  │ Loader  │    │ (U-Net) │    │         │    │
│  └─────────┘    └─────────┘    └─────────┘    │
│                                                 │
│  Nodes connected via typed ports                │
│  Execution in topological order                 │
└─────────────────────────────────────────────────┘
```

### Key Concepts

1. **Nodes**: Functional components (data loading, networks, training, etc.)
2. **Ports**: Input/output connection points with data types
3. **Links**: Connections between ports that transfer data
4. **Graph**: Collection of nodes and their connections
5. **Executor**: Runs the graph in correct order

## Basic Example

```python
from medical_imaging_framework import (
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor
)
import medical_imaging_framework.nodes

# Create graph
graph = ComputationalGraph("My Pipeline")

# Create nodes
loader = NodeRegistry.create_node(
    'DataLoader',
    'data',
    config={'batch_size': 4}
)

model = NodeRegistry.create_node(
    'UNet2D',
    'model',
    config={
        'in_channels': 1,
        'out_channels': 2
    }
)

# Add to graph
graph.add_node(loader)
graph.add_node(model)

# Connect nodes (if applicable)
# graph.connect('source_node', 'output_port', 'target_node', 'input_port')

# Validate
is_valid, errors = graph.validate()
if not is_valid:
    for error in errors:
        print(f"Error: {error}")

# Execute
executor = GraphExecutor(graph)
result = executor.execute()

print(f"Status: {result.status.value}")
```

## Available Node Types

### Data Nodes
- **ImagePathLoader**: Scan directory for medical images
- **DataLoader**: Create PyTorch DataLoader with batching
- **BatchExtractor**: Extract single batch for testing
- **RandomFlip**: Random flipping augmentation
- **RandomRotation**: Random rotation augmentation
- **Normalize**: Intensity normalization

### Network Nodes
- **UNet2D/3D**: U-Net for segmentation
- **AttentionUNet2D**: U-Net with attention gates
- **ResNetEncoder2D/3D**: ResNet encoder with skip connections
- **ResNetDecoder2D**: ResNet decoder
- **TransformerEncoder**: Transformer blocks
- **VisionTransformer2D**: Vision Transformer

### Training Nodes
- **Trainer**: Complete training loop
- **LossFunction**: Loss function definition
- **Optimizer**: Optimizer configuration

### Inference Nodes
- **Predictor**: Single batch inference
- **BatchPredictor**: Batch inference
- **MetricsCalculator**: Calculate metrics

### Visualization Nodes
- **ImageViewer**: Display images and segmentations
- **MetricsPlotter**: Plot training curves
- **Print**: Debug printing

## Creating Custom Nodes

```python
from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

@NodeRegistry.register('custom', 'MyNode', description='My custom node')
class MyNode(BaseNode):
    def _setup_ports(self):
        self.add_input('input', DataType.TENSOR)
        self.add_output('output', DataType.TENSOR)

    def execute(self) -> bool:
        try:
            x = self.get_input_value('input')
            # Your processing
            result = your_function(x)
            self.set_output_value('output', result)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
```

## Hierarchical Nodes

Create composite nodes that contain sub-graphs:

```python
from medical_imaging_framework.core import CompositeNode

class MyPipeline(CompositeNode):
    def __init__(self, name: str, config=None):
        super().__init__(name, config)

        # Add sub-nodes
        encoder = NodeRegistry.create_node('ResNetEncoder2D', 'enc')
        decoder = NodeRegistry.create_node('ResNetDecoder2D', 'dec')

        self.add_sub_node(encoder)
        self.add_sub_node(decoder)

        # Connect internally
        self.connect_internal('enc', 'features', 'dec', 'features')

        # Expose ports to outside
        self.expose_input('enc', 'input', 'image')
        self.expose_output('dec', 'output', 'result')
```

## Workflow Serialization

```python
# Save
graph.save_to_file('my_workflow.json')

# Load
graph.load_from_file('my_workflow.json')
```

## GUI Usage

Launch the visual workflow editor:

```bash
python -m medical_imaging_framework.gui.editor
```

Features:
- Browse available nodes by category
- Add nodes to workflow
- Validate and execute workflows
- Save/load workflows
- View execution results

**Note**: The current GUI is a prototype. Visual node editing is planned for future releases.

## Examples

The `examples/` directory contains:

1. **simple_test.py**: Quick framework verification
2. **segmentation_workflow.py**: Complete segmentation pipeline example

## Next Steps

1. ✅ Verify installation with `simple_test.py`
2. 📚 Read the full [README.md](README.md)
3. 🔬 Explore example workflows
4. 🎨 Try the GUI editor
5. 🛠️ Create your own custom nodes
6. 🚀 Build your medical imaging pipeline!

## Troubleshooting

### Missing dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### Node registration issues
Make sure to import nodes before using them:
```python
import medical_imaging_framework.nodes
```

### Validation errors
Check that:
- All required inputs are connected or have values
- Port data types match between connections
- No cycles in the graph

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review example workflows

## License

MIT License - see LICENSE file for details.
