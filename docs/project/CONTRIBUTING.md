# Contributing to Medical Imaging Framework

Thank you for your interest in contributing to the Medical Imaging Framework!

## 📝 Documentation Contributions

All documentation should be placed in the **`docs/`** folder.

### Adding New Documentation

1. **Create your document in the `docs/` folder:**
   ```bash
   # Create new documentation file
   nano docs/YOUR_NEW_DOC.md
   ```

2. **Update `docs/INDEX.md`:**
   - Add your document to the file list
   - Add navigation links
   - Update the table of contents

3. **Link from root `README.md` if appropriate:**
   - Add link in the Documentation section
   - Use relative paths: `[Your Doc](docs/YOUR_NEW_DOC.md)`

4. **Follow documentation standards:**
   - Use clear, descriptive headers
   - Include code examples
   - Add table of contents for long documents
   - Cross-reference related documents

### Documentation Structure

```
docs/
├── INDEX.md                    # Navigation hub (update this!)
├── README.md                   # Complete framework docs
├── GETTING_STARTED.md          # Quick start guide
├── ENVIRONMENT_SETUP.md        # Environment setup
├── PROJECT_STATUS.md           # Project overview
├── CONTRIBUTING.md             # This file
└── YOUR_NEW_DOC.md            # Your new documentation
```

### Documentation Types

| Type | Purpose | Location |
|------|---------|----------|
| **User Guides** | How to use the framework | `docs/` |
| **API Reference** | Class and method documentation | `docs/API_REFERENCE.md` |
| **Tutorials** | Step-by-step guides | `docs/tutorials/` |
| **Architecture** | Design decisions | `docs/ARCHITECTURE.md` |
| **Examples** | Code examples | `examples/` + `docs/` |

## 🔧 Code Contributions

### Adding New Nodes

1. **Create node class:**
   ```python
   from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType

   @NodeRegistry.register('category', 'NodeName',
                         description='Brief description')
   class YourNode(BaseNode):
       def _setup_ports(self):
           self.add_input('input', DataType.TENSOR)
           self.add_output('output', DataType.TENSOR)

       def execute(self) -> bool:
           # Your implementation
           return True
   ```

2. **Add to appropriate module:**
   - Data nodes: `medical_imaging_framework/nodes/data/`
   - Network nodes: `medical_imaging_framework/nodes/networks/`
   - Training nodes: `medical_imaging_framework/nodes/training/`
   - etc.

3. **Update documentation:**
   - Add to `docs/README.md` - Available Nodes section
   - Create example in `examples/`
   - Add to `docs/INDEX.md`

4. **Add tests:**
   - Create test in `tests/`
   - Verify node registration
   - Test execute() method

### Adding Network Architectures

1. **Create architecture module:**
   ```python
   # medical_imaging_framework/nodes/networks/your_network.py

   import torch.nn as nn
   from ...core import PyTorchModuleNode, NodeRegistry

   class YourNetwork(nn.Module):
       def __init__(self, ...):
           super().__init__()
           # Build network

       def forward(self, x):
           # Forward pass
           return output

   @NodeRegistry.register('networks', 'YourNetworkNode')
   class YourNetworkNode(PyTorchModuleNode):
       def _setup_ports(self):
           self.add_input('input', DataType.TENSOR)
           self.add_output('output', DataType.TENSOR)

       def build_module(self) -> nn.Module:
           return YourNetwork(...)

       def execute(self) -> bool:
           # Implementation
           return True
   ```

2. **Import in `__init__.py`:**
   ```python
   # medical_imaging_framework/nodes/networks/__init__.py
   from . import your_network
   ```

3. **Document and test:**
   - Add to documentation
   - Create usage example
   - Add unit tests

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_your_feature.py

# Run with coverage
python -m pytest --cov=medical_imaging_framework tests/
```

### Writing Tests

```python
# tests/test_your_node.py

import pytest
from medical_imaging_framework import NodeRegistry

def test_your_node_creation():
    node = NodeRegistry.create_node('YourNode', 'test_instance')
    assert node is not None
    assert 'input' in node.inputs
    assert 'output' in node.outputs

def test_your_node_execution():
    node = NodeRegistry.create_node('YourNode', 'test')
    # Set up inputs
    node.inputs['input'].set_value(test_data)
    # Execute
    success = node.execute()
    assert success
    # Check outputs
    result = node.outputs['output'].get_value()
    assert result is not None
```

## 📋 Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small

### Example

```python
from typing import Optional, List
import torch

def process_image(
    image: torch.Tensor,
    normalize: bool = True,
    mean: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Process medical image with normalization.

    Args:
        image: Input image tensor (B, C, H, W)
        normalize: Whether to normalize intensities
        mean: Mean values for normalization

    Returns:
        Processed image tensor

    Example:
        >>> img = torch.randn(1, 1, 256, 256)
        >>> processed = process_image(img, normalize=True)
    """
    if normalize:
        # Normalization logic
        pass
    return image
```

### Commit Messages

Use conventional commits:

```
feat: add new U-Net 3D variant
fix: correct batch size handling in data loader
docs: update API reference for new nodes
test: add tests for transformer encoder
refactor: simplify graph validation logic
```

## 🔀 Pull Request Process

1. **Fork and clone:**
   ```bash
   git clone https://github.com/yourusername/medical_imaging_framework.git
   cd medical_imaging_framework
   ```

2. **Create branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes:**
   - Write code
   - Add tests
   - Update documentation
   - Follow coding standards

4. **Test locally:**
   ```bash
   python -m pytest tests/
   python examples/simple_test.py
   ```

5. **Commit:**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

6. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

7. **PR checklist:**
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] Commit messages are clear
   - [ ] No breaking changes (or documented)

## 📊 Documentation Standards

### Markdown Style

- Use ATX-style headers (`#`, `##`, `###`)
- Include code blocks with language tags
- Add tables for structured information
- Use lists for sequential steps
- Include examples

### Code Examples

Always include working examples:

````markdown
```python
# Example: Creating a custom node
from medical_imaging_framework.core import BaseNode

class MyNode(BaseNode):
    def _setup_ports(self):
        # Port setup
        pass

    def execute(self):
        # Execution logic
        return True
```
````

### Cross-References

Link related documents:

```markdown
See [Getting Started Guide](GETTING_STARTED.md) for basic usage.
For environment setup, refer to [Environment Setup](ENVIRONMENT_SETUP.md).
```

## 🌟 Areas for Contribution

We welcome contributions in these areas:

### High Priority

- [ ] Additional network architectures (YOLO, Mask R-CNN)
- [ ] More data augmentation techniques
- [ ] Enhanced GUI with drag-and-drop
- [ ] Additional loss functions
- [ ] Pre-trained model weights

### Medium Priority

- [ ] Multi-GPU training support
- [ ] TensorBoard integration improvements
- [ ] ONNX export functionality
- [ ] Cloud deployment support
- [ ] Docker containerization

### Documentation

- [ ] More tutorial examples
- [ ] Video tutorials
- [ ] API reference improvements
- [ ] Architecture diagrams
- [ ] Use case documentation

## 💬 Communication

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Pull Requests**: Submit PRs for code contributions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉
