"""
Medical Imaging Framework - Node-based deep learning for medical imaging.
"""

__version__ = '0.1.0'

# Import core components
from .core import (
    BaseNode,
    CompositeNode,
    PyTorchModuleNode,
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor,
    DataType,
    PortType
)

# Import all nodes to trigger registration
from . import nodes

__all__ = [
    'BaseNode',
    'CompositeNode',
    'PyTorchModuleNode',
    'NodeRegistry',
    'ComputationalGraph',
    'GraphExecutor',
    'DataType',
    'PortType',
    'nodes',
]
