"""
Core node system for medical imaging framework.

This module provides the foundational classes for building node-based
computational graphs for medical image processing, training, and inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
import torch.nn as nn
from collections import OrderedDict


class PortType(Enum):
    """Types of ports for node connections."""
    INPUT = "input"
    OUTPUT = "output"


class DataType(Enum):
    """Data types that can flow through ports."""
    TENSOR = "tensor"
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LOSS = "loss"
    METRICS = "metrics"
    IMAGE = "image"
    LABEL = "label"
    BATCH = "batch"
    CONFIG = "config"
    ANY = "any"


class Port:
    """
    Represents an input or output port on a node.

    Ports are connection points that allow data to flow between nodes.
    Each port has a name, type (input/output), and data type specification.
    """

    def __init__(
        self,
        name: str,
        port_type: PortType,
        data_type: DataType = DataType.ANY,
        node: Optional['BaseNode'] = None,
        optional: bool = False
    ):
        self.name = name
        self.port_type = port_type
        self.data_type = data_type
        self.node = node
        self.optional = optional
        self.value: Any = None
        self.metadata: Dict[str, Any] = {}
        self.links: List['Link'] = []

    def connect_to(self, target_port: 'Port') -> 'Link':
        """Create a link from this port to a target port."""
        if self.port_type != PortType.OUTPUT:
            raise ValueError("Can only connect from output ports")
        if target_port.port_type != PortType.INPUT:
            raise ValueError("Can only connect to input ports")

        # Check if already connected
        for link in self.links:
            if link.target == target_port:
                raise ValueError("Ports already connected")

        link = Link(self, target_port)
        self.links.append(link)
        target_port.links.append(link)
        return link

    def disconnect_all(self):
        """Remove all links connected to this port."""
        for link in self.links[:]:
            link.remove()

    def set_value(self, value: Any, metadata: Optional[Dict] = None):
        """Set the port value and optional metadata."""
        self.value = value
        if metadata:
            self.metadata.update(metadata)

    def get_value(self) -> Any:
        """Get the port value."""
        return self.value

    def is_connected(self) -> bool:
        """Check if this port has any connections."""
        return len(self.links) > 0

    def __repr__(self):
        return f"Port({self.name}, {self.port_type.value}, {self.data_type.value})"


class Link:
    """
    Represents a connection between two ports.

    Links transfer data from source (output) ports to target (input) ports.
    """

    def __init__(self, source: Port, target: Port):
        self.source = source
        self.target = target

    def transfer_data(self):
        """Transfer data from source to target port."""
        self.target.set_value(self.source.get_value(), self.source.metadata.copy())

    def remove(self):
        """Remove this link from both ports."""
        if self in self.source.links:
            self.source.links.remove(self)
        if self in self.target.links:
            self.target.links.remove(self)

    def __repr__(self):
        return f"Link({self.source.node.name}.{self.source.name} -> {self.target.node.name}.{self.target.name})"


class BaseNode(ABC):
    """
    Base class for all nodes in the computational graph.

    Nodes represent functional components in the medical imaging pipeline.
    Each node has inputs, outputs, configuration fields, and an execute method.

    Nodes can be simple (atomic operations) or composite (containing sub-nodes).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.inputs: OrderedDict[str, Port] = OrderedDict()
        self.outputs: OrderedDict[str, Port] = OrderedDict()
        self.position: Tuple[float, float] = (0, 0)
        self._executed = False
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For composite nodes
        self.is_composite = False
        self.sub_nodes: List['BaseNode'] = []

        # For visualization
        self.visual_node = None
        self.collapsed = False  # Whether to show sub-nodes or not

        # Initialize node-specific ports
        self._setup_ports()

    @abstractmethod
    def _setup_ports(self):
        """Setup input and output ports for this node. Override in subclasses."""
        pass

    def add_input(
        self,
        name: str,
        data_type: DataType = DataType.ANY,
        optional: bool = False
    ) -> Port:
        """Add an input port to this node."""
        port = Port(name, PortType.INPUT, data_type, self, optional)
        self.inputs[name] = port
        return port

    def add_output(
        self,
        name: str,
        data_type: DataType = DataType.ANY
    ) -> Port:
        """Add an output port to this node."""
        port = Port(name, PortType.OUTPUT, data_type, self)
        self.outputs[name] = port
        return port

    def get_input_value(self, name: str) -> Any:
        """Get the value from an input port."""
        if name not in self.inputs:
            raise ValueError(f"Input '{name}' does not exist on node '{self.name}'")
        return self.inputs[name].get_value()

    def set_output_value(self, name: str, value: Any, metadata: Optional[Dict] = None):
        """Set the value on an output port."""
        if name not in self.outputs:
            raise ValueError(f"Output '{name}' does not exist on node '{self.name}'")
        self.outputs[name].set_value(value, metadata)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value

    @abstractmethod
    def execute(self) -> bool:
        """
        Execute the node's processing logic.

        Returns:
            True if execution was successful, False otherwise.
        """
        pass

    def reset(self):
        """Reset the node state for re-execution."""
        self._executed = False
        for port in self.outputs.values():
            port.value = None
            port.metadata.clear()

    def is_ready(self) -> bool:
        """Check if all required inputs are available."""
        for port in self.inputs.values():
            if not port.optional and port.value is None and not port.is_connected():
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        # Find the registered name for this node class
        # This is important because the class name might be different from the registered name
        # e.g., class "AVTE2DLoaderNode" is registered as "AVTE2DLoader"
        node_type = self.__class__.__name__

        # Look up registered name in NodeRegistry
        try:
            from .registry import NodeRegistry
            all_nodes = NodeRegistry.get_all_nodes()

            # get_all_nodes() returns: {registered_name: {class: ..., category: ..., ...}}
            for registered_name, node_info in all_nodes.items():
                try:
                    # Safely check if this is the right node class
                    if hasattr(node_info, 'get'):
                        node_class = node_info.get('class')
                    elif hasattr(node_info, '__getitem__'):
                        node_class = node_info['class']
                    else:
                        continue

                    if node_class == self.__class__:
                        node_type = registered_name
                        break
                except (TypeError, KeyError, AttributeError):
                    # Skip entries that don't have the expected structure
                    continue
        except Exception:
            # If anything goes wrong, fall back to class name
            pass

        return {
            'type': node_type,
            'name': self.name,
            'position': self.position,
            'config': self.config,
            'is_composite': self.is_composite,
            'collapsed': self.collapsed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseNode':
        """Deserialize node from dictionary."""
        node = cls(data['name'], data.get('config', {}))
        node.position = tuple(data['position'])
        node.collapsed = data.get('collapsed', False)
        return node

    def get_field_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return field definitions for the properties panel.

        Returns a dictionary mapping field names to their definitions.
        Each definition contains: type, label, default, choices (for dropdowns), etc.
        """
        return {}

    def set_device(self, device: Union[str, torch.device]):
        """Set the computation device (cpu/cuda)."""
        self._device = torch.device(device)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeNode(BaseNode):
    """
    A node that contains a sub-graph of nodes.

    Composite nodes allow hierarchical composition - they appear as a single
    node but internally contain a complete graph of sub-nodes.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.is_composite = True
        super().__init__(name, config)
        from .graph import ComputationalGraph
        self.sub_graph = ComputationalGraph(name=f"{name}_subgraph")

    def _setup_ports(self):
        """Composite nodes define ports based on their internal structure."""
        pass  # Ports are added dynamically

    def add_sub_node(self, node: BaseNode):
        """Add a node to the internal sub-graph."""
        self.sub_nodes.append(node)
        self.sub_graph.add_node(node)

    def connect_internal(self, source_node: str, source_port: str,
                        target_node: str, target_port: str):
        """Connect two nodes within the sub-graph."""
        self.sub_graph.connect(source_node, source_port, target_node, target_port)

    def expose_input(self, internal_node: str, internal_port: str,
                     external_name: Optional[str] = None):
        """
        Expose an internal node's input as an external input port.

        This allows data to flow from outside the composite node into
        an internal node.
        """
        node = self.sub_graph.get_node(internal_node)
        if not node or internal_port not in node.inputs:
            raise ValueError(f"Cannot find {internal_node}.{internal_port}")

        port_name = external_name or f"{internal_node}_{internal_port}"
        internal_input = node.inputs[internal_port]

        # Create external port that mirrors the internal port
        external_port = self.add_input(
            port_name,
            internal_input.data_type,
            internal_input.optional
        )

        # Store mapping for execution
        if not hasattr(self, '_input_mappings'):
            self._input_mappings = {}
        self._input_mappings[port_name] = (internal_node, internal_port)

    def expose_output(self, internal_node: str, internal_port: str,
                     external_name: Optional[str] = None):
        """
        Expose an internal node's output as an external output port.

        This allows data from an internal node to flow out of the
        composite node.
        """
        node = self.sub_graph.get_node(internal_node)
        if not node or internal_port not in node.outputs:
            raise ValueError(f"Cannot find {internal_node}.{internal_port}")

        port_name = external_name or f"{internal_node}_{internal_port}"
        internal_output = node.outputs[internal_port]

        # Create external port that mirrors the internal port
        external_port = self.add_output(port_name, internal_output.data_type)

        # Store mapping for execution
        if not hasattr(self, '_output_mappings'):
            self._output_mappings = {}
        self._output_mappings[port_name] = (internal_node, internal_port)

    def execute(self) -> bool:
        """Execute the composite node by executing its sub-graph."""
        try:
            # Transfer external inputs to internal nodes
            if hasattr(self, '_input_mappings'):
                for ext_port, (int_node, int_port) in self._input_mappings.items():
                    value = self.get_input_value(ext_port)
                    node = self.sub_graph.get_node(int_node)
                    node.inputs[int_port].set_value(value)

            # Execute sub-graph
            from .executor import GraphExecutor
            executor = GraphExecutor(self.sub_graph)
            results = executor.execute()

            # Transfer internal outputs to external ports
            if hasattr(self, '_output_mappings'):
                for ext_port, (int_node, int_port) in self._output_mappings.items():
                    node = self.sub_graph.get_node(int_node)
                    value = node.outputs[int_port].get_value()
                    self.set_output_value(ext_port, value)

            return True
        except Exception as e:
            print(f"Error executing composite node {self.name}: {e}")
            return False


class PyTorchModuleNode(BaseNode):
    """
    Base class for nodes that wrap PyTorch nn.Module.

    This provides integration between the node system and PyTorch's
    neural network modules.

    Network nodes should output both:
    - 'model': The PyTorch module itself (DataType.MODEL) for training
    - 'output': Tensor output from forward pass (DataType.TENSOR) for inference
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.module: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.training_mode = False

    @abstractmethod
    def build_module(self) -> nn.Module:
        """Build and return the PyTorch module. Override in subclasses."""
        pass

    def initialize_module(self):
        """Initialize the PyTorch module."""
        if self.module is None:
            self.module = self.build_module()
            self.module.to(self._device)

    def set_training_mode(self, training: bool):
        """Set training/evaluation mode."""
        self.training_mode = training
        if self.module is not None:
            self.module.train(training)

    def get_parameters(self):
        """Get module parameters."""
        if self.module is None:
            self.initialize_module()
        return self.module.parameters()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.module is None:
            raise ValueError("Module not initialized")

        checkpoint = {
            'model_state_dict': self.module.state_dict(),
            'config': self.config,
            'node_type': self.__class__.__name__
        }

        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self._device)

        if self.module is None:
            self.initialize_module()

        self.module.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
