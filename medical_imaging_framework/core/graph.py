"""
Computational graph for managing nodes and their connections.

The graph represents the complete pipeline, managing node execution order
and data flow between components.
"""

import json
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict, deque
from .node import BaseNode, Link
from .registry import NodeRegistry


class ComputationalGraph:
    """
    Manages a collection of nodes and their connections.

    The graph handles:
    - Adding/removing nodes
    - Creating connections between nodes
    - Topological sorting for execution order
    - Serialization/deserialization
    - Cycle detection
    """

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.links: List[Link] = []
        self._execution_order: Optional[List[str]] = None
        self._dirty = True  # Whether execution order needs recomputation

    def add_node(self, node: BaseNode):
        """Add a node to the graph."""
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists")

        self.nodes[node.name] = node
        self._dirty = True
        return node

    def remove_node(self, node_name: str):
        """Remove a node and all its connections."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found")

        node = self.nodes[node_name]

        # Remove all connections
        for port in list(node.inputs.values()) + list(node.outputs.values()):
            for link in port.links[:]:
                self.remove_link(link)

        del self.nodes[node_name]
        self._dirty = True

    def get_node(self, name: str) -> Optional[BaseNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def connect(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str
    ) -> Link:
        """
        Connect two nodes via their ports.

        Args:
            source_node: Name of the source node
            source_port: Name of the output port on source node
            target_node: Name of the target node
            target_port: Name of the input port on target node

        Returns:
            The created Link object

        Raises:
            ValueError: If nodes or ports don't exist, or connection is invalid
        """
        src_node = self.get_node(source_node)
        tgt_node = self.get_node(target_node)

        if not src_node:
            raise ValueError(f"Source node '{source_node}' not found")
        if not tgt_node:
            raise ValueError(f"Target node '{target_node}' not found")

        if source_port not in src_node.outputs:
            raise ValueError(f"Output port '{source_port}' not found on '{source_node}'")
        if target_port not in tgt_node.inputs:
            raise ValueError(f"Input port '{target_port}' not found on '{target_node}'")

        link = src_node.outputs[source_port].connect_to(tgt_node.inputs[target_port])
        self.links.append(link)
        self._dirty = True

        return link

    def disconnect(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str
    ):
        """Disconnect two nodes."""
        for link in self.links[:]:
            if (link.source.node.name == source_node and
                link.source.name == source_port and
                link.target.node.name == target_node and
                link.target.name == target_port):
                self.remove_link(link)
                break

    def remove_link(self, link: Link):
        """Remove a link from the graph."""
        link.remove()
        if link in self.links:
            self.links.remove(link)
        self._dirty = True

    def get_dependencies(self, node_name: str) -> Set[str]:
        """
        Get all nodes that the given node depends on.

        Returns:
            Set of node names that must execute before this node
        """
        dependencies = set()
        node = self.get_node(node_name)

        if node:
            for input_port in node.inputs.values():
                for link in input_port.links:
                    dependencies.add(link.source.node.name)

        return dependencies

    def get_dependents(self, node_name: str) -> Set[str]:
        """
        Get all nodes that depend on the given node.

        Returns:
            Set of node names that need this node's outputs
        """
        dependents = set()
        node = self.get_node(node_name)

        if node:
            for output_port in node.outputs.values():
                for link in output_port.links:
                    dependents.add(link.target.node.name)

        return dependents

    def topological_sort(self) -> List[str]:
        """
        Compute topological sort of nodes for execution order.

        Returns:
            List of node names in execution order

        Raises:
            RuntimeError: If a cycle is detected
        """
        # Build adjacency list and in-degree count
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)

        for node_name in self.nodes:
            in_degree[node_name] = 0

        for link in self.links:
            source = link.source.node.name
            target = link.target.node.name
            adjacency[source].append(target)
            in_degree[target] += 1

        # Kahn's algorithm
        queue = deque([name for name in self.nodes if in_degree[name] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise RuntimeError("Cycle detected in computational graph")

        return result

    def get_execution_order(self, force_recompute: bool = False) -> List[str]:
        """
        Get the execution order for nodes.

        Args:
            force_recompute: Force recomputation even if cached

        Returns:
            List of node names in execution order
        """
        if self._dirty or force_recompute or self._execution_order is None:
            self._execution_order = self.topological_sort()
            self._dirty = False

        return self._execution_order

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the graph for common issues.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Check for cycles
        try:
            self.topological_sort()
        except RuntimeError as e:
            errors.append(str(e))

        # Check for disconnected required inputs
        for node_name, node in self.nodes.items():
            for port_name, port in node.inputs.items():
                if not port.optional and not port.is_connected() and port.value is None:
                    errors.append(
                        f"Required input '{port_name}' on node '{node_name}' is not connected"
                    )

        # Check for type mismatches
        from .node import DataType
        for link in self.links:
            source_type = link.source.data_type
            target_type = link.target.data_type

            if (source_type != DataType.ANY and
                target_type != DataType.ANY and
                source_type != target_type):
                errors.append(
                    f"Type mismatch: {link.source.node.name}.{link.source.name} "
                    f"({source_type.value}) -> {link.target.node.name}.{link.target.name} "
                    f"({target_type.value})"
                )

        return len(errors) == 0, errors

    def reset(self):
        """Reset all nodes in the graph."""
        for node in self.nodes.values():
            node.reset()

    def save_to_file(self, filename: str):
        """Save graph to JSON file."""
        data = {
            'name': self.name,
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'links': [
                {
                    'source_node': link.source.node.name,
                    'source_port': link.source.name,
                    'target_node': link.target.node.name,
                    'target_port': link.target.name
                }
                for link in self.links
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        """Load graph from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.name = data['name']
        self.nodes.clear()
        self.links.clear()

        # Create nodes
        for node_data in data['nodes']:
            node = NodeRegistry.create_node(
                node_data['type'],
                node_data['name'],
                node_data.get('config', {})
            )
            if node:
                node.position = tuple(node_data['position'])
                node.collapsed = node_data.get('collapsed', False)
                self.add_node(node)

        # Create links
        for link_data in data['links']:
            try:
                self.connect(
                    link_data['source_node'],
                    link_data['source_port'],
                    link_data['target_node'],
                    link_data['target_port']
                )
            except ValueError as e:
                print(f"Warning: Could not create link: {e}")

        self._dirty = True

    def get_subgraph(self, node_names: List[str]) -> 'ComputationalGraph':
        """
        Extract a subgraph containing only the specified nodes.

        Args:
            node_names: List of node names to include

        Returns:
            New ComputationalGraph containing only specified nodes
        """
        subgraph = ComputationalGraph(f"{self.name}_subgraph")

        # Add nodes
        for name in node_names:
            if name in self.nodes:
                node = self.nodes[name]
                # Create a new instance with same config
                new_node = NodeRegistry.create_node(
                    node.__class__.__name__,
                    node.name,
                    node.config.copy()
                )
                if new_node:
                    new_node.position = node.position
                    subgraph.add_node(new_node)

        # Add links that connect nodes within the subgraph
        for link in self.links:
            if (link.source.node.name in node_names and
                link.target.node.name in node_names):
                subgraph.connect(
                    link.source.node.name,
                    link.source.name,
                    link.target.node.name,
                    link.target.name
                )

        return subgraph

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"ComputationalGraph(name='{self.name}', nodes={len(self.nodes)}, links={len(self.links)})"
