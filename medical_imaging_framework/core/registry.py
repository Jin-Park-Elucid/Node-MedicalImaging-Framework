"""
Node registry for managing available node types.

The registry provides a centralized system for registering, discovering,
and instantiating nodes in the medical imaging framework.
"""

from typing import Dict, Type, Optional, List, Callable
from .node import BaseNode


class NodeRegistry:
    """
    Registry for managing available node types.

    The registry maintains a catalog of all node classes that can be
    instantiated in the framework, organized by category.
    """

    _registry: Dict[str, Dict[str, any]] = {}
    _categories: Dict[str, List[str]] = {}

    @classmethod
    def register(
        cls,
        category: str,
        name: Optional[str] = None,
        description: str = "",
        icon: str = "node"
    ) -> Callable:
        """
        Decorator to register a node class.

        Args:
            category: Category name (e.g., 'data', 'networks', 'training')
            name: Display name (defaults to class name)
            description: Description of the node's functionality
            icon: Icon identifier for GUI display

        Example:
            @NodeRegistry.register('data', 'DataLoader')
            class DataLoaderNode(BaseNode):
                ...
        """
        def decorator(node_class: Type[BaseNode]) -> Type[BaseNode]:
            node_name = name or node_class.__name__

            cls._registry[node_name] = {
                'class': node_class,
                'category': category,
                'description': description,
                'icon': icon,
                'name': node_name
            }

            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(node_name)

            return node_class

        return decorator

    @classmethod
    def unregister(cls, name: str):
        """Unregister a node type."""
        if name in cls._registry:
            category = cls._registry[name]['category']
            if category in cls._categories and name in cls._categories[category]:
                cls._categories[category].remove(name)
            del cls._registry[name]

    @classmethod
    def get_node_class(cls, name: str) -> Optional[Type[BaseNode]]:
        """Get a node class by name."""
        entry = cls._registry.get(name)
        return entry['class'] if entry else None

    @classmethod
    def get_node_info(cls, name: str) -> Optional[Dict]:
        """Get full information about a node type."""
        return cls._registry.get(name)

    @classmethod
    def create_node(
        cls,
        class_name: str,
        instance_name: str,
        config: Optional[Dict] = None
    ) -> Optional[BaseNode]:
        """
        Create a node instance by class name.

        Args:
            class_name: Name of the node class
            instance_name: Unique name for this instance
            config: Configuration dictionary

        Returns:
            Node instance or None if class not found
        """
        node_class = cls.get_node_class(class_name)
        if node_class:
            return node_class(instance_name, config)
        return None

    @classmethod
    def get_all_nodes(cls) -> Dict[str, Dict]:
        """Get all registered nodes."""
        return cls._registry.copy()

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all category names."""
        return list(cls._categories.keys())

    @classmethod
    def get_nodes_by_category(cls, category: str) -> List[str]:
        """Get all node names in a category."""
        return cls._categories.get(category, []).copy()

    @classmethod
    def search_nodes(cls, query: str) -> List[Dict]:
        """
        Search for nodes by name or description.

        Args:
            query: Search query string

        Returns:
            List of matching node information dictionaries
        """
        query_lower = query.lower()
        results = []

        for name, info in cls._registry.items():
            if (query_lower in name.lower() or
                query_lower in info['description'].lower() or
                query_lower in info['category'].lower()):
                results.append(info)

        return results

    @classmethod
    def clear(cls):
        """Clear the entire registry. Mainly for testing."""
        cls._registry.clear()
        cls._categories.clear()

    @classmethod
    def get_node_hierarchy(cls) -> Dict[str, List[Dict]]:
        """
        Get nodes organized by category hierarchy.

        Returns:
            Dictionary mapping categories to lists of node info
        """
        hierarchy = {}
        for category in cls._categories:
            hierarchy[category] = [
                cls._registry[node_name]
                for node_name in cls._categories[category]
            ]
        return hierarchy

    @classmethod
    def validate_node_class(cls, node_class: Type) -> bool:
        """
        Validate that a class can be registered as a node.

        Args:
            node_class: Class to validate

        Returns:
            True if valid, False otherwise
        """
        from .node import BaseNode

        if not issubclass(node_class, BaseNode):
            return False

        required_methods = ['_setup_ports', 'execute']
        for method in required_methods:
            if not hasattr(node_class, method):
                return False

        return True
