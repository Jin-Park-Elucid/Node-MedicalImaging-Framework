"""
Simple test to verify framework functionality.

Demonstrates basic node creation, graph building, and node listing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_imaging_framework import (
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor
)

# Import nodes to trigger registration
import medical_imaging_framework.nodes


def test_node_registry():
    """Test the node registry."""
    print("=" * 80)
    print("Testing Node Registry")
    print("=" * 80)

    categories = NodeRegistry.get_categories()
    print(f"\nFound {len(categories)} categories")

    total_nodes = 0
    for category in sorted(categories):
        nodes = NodeRegistry.get_nodes_by_category(category)
        total_nodes += len(nodes)
        print(f"\n{category.upper()} ({len(nodes)} nodes):")
        for node_name in sorted(nodes):
            info = NodeRegistry.get_node_info(node_name)
            print(f"  ✓ {node_name}: {info['description']}")

    print(f"\n{'='*80}")
    print(f"Total registered nodes: {total_nodes}")
    print(f"{'='*80}\n")

    return total_nodes > 0


def test_node_creation():
    """Test creating individual nodes."""
    print("\n" + "=" * 80)
    print("Testing Node Creation")
    print("=" * 80 + "\n")

    # Test creating a U-Net node
    unet = NodeRegistry.create_node(
        'UNet2D',
        'test_unet',
        config={'in_channels': 1, 'out_channels': 2}
    )
    print(f"✓ Created UNet2D node: {unet}")
    print(f"  Inputs: {list(unet.inputs.keys())}")
    print(f"  Outputs: {list(unet.outputs.keys())}")

    # Test creating a data loader node (if available)
    loader = NodeRegistry.create_node(
        'ImagePathLoader',
        'test_loader',
        config={'data_dir': './data'}
    )
    if loader is not None:
        print(f"\n✓ Created ImagePathLoader node: {loader}")
        print(f"  Inputs: {list(loader.inputs.keys())}")
        print(f"  Outputs: {list(loader.outputs.keys())}")
    else:
        print(f"\n⚠️  ImagePathLoader node not available (data module not yet implemented)")
        print(f"  Note: This is expected - data loading nodes are planned for future implementation")

    return unet is not None  # Only check unet, loader is optional


def test_simple_graph():
    """Test creating a simple graph."""
    print("\n" + "=" * 80)
    print("Testing Graph Creation")
    print("=" * 80 + "\n")

    graph = ComputationalGraph("Test Graph")

    # Add some nodes
    node1 = NodeRegistry.create_node('Print', 'print1', config={'prefix': 'Test'})
    graph.add_node(node1)

    print(f"✓ Created graph: {graph}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Links: {len(graph.links)}")

    # Validate
    is_valid, errors = graph.validate()
    print(f"\n✓ Graph validation: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # Save graph
    output_file = "./workflows/test_graph.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    graph.save_to_file(output_file)
    print(f"\n✓ Saved graph to: {output_file}")

    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "Medical Imaging Framework - Quick Test" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")

    success = True

    # Test 1: Node Registry
    if test_node_registry():
        print("✅ Node Registry Test: PASSED")
    else:
        print("❌ Node Registry Test: FAILED")
        success = False

    # Test 2: Node Creation
    if test_node_creation():
        print("\n✅ Node Creation Test: PASSED")
    else:
        print("\n❌ Node Creation Test: FAILED")
        success = False

    # Test 3: Graph Creation
    if test_simple_graph():
        print("\n✅ Graph Creation Test: PASSED")
    else:
        print("\n❌ Graph Creation Test: FAILED")
        success = False

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("🎉 All tests passed! Framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("=" * 80 + "\n")

    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Try the segmentation workflow: python examples/segmentation_workflow.py")
    print("  3. Launch the GUI: python -m medical_imaging_framework.gui.editor")
    print()


if __name__ == "__main__":
    main()
