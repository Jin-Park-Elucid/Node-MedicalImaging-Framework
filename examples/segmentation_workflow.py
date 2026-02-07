"""
Example: Medical Image Segmentation Workflow

This example demonstrates how to build a complete segmentation pipeline
using the node-based medical imaging framework.
"""

import torch
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


def create_segmentation_workflow():
    """
    Create a segmentation workflow using U-Net.

    Pipeline:
    1. Load image paths
    2. Create data loader
    3. Define U-Net model
    4. Train the model
    5. Run inference
    6. Calculate metrics
    7. Visualize results
    """

    graph = ComputationalGraph("Segmentation Pipeline")

    # 1. Load image paths
    path_loader = NodeRegistry.create_node(
        'ImagePathLoader',
        'path_loader',
        config={
            'data_dir': './data/images',
            'pattern': '*.nii.gz'
        }
    )
    graph.add_node(path_loader)

    # 2. Create data loader
    data_loader = NodeRegistry.create_node(
        'DataLoader',
        'train_loader',
        config={
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 0
        }
    )
    graph.add_node(data_loader)

    # Connect path loader to data loader
    graph.connect('path_loader', 'image_paths', 'train_loader', 'image_paths')

    # 3. Define U-Net model
    unet = NodeRegistry.create_node(
        'UNet2D',
        'unet_model',
        config={
            'in_channels': 1,
            'out_channels': 2,
            'base_channels': 64,
            'depth': 4
        }
    )
    graph.add_node(unet)

    # 4. Create loss function
    loss_fn = NodeRegistry.create_node(
        'LossFunction',
        'loss_function',
        config={'loss_type': 'dice'}
    )
    graph.add_node(loss_fn)

    # 5. Train the model
    trainer = NodeRegistry.create_node(
        'Trainer',
        'trainer',
        config={
            'num_epochs': 10,
            'learning_rate': 0.001
        }
    )
    graph.add_node(trainer)

    # Connect to trainer
    graph.connect('unet_model', 'output', 'trainer', 'model')
    graph.connect('train_loader', 'batch', 'trainer', 'dataloader')
    graph.connect('loss_function', 'loss_fn', 'trainer', 'loss_fn')

    # 6. Plot training metrics
    metrics_plotter = NodeRegistry.create_node(
        'MetricsPlotter',
        'metrics_plot',
        config={
            'save_path': './outputs/training_loss.png',
            'show': 'False'
        }
    )
    graph.add_node(metrics_plotter)
    graph.connect('trainer', 'metrics', 'metrics_plot', 'metrics')

    # 7. Run inference on test data
    test_loader = NodeRegistry.create_node(
        'DataLoader',
        'test_loader',
        config={
            'batch_size': 1,
            'shuffle': False
        }
    )
    graph.add_node(test_loader)

    predictor = NodeRegistry.create_node(
        'BatchPredictor',
        'predictor',
        config={}
    )
    graph.add_node(predictor)

    graph.connect('trainer', 'trained_model', 'predictor', 'model')
    graph.connect('test_loader', 'batch', 'predictor', 'dataloader')

    # 8. Calculate metrics
    metrics_calc = NodeRegistry.create_node(
        'MetricsCalculator',
        'metrics',
        config={}
    )
    graph.add_node(metrics_calc)

    graph.connect('predictor', 'all_predictions', 'metrics', 'predictions')
    graph.connect('predictor', 'all_labels', 'metrics', 'labels')

    # 9. Visualize results
    print_metrics = NodeRegistry.create_node(
        'Print',
        'print_metrics',
        config={'prefix': 'Final Metrics'}
    )
    graph.add_node(print_metrics)
    graph.connect('metrics', 'metrics', 'print_metrics', 'input')

    return graph


def create_simple_inference_workflow():
    """
    Create a simple inference workflow.

    Pipeline:
    1. Load a trained model
    2. Load test data
    3. Run inference
    4. Visualize results
    """

    graph = ComputationalGraph("Inference Pipeline")

    # 1. Load model
    unet = NodeRegistry.create_node(
        'UNet2D',
        'unet_model',
        config={
            'in_channels': 1,
            'out_channels': 2,
            'base_channels': 64
        }
    )
    graph.add_node(unet)

    # 2. Load test data
    test_loader = NodeRegistry.create_node(
        'DataLoader',
        'test_loader',
        config={'batch_size': 1, 'shuffle': False}
    )
    graph.add_node(test_loader)

    # 3. Extract single batch
    batch_extractor = NodeRegistry.create_node(
        'BatchExtractor',
        'batch_extract',
        config={'batch_idx': 0}
    )
    graph.add_node(batch_extractor)
    graph.connect('test_loader', 'batch', 'batch_extract', 'dataloader')

    # 4. Run inference
    predictor = NodeRegistry.create_node(
        'Predictor',
        'predictor',
        config={}
    )
    graph.add_node(predictor)
    graph.connect('unet_model', 'output', 'predictor', 'model')
    graph.connect('batch_extract', 'images', 'predictor', 'input')

    # 5. Visualize
    viewer = NodeRegistry.create_node(
        'ImageViewer',
        'viewer',
        config={
            'save_path': './outputs/prediction.png',
            'show': 'False'
        }
    )
    graph.add_node(viewer)
    graph.connect('batch_extract', 'images', 'viewer', 'image')
    graph.connect('predictor', 'predictions', 'viewer', 'segmentation')

    return graph


def main():
    """Main execution function."""

    print("=" * 80)
    print("Medical Imaging Framework - Segmentation Example")
    print("=" * 80)

    # Show available nodes
    print("\nAvailable Node Categories:")
    categories = NodeRegistry.get_categories()
    for category in categories:
        nodes = NodeRegistry.get_nodes_by_category(category)
        print(f"\n{category.upper()}:")
        for node_name in nodes:
            info = NodeRegistry.get_node_info(node_name)
            print(f"  - {node_name}: {info['description']}")

    print("\n" + "=" * 80)
    print("Creating Segmentation Workflow...")
    print("=" * 80 + "\n")

    # Create workflow
    graph = create_segmentation_workflow()

    print(f"Created graph: {graph}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Links: {len(graph.links)}")

    # Validate graph
    is_valid, errors = graph.validate()
    if not is_valid:
        print("\nGraph validation failed:")
        for error in errors:
            print(f"  - {error}")
        return

    print("\nGraph is valid!")

    # Get execution order
    execution_order = graph.get_execution_order()
    print(f"\nExecution order:")
    for i, node_name in enumerate(execution_order, 1):
        node = graph.get_node(node_name)
        print(f"  {i}. {node_name} ({node.__class__.__name__})")

    # Save graph
    output_file = "./workflows/segmentation_workflow.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    graph.save_to_file(output_file)
    print(f"\nWorkflow saved to: {output_file}")

    # Execute (commented out - requires actual data)
    # print("\nExecuting workflow...")
    # executor = GraphExecutor(graph, progress_callback=lambda name, prog: print(f"  {name}: {prog:.1%}"))
    # result = executor.execute()
    # print(f"\nExecution completed: {result.status.value}")
    # print(f"Time: {result.execution_time:.2f}s")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
