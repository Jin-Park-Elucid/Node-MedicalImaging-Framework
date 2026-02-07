"""
Medical Image Segmentation Training Pipeline.

This script demonstrates a complete training pipeline using the framework nodes:
- MedicalSegmentationLoaderNode for data loading
- UNet2DNode for the segmentation network
- LossFunctionNode for Dice loss
- OptimizerNode for Adam optimizer
- TrainerNode for the training loop
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_imaging_framework.core import (
    ComputationalGraph,
    GraphExecutor,
    NodeRegistry
)

# Import all node modules to ensure registration
import medical_imaging_framework.nodes

# Import the custom dataloader
from custom_dataloader import MedicalSegmentationLoaderNode


def main():
    """Main training pipeline."""
    print("="*80)
    print("MEDICAL IMAGE SEGMENTATION TRAINING PIPELINE")
    print("="*80)

    # Create computational graph
    graph = ComputationalGraph("MedicalSegmentationTraining")

    # Configuration
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)

    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-3

    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model directory: {model_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # 1. Create data loader node
    print("\n1. Setting up data loader...")
    loader_node = NodeRegistry.create_node(
        'MedicalSegmentationLoader',
        'data_loader',
        config={
            'data_dir': str(data_dir),
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle_train': True
        }
    )
    graph.add_node(loader_node)

    # 2. Create UNet2D model
    print("2. Setting up UNet2D model...")
    model_node = NodeRegistry.create_node(
        'UNet2D',
        'unet_model',
        config={
            'in_channels': 1,      # Grayscale input
            'out_channels': 2,     # Binary segmentation (background, foreground)
            'base_channels': 32,   # Smaller for faster training
            'depth': 3             # 3 levels
        }
    )
    graph.add_node(model_node)

    # 3. Create loss function (Dice loss for segmentation)
    print("3. Setting up Dice loss...")
    loss_node = NodeRegistry.create_node(
        'LossFunction',
        'loss',
        config={
            'loss_type': 'dice'
        }
    )
    graph.add_node(loss_node)

    # 4. Create optimizer (Adam)
    print("4. Setting up Adam optimizer...")
    optimizer_node = NodeRegistry.create_node(
        'Optimizer',
        'optimizer',
        config={
            'optimizer_type': 'adam',
            'lr': learning_rate,
            'weight_decay': 1e-5
        }
    )
    graph.add_node(optimizer_node)

    # 5. Create trainer node
    print("5. Setting up trainer...")
    trainer_node = NodeRegistry.create_node(
        'Trainer',
        'trainer',
        config={
            'num_epochs': num_epochs,
            'device': 'cuda',  # Will fallback to CPU if CUDA unavailable
            'save_dir': str(model_dir),
            'save_best': True,
            'patience': 5
        }
    )
    graph.add_node(trainer_node)

    # 6. Connect nodes
    print("\n6. Connecting nodes...")

    # Data loader outputs -> Trainer inputs
    graph.connect('data_loader', 'train_loader', 'trainer', 'train_loader')
    graph.connect('data_loader', 'test_loader', 'trainer', 'val_loader')

    # Model -> Trainer
    graph.connect('unet_model', 'module', 'trainer', 'model')

    # Loss -> Trainer
    graph.connect('loss', 'loss', 'trainer', 'loss_fn')

    # Optimizer needs model parameters
    graph.connect('unet_model', 'module', 'optimizer', 'model')
    graph.connect('optimizer', 'optimizer', 'trainer', 'optimizer')

    print("✓ Graph construction complete!")

    # 7. Validate graph
    print("\n7. Validating graph...")
    is_valid, errors = graph.validate()

    if not is_valid:
        print("✗ Graph validation failed:")
        for error in errors:
            print(f"  - {error}")
        return

    print("✓ Graph validation passed!")

    # 8. Visualize graph structure
    print("\n8. Graph structure:")
    print(f"  Nodes: {len(graph.nodes)}")
    for node_name, node in graph.nodes.items():
        print(f"    - {node_name} ({node.__class__.__name__})")
    print(f"  Links: {len(graph.links)}")
    for link in graph.links:
        print(f"    - {link.source_node.name}.{link.source_port.name} -> "
              f"{link.target_node.name}.{link.target_port.name}")

    # 9. Execute training pipeline
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    executor = GraphExecutor(graph)

    try:
        success = executor.execute()

        if success:
            print("\n" + "="*80)
            print("✓ TRAINING COMPLETE!")
            print("="*80)

            # Get training results
            final_loss = trainer_node.outputs['final_loss'].get_value()
            best_loss = trainer_node.outputs['best_loss'].get_value()
            model_path = trainer_node.outputs['model_path'].get_value()

            print(f"\nTraining Results:")
            print(f"  Final Loss: {final_loss:.4f}")
            print(f"  Best Loss: {best_loss:.4f}")
            print(f"  Model saved to: {model_path}")

            print(f"\nNext steps:")
            print(f"  1. Run: python examples/medical_segmentation_pipeline/test_pipeline.py")
            print(f"  2. Check results in: examples/medical_segmentation_pipeline/results/")

        else:
            print("\n✗ Training failed!")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
