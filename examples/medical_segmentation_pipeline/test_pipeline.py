"""
Medical Image Segmentation Testing Pipeline with Visualization.

This script demonstrates inference and visualization:
- Loads trained model
- Runs inference on test data
- Generates comparison images (input, ground truth, prediction)
- Saves visualization images
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_imaging_framework.core import (
    ComputationalGraph,
    GraphExecutor,
    NodeRegistry
)

# Import all node modules
import medical_imaging_framework.nodes

# Import custom dataloader
from custom_dataloader import MedicalSegmentationLoaderNode


def visualize_predictions(images, masks_gt, masks_pred, output_dir, num_samples=10):
    """
    Create visualization comparing input, ground truth, and predictions.

    Args:
        images: Input images tensor (B, 1, H, W)
        masks_gt: Ground truth masks tensor (B, H, W)
        masks_pred: Predicted masks tensor (B, H, W)
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    num_samples = min(num_samples, images.shape[0])

    print(f"\nGenerating visualizations for {num_samples} samples...")

    for i in range(num_samples):
        # Extract single sample
        img = images[i, 0].cpu().numpy()  # (H, W)
        gt = masks_gt[i].cpu().numpy()    # (H, W)
        pred = masks_pred[i].cpu().numpy()  # (H, W)

        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 1. Input image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # 2. Ground truth mask
        axes[1].imshow(gt, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 3. Predicted mask
        axes[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # 4. Overlay comparison
        # Create RGB image: Red=GT only, Green=Pred only, Yellow=Both
        overlay = np.zeros((*gt.shape, 3))
        overlay[gt == 1, 0] = 1.0       # Red channel for GT
        overlay[pred == 1, 1] = 1.0     # Green channel for prediction
        # Where both are 1, it becomes yellow

        axes[3].imshow(img, cmap='gray')
        axes[3].imshow(overlay, alpha=0.5)
        axes[3].set_title('Overlay (R=GT, G=Pred, Y=Both)')
        axes[3].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"comparison_{i:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if (i + 1) % 5 == 0:
            print(f"  Saved {i + 1}/{num_samples} visualizations")

    print(f"✓ Saved {num_samples} visualizations to {output_dir}")


def calculate_metrics(masks_gt, masks_pred):
    """Calculate segmentation metrics."""
    # Flatten
    gt_flat = masks_gt.reshape(-1)
    pred_flat = masks_pred.reshape(-1)

    # Calculate metrics
    tp = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (gt_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (gt_flat == 0)).sum().item()

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # IoU (Intersection over Union)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dice': dice,
        'iou': iou
    }


def main():
    """Main testing pipeline."""
    print("="*80)
    print("MEDICAL IMAGE SEGMENTATION TESTING PIPELINE")
    print("="*80)

    # Configuration
    data_dir = Path(__file__).parent / "data"
    model_dir = Path(__file__).parent / "models"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    batch_size = 8

    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model directory: {model_dir}")
    print(f"  Results directory: {results_dir}")
    print(f"  Batch size: {batch_size}")

    # Find best model
    model_files = list(model_dir.glob("best_model*.pth"))
    if not model_files:
        print(f"\n✗ No trained model found in {model_dir}")
        print("Please run train_pipeline.py first!")
        return

    model_path = model_files[0]
    print(f"  Model: {model_path.name}")

    # Create computational graph
    graph = ComputationalGraph("MedicalSegmentationTesting")

    # 1. Create data loader
    print("\n1. Setting up data loader...")
    loader_node = NodeRegistry.create_node(
        'MedicalSegmentationLoader',
        'data_loader',
        config={
            'data_dir': str(data_dir),
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle_train': False
        }
    )
    graph.add_node(loader_node)

    # 2. Create UNet2D model
    print("2. Setting up UNet2D model...")
    model_node = NodeRegistry.create_node(
        'UNet2D',
        'unet_model',
        config={
            'in_channels': 1,
            'out_channels': 2,
            'base_channels': 32,
            'depth': 3
        }
    )
    graph.add_node(model_node)

    # 3. Create predictor
    print("3. Setting up batch predictor...")
    predictor_node = NodeRegistry.create_node(
        'BatchPredictor',
        'predictor',
        config={
            'device': 'cuda'  # Will fallback to CPU if unavailable
        }
    )
    graph.add_node(predictor_node)

    # Connect nodes
    print("\n4. Connecting nodes...")
    graph.connect('data_loader', 'test_loader', 'predictor', 'dataloader')
    graph.connect('unet_model', 'module', 'predictor', 'model')

    print("✓ Graph construction complete!")

    # Load trained weights
    print(f"\n5. Loading trained model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Build the model first by executing the model node
        model_node.execute()
        model = model_node.outputs['module'].get_value()
        model.load_state_dict(state_dict)
        print("✓ Model loaded successfully!")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Execute inference
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80 + "\n")

    try:
        # Execute data loader
        loader_node.execute()
        test_loader = loader_node.outputs['test_loader'].get_value()
        num_test = loader_node.outputs['num_test'].get_value()

        print(f"Test set: {num_test} samples, {len(test_loader)} batches")

        # Run inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model.to(device)
        model.eval()

        all_images = []
        all_masks_gt = []
        all_masks_pred = []

        print("\nProcessing batches...")
        with torch.no_grad():
            for batch_idx, (images, masks_gt) in enumerate(test_loader):
                images = images.to(device)
                masks_gt = masks_gt.to(device)

                # Forward pass
                outputs = model(images)  # (B, 2, H, W)

                # Get predictions (argmax over classes)
                masks_pred = torch.argmax(outputs, dim=1)  # (B, H, W)

                # Store for visualization
                all_images.append(images.cpu())
                all_masks_gt.append(masks_gt.cpu())
                all_masks_pred.append(masks_pred.cpu())

                if (batch_idx + 1) % 2 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

        # Concatenate all batches
        all_images = torch.cat(all_images, dim=0)
        all_masks_gt = torch.cat(all_masks_gt, dim=0)
        all_masks_pred = torch.cat(all_masks_pred, dim=0)

        print(f"\n✓ Inference complete!")
        print(f"  Processed {all_images.shape[0]} samples")

        # Calculate metrics
        print("\n" + "="*80)
        print("CALCULATING METRICS")
        print("="*80)

        metrics = calculate_metrics(all_masks_gt, all_masks_pred)

        print("\nSegmentation Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Dice:      {metrics['dice']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")

        # Save metrics
        metrics_file = results_dir / "test_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Medical Segmentation Test Metrics\n")
            f.write("="*50 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key.capitalize():12s}: {value:.4f}\n")

        print(f"\n✓ Metrics saved to {metrics_file}")

        # Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        visualize_predictions(
            all_images,
            all_masks_gt,
            all_masks_pred,
            results_dir / "visualizations",
            num_samples=min(20, all_images.shape[0])
        )

        print("\n" + "="*80)
        print("✓ TESTING COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {results_dir}")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Visualizations: {results_dir / 'visualizations'}/")
        print(f"\nVisualization Legend:")
        print(f"  - Red:    Ground truth only")
        print(f"  - Green:  Prediction only")
        print(f"  - Yellow: Both (correct prediction)")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
