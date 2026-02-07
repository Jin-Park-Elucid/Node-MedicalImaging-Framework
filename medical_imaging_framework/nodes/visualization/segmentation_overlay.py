"""
Segmentation overlay visualization node.

Creates visualization overlays of ground truth and predictions on input images.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ...core import BaseNode, NodeRegistry, DataType


@NodeRegistry.register('visualization', 'SegmentationOverlay',
                      description='Visualize segmentation overlays on images')
class SegmentationOverlayNode(BaseNode):
    """
    Create overlay visualizations of segmentation results.

    Takes input images, ground truth masks, and predictions, and creates
    visualizations showing both overlaid on the original images.
    Saves results to disk.
    """

    def _setup_ports(self):
        self.add_input('images', DataType.TENSOR)
        self.add_input('labels', DataType.TENSOR)
        self.add_input('predictions', DataType.TENSOR)
        self.add_output('num_saved', DataType.ANY)

    def execute(self) -> bool:
        try:
            images = self.get_input_value('images')
            labels = self.get_input_value('labels')
            predictions = self.get_input_value('predictions')

            if images is None or labels is None or predictions is None:
                print("SegmentationOverlay: Missing required inputs")
                return False

            # Get configuration
            output_dir = self.get_config('output_dir', 'visualization_output')
            max_images = int(self.get_config('max_images', 10))
            alpha = float(self.get_config('alpha', 0.4))
            save_individual = self.get_config('save_individual', 'True').lower() in ['true', '1', 'yes']
            save_grid = self.get_config('save_grid', 'True').lower() in ['true', '1', 'yes']

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"SEGMENTATION VISUALIZATION")
            print(f"{'='*60}")
            print(f"Output directory: {output_path.absolute()}")

            # Move tensors to CPU and convert to numpy
            images = images.cpu()
            labels = labels.cpu()
            predictions = predictions.cpu()

            # Determine number of images to save
            num_images = min(len(images), max_images)
            print(f"Creating visualizations for {num_images} images...")

            # Handle batch dimension if present
            if images.dim() == 4 and images.shape[1] == 1:
                # (B, 1, H, W) -> (B, H, W)
                images = images.squeeze(1)

            saved_count = 0

            # Create individual visualizations
            if save_individual:
                for i in range(num_images):
                    img = images[i].numpy()
                    label = labels[i].numpy()
                    pred = predictions[i].numpy()

                    # Create overlay visualization
                    self._create_overlay(
                        img, label, pred, alpha,
                        output_path / f"overlay_{i:04d}.png"
                    )
                    saved_count += 1

                print(f"✓ Saved {saved_count} individual overlay images")

            # Create grid visualization
            if save_grid:
                grid_images = min(num_images, 16)  # Max 16 images in grid
                self._create_grid(
                    images[:grid_images],
                    labels[:grid_images],
                    predictions[:grid_images],
                    alpha,
                    output_path / "overlay_grid.png"
                )
                print(f"✓ Saved grid visualization with {grid_images} images")

            # Create legend
            self._create_legend(output_path / "legend.png")
            print(f"✓ Saved legend")

            print(f"{'='*60}\n")

            # Set output
            self.set_output_value('num_saved', saved_count)

            return True

        except Exception as e:
            print(f"Error in SegmentationOverlayNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_overlay(self, image, label, pred, alpha, save_path):
        """Create a single overlay visualization."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Normalize image to [0, 1] for display
        img_display = self._normalize_image(image)

        # 1. Original image
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Input Image', fontsize=12)
        axes[0].axis('off')

        # 2. Ground truth overlay
        axes[1].imshow(img_display, cmap='gray')
        gt_overlay = self._create_colored_mask(label, color='green')
        axes[1].imshow(gt_overlay, alpha=alpha)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')

        # 3. Prediction overlay
        axes[2].imshow(img_display, cmap='gray')
        pred_overlay = self._create_colored_mask(pred, color='red')
        axes[2].imshow(pred_overlay, alpha=alpha)
        axes[2].set_title('Prediction', fontsize=12)
        axes[2].axis('off')

        # 4. Combined overlay (GT=green, Pred=red, overlap=yellow)
        axes[3].imshow(img_display, cmap='gray')
        combined = self._create_comparison_overlay(label, pred)
        axes[3].imshow(combined, alpha=alpha)
        axes[3].set_title('Comparison (GT=Green, Pred=Red)', fontsize=12)
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_grid(self, images, labels, predictions, alpha, save_path):
        """Create a grid of overlay visualizations."""
        n = len(images)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            if idx < n:
                img = self._normalize_image(images[idx].numpy())
                label = labels[idx].numpy()
                pred = predictions[idx].numpy()

                # Show image with combined overlay
                ax.imshow(img, cmap='gray')
                combined = self._create_comparison_overlay(label, pred)
                ax.imshow(combined, alpha=alpha)
                ax.set_title(f'Image {idx}', fontsize=10)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle('Segmentation Results (Green=GT, Red=Pred, Yellow=Overlap)',
                     fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_legend(self, save_path):
        """Create a legend explaining the color coding."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.5, label='Ground Truth'),
            plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5, label='Prediction'),
            plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.5, label='Overlap (Correct)'),
        ]

        ax.legend(handles=legend_elements, loc='center', fontsize=14, frameon=True)
        ax.set_title('Color Legend', fontsize=16, pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _normalize_image(self, image):
        """Normalize image to [0, 1] range."""
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image

    def _create_colored_mask(self, mask, color='red'):
        """Create a colored mask overlay."""
        h, w = mask.shape
        colored = np.zeros((h, w, 4))

        if color == 'red':
            colored[mask > 0] = [1, 0, 0, 1]  # Red
        elif color == 'green':
            colored[mask > 0] = [0, 1, 0, 1]  # Green
        elif color == 'blue':
            colored[mask > 0] = [0, 0, 1, 1]  # Blue

        return colored

    def _create_comparison_overlay(self, gt, pred):
        """Create overlay showing GT, prediction, and overlap."""
        h, w = gt.shape
        overlay = np.zeros((h, w, 4))

        # Convert to binary masks
        gt_mask = (gt > 0)
        pred_mask = (pred > 0)

        # True positives: overlap (yellow)
        tp = gt_mask & pred_mask
        overlay[tp] = [1, 1, 0, 1]  # Yellow

        # False positives: pred only (red)
        fp = pred_mask & ~gt_mask
        overlay[fp] = [1, 0, 0, 1]  # Red

        # False negatives: gt only (green)
        fn = gt_mask & ~pred_mask
        overlay[fn] = [0, 1, 0, 1]  # Green

        return overlay

    def get_field_definitions(self):
        return {
            'output_dir': {
                'type': 'text',
                'label': 'Output Directory',
                'default': 'visualization_output'
            },
            'max_images': {
                'type': 'text',
                'label': 'Max Images to Save',
                'default': '10'
            },
            'alpha': {
                'type': 'text',
                'label': 'Overlay Transparency (0-1)',
                'default': '0.4'
            },
            'save_individual': {
                'type': 'choice',
                'label': 'Save Individual Images',
                'choices': ['True', 'False'],
                'default': 'True'
            },
            'save_grid': {
                'type': 'choice',
                'label': 'Save Grid Visualization',
                'choices': ['True', 'False'],
                'default': 'True'
            }
        }
