"""
Visualization nodes for medical images and results.

Provides visualization capabilities for images, segmentations, and metrics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ...core import BaseNode, NodeRegistry, DataType


@NodeRegistry.register('visualization', 'ImageViewer',
                      description='Visualize medical images')
class ImageViewerNode(BaseNode):
    """Visualize 2D/3D medical images."""

    def _setup_ports(self):
        self.add_input('image', DataType.TENSOR)
        self.add_input('segmentation', DataType.TENSOR, optional=True)

    def execute(self) -> bool:
        try:
            image = self.get_input_value('image')
            if image is None:
                return False

            segmentation = self.get_input_value('segmentation')

            # Convert to numpy
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if segmentation is not None and isinstance(segmentation, torch.Tensor):
                segmentation = segmentation.detach().cpu().numpy()

            # Handle batch dimension
            if image.ndim == 4:  # (B, C, H, W)
                image = image[0]  # Take first in batch
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
                image = image[0] if image.shape[0] == 1 else np.transpose(image, (1, 2, 0))

            # Create figure
            fig_size = (12, 6) if segmentation is not None else (6, 6)
            fig, axes = plt.subplots(1, 2 if segmentation is not None else 1, figsize=fig_size)

            if segmentation is None:
                axes = [axes]
            else:
                if segmentation.ndim == 4:
                    segmentation = segmentation[0]
                if segmentation.ndim == 3:
                    segmentation = segmentation[0]

            # Plot image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            # Plot segmentation if provided
            if segmentation is not None:
                axes[1].imshow(image, cmap='gray')
                axes[1].imshow(segmentation, alpha=0.5, cmap='jet')
                axes[1].set_title('Segmentation Overlay')
                axes[1].axis('off')

            plt.tight_layout()

            # Save if requested
            save_path = self.get_config('save_path', '')
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"Saved visualization to {save_path}")

            show = self.get_config('show', 'False') == 'True'
            if show:
                plt.show()
            else:
                plt.close()

            return True

        except Exception as e:
            print(f"Error in ImageViewerNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        return {
            'save_path': {'type': 'text', 'label': 'Save Path', 'default': ''},
            'show': {
                'type': 'choice',
                'label': 'Show Plot',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('visualization', 'MetricsPlotter',
                      description='Plot training metrics')
class MetricsPlotterNode(BaseNode):
    """Plot training metrics over epochs."""

    def _setup_ports(self):
        self.add_input('metrics', DataType.METRICS)

    def execute(self) -> bool:
        try:
            metrics = self.get_input_value('metrics')
            if metrics is None:
                return False

            # Plot loss curve
            if 'epoch_losses' in metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['epoch_losses'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.grid(True)

                save_path = self.get_config('save_path', '')
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)

                show = self.get_config('show', 'False') == 'True'
                if show:
                    plt.show()
                else:
                    plt.close()

            return True

        except Exception as e:
            print(f"Error in MetricsPlotterNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'save_path': {'type': 'text', 'label': 'Save Path', 'default': ''},
            'show': {
                'type': 'choice',
                'label': 'Show Plot',
                'choices': ['False', 'True'],
                'default': 'False'
            }
        }


@NodeRegistry.register('visualization', 'Print',
                      description='Print values to console')
class PrintNode(BaseNode):
    """Print node outputs to console for debugging."""

    def _setup_ports(self):
        self.add_input('input', DataType.ANY)

    def execute(self) -> bool:
        try:
            value = self.get_input_value('input')
            prefix = self.get_config('prefix', 'Output')

            print(f"\n{prefix}:")
            if isinstance(value, torch.Tensor):
                print(f"  Tensor shape: {value.shape}")
                print(f"  Tensor dtype: {value.dtype}")
                print(f"  Tensor device: {value.device}")
                print(f"  Tensor range: [{value.min():.4f}, {value.max():.4f}]")
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")

            return True

        except Exception as e:
            print(f"Error in PrintNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'prefix': {'type': 'text', 'label': 'Prefix', 'default': 'Output'}
        }
