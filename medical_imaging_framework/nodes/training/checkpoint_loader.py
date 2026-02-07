"""
Checkpoint loading node for restoring trained models.

Allows loading saved model weights from training checkpoints.
"""

import torch
from pathlib import Path
from ...core import BaseNode, NodeRegistry, DataType


@NodeRegistry.register('training', 'CheckpointLoader',
                      description='Load model weights from checkpoint')
class CheckpointLoaderNode(BaseNode):
    """
    Load model weights from a saved checkpoint.

    This node takes a model and a checkpoint file path, loads the
    saved weights, and outputs the model with loaded weights.
    """

    def _setup_ports(self):
        self.add_input('model', DataType.MODEL)
        self.add_output('model', DataType.MODEL)
        self.add_output('checkpoint_info', DataType.METRICS)

    def execute(self) -> bool:
        try:
            model = self.get_input_value('model')
            if model is None:
                print("CheckpointLoader: model input is None")
                return False

            # Get checkpoint path
            checkpoint_path = self.get_config('checkpoint_path', '')

            if not checkpoint_path:
                print("CheckpointLoader: No checkpoint path specified")
                print("Please set 'checkpoint_path' in the node configuration")
                return False

            checkpoint_file = Path(checkpoint_path)

            if not checkpoint_file.exists():
                print(f"CheckpointLoader: Checkpoint file not found: {checkpoint_file}")
                return False

            # Load checkpoint
            print(f"Loading checkpoint from: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            # Check if it's a full checkpoint or just state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                model.load_state_dict(checkpoint['model_state_dict'])

                # Extract info
                checkpoint_info = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'loss': checkpoint.get('loss', 'unknown'),
                    'checkpoint_path': str(checkpoint_file)
                }

                print(f"✓ Loaded checkpoint from epoch {checkpoint_info['epoch']}")
                if checkpoint_info['loss'] != 'unknown':
                    print(f"  Loss at checkpoint: {checkpoint_info['loss']:.4f}")

            elif isinstance(checkpoint, dict):
                # Just state dict
                model.load_state_dict(checkpoint)
                checkpoint_info = {
                    'checkpoint_path': str(checkpoint_file)
                }
                print(f"✓ Loaded model state dict from: {checkpoint_file.name}")
            else:
                print(f"CheckpointLoader: Unexpected checkpoint format")
                return False

            # Set outputs
            self.set_output_value('model', model)
            self.set_output_value('checkpoint_info', checkpoint_info)

            return True

        except Exception as e:
            print(f"Error in CheckpointLoaderNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        return {
            'checkpoint_path': {
                'type': 'text',
                'label': 'Checkpoint Path',
                'default': 'checkpoints/best_model.pt'
            }
        }
