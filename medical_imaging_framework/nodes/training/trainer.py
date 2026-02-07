"""
Training nodes for deep learning models.

Provides training loop, optimization, and metric tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from pathlib import Path
from ...core import BaseNode, NodeRegistry, DataType


@NodeRegistry.register('training', 'Trainer',
                      description='Train a model with dataloader')
class TrainerNode(BaseNode):
    """
    Training node for deep learning models.

    Handles training loop, optimization, and loss computation.
    """

    def _setup_ports(self):
        self.add_input('model', DataType.MODEL)
        self.add_input('dataloader', DataType.BATCH)
        self.add_input('loss_fn', DataType.LOSS, optional=True)
        self.add_input('optimizer', DataType.OPTIMIZER, optional=True)
        self.add_output('trained_model', DataType.MODEL)
        self.add_output('metrics', DataType.METRICS)

    def execute(self) -> bool:
        try:
            model = self.get_input_value('model')
            dataloader = self.get_input_value('dataloader')

            if model is None or dataloader is None:
                print("Model or dataloader not provided")
                return False

            # Get loss function
            loss_fn = self.get_input_value('loss_fn')
            if loss_fn is None:
                loss_type = self.get_config('loss_type', 'cross_entropy')
                if loss_type == 'cross_entropy':
                    loss_fn = nn.CrossEntropyLoss()
                elif loss_type == 'dice':
                    loss_fn = DiceLoss()
                else:
                    loss_fn = nn.MSELoss()

            # Get optimizer
            optimizer = self.get_input_value('optimizer')
            if optimizer is None:
                lr = float(self.get_config('learning_rate', 0.001))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training settings
            num_epochs = int(self.get_config('num_epochs', 10))
            device = self._device

            # Checkpoint settings
            checkpoint_dir = self.get_config('checkpoint_dir', '')
            save_every_n_epochs = int(self.get_config('save_every_n_epochs', 5))

            # Create checkpoint directory if specified
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Checkpoints will be saved to: {checkpoint_path}")

            model.to(device)
            model.train()

            # Training metrics
            epoch_losses = []
            best_loss = float('inf')

            # Training loop
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, (images, labels) in enumerate(dataloader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)

                    # Compute loss
                    loss = loss_fn(outputs, labels)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                epoch_losses.append(avg_loss)

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

                # Save checkpoint
                if checkpoint_dir:
                    # Save periodic checkpoint
                    if (epoch + 1) % save_every_n_epochs == 0:
                        checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt"
                        self._save_checkpoint(
                            checkpoint_file, model, optimizer, epoch + 1,
                            avg_loss, epoch_losses
                        )
                        print(f"  ✓ Saved checkpoint: {checkpoint_file.name}")

                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_checkpoint = checkpoint_path / "best_model.pt"
                        self._save_checkpoint(
                            best_checkpoint, model, optimizer, epoch + 1,
                            avg_loss, epoch_losses
                        )
                        print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")

            # Save final checkpoint
            if checkpoint_dir:
                final_checkpoint = checkpoint_path / "final_model.pt"
                self._save_checkpoint(
                    final_checkpoint, model, optimizer, num_epochs,
                    epoch_losses[-1] if epoch_losses else 0.0, epoch_losses
                )
                print(f"\n✓ Training complete. Final model saved to: {final_checkpoint}")

            # Save outputs
            self.set_output_value('trained_model', model)
            self.set_output_value('metrics', {
                'epoch_losses': epoch_losses,
                'final_loss': epoch_losses[-1] if epoch_losses else 0.0
            })

            return True

        except Exception as e:
            print(f"Error in TrainerNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_checkpoint(self, filepath, model, optimizer, epoch, loss, epoch_losses):
        """Save training checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch_losses': epoch_losses,
            }
            torch.save(checkpoint, filepath)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def get_field_definitions(self):
        return {
            'num_epochs': {'type': 'text', 'label': 'Number of Epochs', 'default': '10'},
            'learning_rate': {'type': 'text', 'label': 'Learning Rate', 'default': '0.001'},
            'loss_type': {
                'type': 'choice',
                'label': 'Loss Function',
                'choices': ['cross_entropy', 'dice', 'mse'],
                'default': 'cross_entropy'
            },
            'checkpoint_dir': {
                'type': 'text',
                'label': 'Checkpoint Directory',
                'default': 'checkpoints'
            },
            'save_every_n_epochs': {
                'type': 'text',
                'label': 'Save Every N Epochs',
                'default': '5'
            }
        }


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.softmax(predictions, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
        targets_one_hot = targets_one_hot.permute(0, -1, *range(1, len(targets.shape)))

        intersection = (predictions * targets_one_hot).sum(dim=(2, 3, 4) if predictions.dim() == 5 else (2, 3))
        union = predictions.sum(dim=(2, 3, 4) if predictions.dim() == 5 else (2, 3)) + \
                targets_one_hot.sum(dim=(2, 3, 4) if targets_one_hot.dim() == 5 else (2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


@NodeRegistry.register('training', 'LossFunction',
                      description='Define a loss function')
class LossFunctionNode(BaseNode):
    """Loss function node."""

    def _setup_ports(self):
        self.add_output('loss_fn', DataType.LOSS)

    def execute(self) -> bool:
        try:
            loss_type = self.get_config('loss_type', 'cross_entropy')

            if loss_type == 'cross_entropy':
                loss_fn = nn.CrossEntropyLoss()
            elif loss_type == 'dice':
                loss_fn = DiceLoss()
            elif loss_type == 'mse':
                loss_fn = nn.MSELoss()
            elif loss_type == 'bce':
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()

            self.set_output_value('loss_fn', loss_fn)
            return True

        except Exception as e:
            print(f"Error in LossFunctionNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'loss_type': {
                'type': 'choice',
                'label': 'Loss Type',
                'choices': ['cross_entropy', 'dice', 'mse', 'bce'],
                'default': 'cross_entropy'
            }
        }


@NodeRegistry.register('training', 'Optimizer',
                      description='Define an optimizer')
class OptimizerNode(BaseNode):
    """Optimizer node."""

    def _setup_ports(self):
        self.add_input('model', DataType.MODEL)
        self.add_output('optimizer', DataType.OPTIMIZER)

    def execute(self) -> bool:
        try:
            model = self.get_input_value('model')
            if model is None:
                print("OptimizerNode: model input is None")
                return False

            optimizer_type = self.get_config('optimizer_type', 'adam').lower()
            lr = float(self.get_config('learning_rate', 0.001))
            weight_decay = float(self.get_config('weight_decay', 0.0))

            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_type == 'sgd':
                momentum = float(self.get_config('momentum', 0.9))
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            self.set_output_value('optimizer', optimizer)
            return True

        except Exception as e:
            print(f"Error in OptimizerNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        return {
            'optimizer_type': {
                'type': 'choice',
                'label': 'Optimizer Type',
                'choices': ['adam', 'sgd', 'adamw'],
                'default': 'adam'
            },
            'learning_rate': {'type': 'text', 'label': 'Learning Rate', 'default': '0.001'},
            'weight_decay': {'type': 'text', 'label': 'Weight Decay', 'default': '0.0'},
            'momentum': {'type': 'text', 'label': 'Momentum (SGD)', 'default': '0.9'}
        }
