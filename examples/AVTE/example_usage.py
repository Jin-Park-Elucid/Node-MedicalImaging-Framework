#!/usr/bin/env python3
"""
Example usage of AVTE 2D segmentation preprocessing and dataloader.

This script demonstrates:
1. How to use the preprocessing script
2. How to load preprocessed data
3. Basic training loop structure
4. Visualization of samples

Author: Medical Imaging Framework
Date: 2026-02-08
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the AVTE dataloader
from avte_dataloader import AVTE2DDataset, create_avte_dataloaders


def visualize_sample(image: torch.Tensor, label: torch.Tensor, title: str = "Sample"):
    """
    Visualize a sample with multi-channel input.

    Args:
        image: Tensor of shape (C, H, W)
        label: Tensor of shape (H, W)
        title: Plot title
    """
    num_channels = image.shape[0]

    # Create subplot grid
    fig, axes = plt.subplots(1, num_channels + 1, figsize=(4 * (num_channels + 1), 4))

    # Plot each input channel
    for i in range(num_channels):
        axes[i].imshow(image[i].cpu().numpy(), cmap='gray')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    # Plot label
    axes[num_channels].imshow(label.cpu().numpy(), cmap='jet')
    axes[num_channels].set_title('Segmentation')
    axes[num_channels].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


class SimpleUNet2D(nn.Module):
    """
    Simple 2D U-Net for demonstration.

    Note: This is a minimal implementation. For production,
    use a proper U-Net implementation.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        # Decoder
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)

        # Final convolution
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d2 = self.upsample(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final
        out = self.final(d1)
        return out


def example_training_loop(
    data_dir: str,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """
    Example training loop for AVTE 2D segmentation.

    Args:
        data_dir: Path to preprocessed data
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    print("="*60)
    print("EXAMPLE TRAINING LOOP FOR AVTE 2D SEGMENTATION")
    print("="*60)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_avte_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Get a sample to determine input channels
    sample_image, sample_label = next(iter(train_loader))
    num_channels = sample_image.shape[1]  # (B, C, H, W)
    num_classes = len(torch.unique(sample_label)) + 1  # +1 for background

    print(f"\nDataset info:")
    print(f"  Input channels: {num_channels}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Image shape: {sample_image.shape}")
    print(f"  Label shape: {sample_label.shape}")

    # Create model
    print("\nInitializing model...")
    model = SimpleUNet2D(in_channels=num_channels, num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.numel()
            train_correct += predicted.eq(labels).sum().item()

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        # Epoch statistics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.numel()
                    val_correct += predicted.eq(labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total

            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")

        print()

    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    return model


def main():
    """Main function demonstrating the preprocessing and training pipeline."""

    # Default data directory
    data_dir = "/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data"

    # Check if data exists
    if not Path(data_dir).exists():
        print(f"Error: Preprocessed data not found at {data_dir}")
        print("\nYou need to run the preprocessing script first:")
        print("\n  python preprocess_2d_slices.py \\")
        print("      --input_dir /data/avte_training/nnUNet_raw/Dataset006_model_9_4 \\")
        print("      --output_dir /home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data \\")
        print("      --window_size 2 \\")
        print("      --padding_mode replicate")
        print()
        sys.exit(1)

    # Example 1: Load and visualize samples
    print("="*60)
    print("EXAMPLE 1: LOADING AND VISUALIZING SAMPLES")
    print("="*60)

    train_dataset = AVTE2DDataset(data_dir, split='train')

    # Load a sample
    image, label = train_dataset[0]

    print(f"\nLoaded sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Unique labels: {torch.unique(label).tolist()}")

    # Get sample info
    print("\n" + train_dataset.get_sample_info(0))

    # Visualize (save to file instead of showing)
    output_dir = Path("./visualization_outputs")
    output_dir.mkdir(exist_ok=True)

    fig = visualize_sample(image, label, title="Training Sample 0")
    fig.savefig(output_dir / "sample_visualization.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_dir / 'sample_visualization.png'}")
    plt.close(fig)

    # Example 2: Run training loop
    print("\n")
    print("="*60)
    print("EXAMPLE 2: TRAINING LOOP")
    print("="*60)

    try:
        model = example_training_loop(
            data_dir=data_dir,
            num_epochs=2,  # Just 2 epochs for demonstration
            batch_size=4,
            learning_rate=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Save model
        model_path = output_dir / "example_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model to: {model_path}")

    except Exception as e:
        print(f"\nTraining example skipped due to error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Adjust hyperparameters for your use case")
    print("2. Implement proper U-Net or other segmentation architecture")
    print("3. Add data augmentation")
    print("4. Implement proper validation and metrics (Dice, IoU, etc.)")
    print("5. Add model checkpointing and early stopping")
    print()


if __name__ == '__main__':
    main()
