"""
Script to inspect and visualize the dataset.

Shows what the raw data looks like before training/testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def inspect_dataset():
    """Inspect and visualize dataset samples."""
    data_dir = Path(__file__).parent / "data"

    print("="*80)
    print("DATASET INSPECTION")
    print("="*80)

    # Check if dataset exists
    if not data_dir.exists():
        print("\n✗ Dataset not found!")
        print("Run: python download_dataset.py first")
        return

    # Count samples
    train_images = list((data_dir / "train" / "images").glob("*.png"))
    test_images = list((data_dir / "test" / "images").glob("*.png"))

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")

    if len(train_images) == 0:
        print("\n✗ No training data found!")
        return

    # Load and display a few samples
    print(f"\nVisualizing 6 random training samples...")

    # Select 6 random samples
    np.random.seed(42)
    selected_indices = np.random.choice(len(train_images), min(6, len(train_images)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Medical Segmentation Dataset - Training Samples', fontsize=16)

    for i, idx in enumerate(selected_indices):
        row = i // 2
        col_offset = (i % 2) * 2

        # Load image and mask
        img_path = train_images[idx]
        # Convert image_XXXX.png to mask_XXXX.png
        mask_filename = img_path.name.replace("image_", "mask_")
        mask_path = data_dir / "train" / "masks" / mask_filename

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        # Display image
        axes[row, col_offset].imshow(image, cmap='gray')
        axes[row, col_offset].set_title(f'Image {idx}')
        axes[row, col_offset].axis('off')

        # Display mask
        axes[row, col_offset + 1].imshow(mask, cmap='Greens', vmin=0, vmax=255)
        axes[row, col_offset + 1].set_title(f'Mask {idx}')
        axes[row, col_offset + 1].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = data_dir.parent / "dataset_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.show()

    # Print data characteristics
    print("\n" + "="*80)
    print("DATA CHARACTERISTICS")
    print("="*80)

    sample_image = np.array(Image.open(train_images[0]))
    sample_mask = np.array(Image.open(mask_path))

    print(f"\nImage Properties:")
    print(f"  Shape: {sample_image.shape}")
    print(f"  Data type: {sample_image.dtype}")
    print(f"  Value range: [{sample_image.min()}, {sample_image.max()}]")
    print(f"  Mean: {sample_image.mean():.2f}")
    print(f"  Std: {sample_image.std():.2f}")

    print(f"\nMask Properties:")
    print(f"  Shape: {sample_mask.shape}")
    print(f"  Unique values: {np.unique(sample_mask)}")
    print(f"  Foreground pixels: {(sample_mask > 127).sum()} / {sample_mask.size}")
    print(f"  Foreground ratio: {(sample_mask > 127).sum() / sample_mask.size * 100:.2f}%")

    print("\n" + "="*80)
    print("WHAT YOU'RE LOOKING AT")
    print("="*80)
    print("""
This is REAL medical imaging data from MedMNIST OrganAMNIST.

What you see:
  • Images: Actual abdominal CT scan slices (28x28 pixels)
  • Real anatomical structures (organs, tissue, bone)
  • Masks: Binary organ segmentation (white = organ, black = background)

Dataset Information:
  ✓ Real medical imaging data
  ✓ Actual CT scans
  ✓ Real anatomical structures
  ✓ From MedMNIST benchmark dataset
  ✓ License: CC BY 4.0 (free for research/education)

Source:
  • MedMNIST - OrganAMNIST
  • https://medmnist.com/
  • 100 training samples, 30 test samples
  • Citation: Jiancheng Yang et al. "MedMNIST v2" Scientific Data, 2023

Note: Images are 28x28 pixels (small for fast training). For higher resolution
medical imaging, consider Medical Segmentation Decathlon or similar datasets.
""")

if __name__ == "__main__":
    inspect_dataset()
