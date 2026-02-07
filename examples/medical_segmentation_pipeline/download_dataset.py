"""
Download public medical imaging dataset for segmentation.

This script downloads a small subset of publicly available medical images
for demonstrating the segmentation pipeline.

Dataset: COVID-19 CT Segmentation or similar public dataset
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_file(url, destination):
    """Download file with progress."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False


def create_synthetic_dataset(data_dir, num_train=50, num_test=20):
    """
    Create synthetic medical imaging dataset for demonstration.

    This creates simple 2D images that simulate medical imaging data
    with segmentation masks (e.g., simulating organ/lesion segmentation).
    """
    print("\nCreating synthetic medical imaging dataset...")
    print("(In production, replace this with real medical images)")

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Create directories
    for split in [train_dir, test_dir]:
        (split / "images").mkdir(parents=True, exist_ok=True)
        (split / "masks").mkdir(parents=True, exist_ok=True)

    # Generate training data
    print(f"\nGenerating {num_train} training samples...")
    for i in range(num_train):
        # Create synthetic medical image (simulating CT/MRI)
        # 256x256 grayscale image
        image = np.random.randn(256, 256) * 30 + 128

        # Add synthetic "organs" or "lesions"
        num_objects = np.random.randint(1, 4)
        mask = np.zeros((256, 256), dtype=np.uint8)

        for obj in range(num_objects):
            # Random circular object
            center_x = np.random.randint(50, 206)
            center_y = np.random.randint(50, 206)
            radius = np.random.randint(10, 40)

            # Create circular region
            y, x = np.ogrid[:256, :256]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

            # Add to image with different intensity
            image[circle_mask] += np.random.randn() * 40 + 60
            mask[circle_mask] = 1  # Binary segmentation

        # Normalize image
        image = np.clip(image, 0, 255).astype(np.uint8)

        # Save image and mask
        img_path = train_dir / "images" / f"image_{i:04d}.png"
        mask_path = train_dir / "masks" / f"mask_{i:04d}.png"

        Image.fromarray(image).save(img_path)
        Image.fromarray(mask * 255).save(mask_path)  # Scale mask for visibility

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_train} training samples")

    # Generate test data
    print(f"\nGenerating {num_test} test samples...")
    for i in range(num_test):
        # Similar process for test data
        image = np.random.randn(256, 256) * 30 + 128
        num_objects = np.random.randint(1, 4)
        mask = np.zeros((256, 256), dtype=np.uint8)

        for obj in range(num_objects):
            center_x = np.random.randint(50, 206)
            center_y = np.random.randint(50, 206)
            radius = np.random.randint(10, 40)

            y, x = np.ogrid[:256, :256]
            circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

            image[circle_mask] += np.random.randn() * 40 + 60
            mask[circle_mask] = 1

        image = np.clip(image, 0, 255).astype(np.uint8)

        img_path = test_dir / "images" / f"image_{i:04d}.png"
        mask_path = test_dir / "masks" / f"mask_{i:04d}.png"

        Image.fromarray(image).save(img_path)
        Image.fromarray(mask * 255).save(mask_path)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_test} test samples")

    # Create dataset info file
    dataset_info = {
        "name": "Synthetic Medical Segmentation Dataset",
        "description": "Synthetic dataset for demonstration purposes",
        "num_train": num_train,
        "num_test": num_test,
        "image_size": [256, 256],
        "num_classes": 2,
        "classes": ["background", "foreground"],
        "note": "Replace with real medical imaging data for production use"
    }

    with open(data_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n✓ Dataset creation complete!")
    print(f"  Training samples: {num_train}")
    print(f"  Test samples: {num_test}")
    print(f"  Location: {data_dir}")

    return True


def download_real_dataset(data_dir):
    """
    Download real medical imaging dataset.

    Options:
    1. COVID-19 CT Segmentation Dataset
    2. Medical Segmentation Decathlon subset
    3. Other publicly available datasets

    For this demo, we'll provide instructions for manual download
    or use synthetic data.
    """
    print("\n" + "="*80)
    print("DOWNLOADING REAL MEDICAL IMAGING DATASET")
    print("="*80)

    print("""
For real medical imaging data, you can download from:

1. Medical Segmentation Decathlon:
   http://medicaldecathlon.com/
   - Download Task01_BrainTumour or other tasks
   - Extract to: examples/medical_segmentation_pipeline/data/

2. COVID-19 CT Segmentation:
   https://www.kaggle.com/datasets/
   - Search for "COVID-19 CT segmentation"
   - Download and extract

3. Grand Challenge:
   https://grand-challenge.org/
   - Many public medical imaging datasets

For this demo, we'll create synthetic data that simulates medical images.
""")

    return False


def main():
    """Main download function."""
    print("="*80)
    print("MEDICAL IMAGING DATASET SETUP")
    print("="*80)

    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Check if data already exists
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        print(f"\n✓ Dataset already exists at {data_dir}")

        # Count files
        train_images = len(list((data_dir / "train" / "images").glob("*.png")))
        test_images = len(list((data_dir / "test" / "images").glob("*.png")))

        print(f"  Training images: {train_images}")
        print(f"  Test images: {test_images}")

        response = input("\nRe-create dataset? (y/N): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return

    # Try to download real dataset or create synthetic
    print("\nDataset Options:")
    print("1. Create synthetic dataset (demo)")
    print("2. Download real dataset (manual)")

    choice = input("\nChoice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        # Create synthetic dataset
        num_train = int(input("Number of training samples (default=50): ").strip() or "50")
        num_test = int(input("Number of test samples (default=20): ").strip() or "20")

        create_synthetic_dataset(data_dir, num_train, num_test)

    elif choice == "2":
        download_real_dataset(data_dir)
        print("\nAfter downloading real data, organize it as:")
        print("  data/")
        print("    train/")
        print("      images/  <- training images")
        print("      masks/   <- training masks")
        print("    test/")
        print("      images/  <- test images")
        print("      masks/   <- test masks")

    print("\n" + "="*80)
    print("✓ Dataset setup complete!")
    print("="*80)
    print(f"\nData location: {data_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python examples/medical_segmentation_pipeline/train_pipeline.py")
    print("  2. Run: python examples/medical_segmentation_pipeline/test_pipeline.py")


if __name__ == "__main__":
    main()
