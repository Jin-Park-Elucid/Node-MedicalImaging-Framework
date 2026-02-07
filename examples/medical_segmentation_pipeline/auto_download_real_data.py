"""
Automated download of real medical imaging data (no user prompts).

Downloads MedMNIST dataset automatically.
"""

import os
import sys
import urllib.request
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import json

def download_file(url, destination):
    """Download file with progress."""
    print(f"Downloading from: {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                  end='', flush=True)

    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print(f"\n✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    """Main function - auto-download MedMNIST."""
    print("="*80)
    print("AUTO-DOWNLOADING REAL MEDICAL IMAGING DATA")
    print("="*80)

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    temp_dir = script_dir / "temp_download"

    # Create temp directory
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("\nDataset: MedMNIST - OrganAMNIST")
    print("Source: https://medmnist.com/")
    print("License: CC BY 4.0")
    print("Description: Real abdominal CT organ segmentation images\n")

    # Backup existing data if present
    if data_dir.exists():
        backup_dir = script_dir / "data_synthetic_backup"
        if (data_dir / "train" / "images").exists():
            print(f"Backing up existing data to {backup_dir.name}...")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(data_dir, backup_dir)
            print("✓ Backup complete")

    # Download MedMNIST
    print("\n" + "="*80)
    print("DOWNLOADING MEDMNIST DATASET")
    print("="*80 + "\n")

    npz_file = temp_dir / "organamnist.npz"

    # MedMNIST direct download URL
    url = "https://zenodo.org/record/6496656/files/organamnist.npz?download=1"

    print("Downloading OrganAMNIST dataset...")
    if not download_file(url, npz_file):
        print("\n✗ Failed to download dataset")
        print("Please check your internet connection and try again.")
        shutil.rmtree(temp_dir)
        return False

    # Load dataset
    print("\n" + "="*80)
    print("PROCESSING DATASET")
    print("="*80 + "\n")

    print("Loading NPZ file...")
    data = np.load(npz_file)

    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    print(f"✓ Loaded dataset:")
    print(f"  Training shape: {train_images.shape}")
    print(f"  Test shape: {test_images.shape}")

    # Create output directories
    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"
    test_img_dir = data_dir / "test" / "images"
    test_mask_dir = data_dir / "test" / "masks"

    for dir_path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Convert training data
    print("\nConverting training data to PNG...")
    num_train = min(len(train_images), 100)  # Use first 100 samples

    for i in range(num_train):
        # Get image and label
        img = train_images[i]
        label_class = train_labels[i].item()  # Get class index

        # Convert to grayscale if needed
        if img.ndim == 3 and img.shape[-1] == 3:
            # RGB to grayscale
            img_gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img_gray = img.squeeze()
        else:
            img_gray = img

        # Normalize if needed
        if img_gray.max() > 255 or img_gray.min() < 0:
            img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)

        # Create binary mask: foreground (class > 0) vs background (class == 0)
        # Since labels are class indices, create a simple binary mask
        # where class 0 = background, any other class = foreground
        mask = np.ones_like(img_gray, dtype=np.uint8) * 255 if label_class > 0 else np.zeros_like(img_gray, dtype=np.uint8)

        # Save
        Image.fromarray(img_gray).save(train_img_dir / f"image_{i:04d}.png")
        Image.fromarray(mask).save(train_mask_dir / f"mask_{i:04d}.png")

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_train} training samples")

    # Convert test data
    print("\nConverting test data to PNG...")
    num_test = min(len(test_images), 30)  # Use first 30 samples

    for i in range(num_test):
        img = test_images[i]
        label_class = test_labels[i].item()

        # Same processing
        if img.ndim == 3 and img.shape[-1] == 3:
            img_gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img_gray = img.squeeze()
        else:
            img_gray = img

        if img_gray.max() > 255 or img_gray.min() < 0:
            img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min()) * 255).astype(np.uint8)

        # Create binary mask
        mask = np.ones_like(img_gray, dtype=np.uint8) * 255 if label_class > 0 else np.zeros_like(img_gray, dtype=np.uint8)

        Image.fromarray(img_gray).save(test_img_dir / f"image_{i:04d}.png")
        Image.fromarray(mask).save(test_mask_dir / f"mask_{i:04d}.png")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_test} test samples")

    # Create dataset info
    dataset_info = {
        "name": "MedMNIST OrganAMNIST - Real Medical Imaging Data",
        "description": "Real abdominal CT scans with organ segmentation masks",
        "source": "https://medmnist.com/",
        "dataset": "OrganAMNIST",
        "num_train": num_train,
        "num_test": num_test,
        "image_size": [28, 28],
        "num_classes": 2,
        "classes": ["background", "organ"],
        "license": "CC BY 4.0",
        "citation": "Jiancheng Yang et al. 'MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification' Scientific Data, 2023",
        "note": "This is REAL medical imaging data from CT scans"
    }

    with open(data_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print("\n" + "="*80)
    print("✓ DOWNLOAD AND CONVERSION COMPLETE!")
    print("="*80)

    print(f"\nDataset location: {data_dir.absolute()}")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Image size: 28x28 pixels")

    print("\nDataset Information:")
    print("  • Real CT scan slices from MedMNIST")
    print("  • Abdominal organ segmentation")
    print("  • License: CC BY 4.0 (freely usable)")
    print("  • Small size (28x28) - good for quick training")

    print("\nNext steps:")
    print("  1. Inspect data: python inspect_dataset.py")
    print("  2. Train model: python train_pipeline.py")
    print("  3. Test model: python test_pipeline.py")
    print("  4. Or use GUI to train and test")

    print("\nNote: Image size is 28x28. Update network config if needed:")
    print("  • Models should handle 28x28 input (default is 256x256)")
    print("  • Or resize images to 256x256 in dataloader")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
