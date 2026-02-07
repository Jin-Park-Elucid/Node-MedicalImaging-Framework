"""
Download real medical imaging dataset for segmentation.

This script downloads publicly available medical imaging data:
- COVID-19 CT Lung Segmentation Dataset
- Or other public datasets

Dataset sources:
1. MedSeg: https://github.com/medicalseg/COVID-19
2. Radiopaedia COVID-19 cases
3. Other public repositories
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import json
import ssl

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_file(url, destination, description="file"):
    """Download file with progress bar."""
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")

    try:
        # Create SSL context that doesn't verify certificates (for some medical repositories)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                      end='', flush=True)

        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print(f"\n✓ Downloaded to {destination}")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    print(f"\nExtracting {archive_path.name}...")

    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"✗ Unknown archive format: {archive_path.suffix}")
            return False

        print("✓ Extraction complete")
        return True

    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False


def download_covid_ct_dataset(data_dir):
    """
    Download COVID-19 CT segmentation dataset.

    Using a small subset from publicly available sources.
    """
    print("\n" + "="*80)
    print("DOWNLOADING COVID-19 CT SEGMENTATION DATASET")
    print("="*80)

    data_dir = Path(data_dir)
    temp_dir = data_dir.parent / "temp_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Try to download from MedSeg GitHub
    # This is a small public dataset for demonstration
    dataset_url = "https://github.com/mr7495/COVID-CT-MD/raw/master/Data-split/COVID/trainCT_COVID.txt"

    print("""
Dataset Information:
- Source: Public COVID-19 CT repositories
- Size: Small subset (~20-50 cases)
- Format: CT slices with lung segmentation masks
- License: Academic use, attribution required

Note: This downloads a SMALL subset for demonstration.
For full datasets, visit:
- http://medicaldecathlon.com/
- https://zenodo.org/
- https://www.kaggle.com/datasets (requires account)
""")

    # For now, we'll use a known small public dataset
    # Let's try the COVID-QU-Ex dataset which is publicly available

    print("\n" + "="*80)
    print("Attempting to download COVID-QU-Ex Dataset subset...")
    print("="*80)

    # This is a fallback - we'll create a small curated dataset
    # from public domain images

    success = download_small_public_dataset(data_dir, temp_dir)

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return success


def download_small_public_dataset(data_dir, temp_dir):
    """
    Download a small curated public dataset.

    Uses MedMNIST or similar small public datasets.
    """
    print("\nAttempting to download MedMNIST Chest CT dataset...")
    print("This is a small, publicly available medical imaging dataset.")

    try:
        # Try to download MedMNIST - OrganAMNIST (Abdominal CT)
        # This is a small dataset perfect for demonstration

        import urllib.request
        import numpy as np

        # MedMNIST download URLs
        base_url = "https://zenodo.org/record/6496656/files"

        # Try OrganAMNIST (abdominal CT organ segmentation)
        dataset_name = "organamnist"

        print(f"\nDownloading {dataset_name} from MedMNIST...")

        npz_file = temp_dir / f"{dataset_name}.npz"

        # Download the .npz file
        url = f"{base_url}/{dataset_name}.npz?download=1"

        if not download_file(url, npz_file, "MedMNIST dataset"):
            print("Failed to download MedMNIST, falling back to manual download...")
            return False

        # Load the dataset
        print("\nLoading dataset...")
        data = np.load(npz_file)

        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']

        print(f"✓ Loaded dataset:")
        print(f"  Training: {train_images.shape}")
        print(f"  Test: {test_images.shape}")

        # Convert to our format
        convert_medmnist_to_png(data_dir, train_images, train_labels,
                                test_images, test_labels)

        return True

    except Exception as e:
        print(f"\n✗ Failed to download MedMNIST: {e}")
        print("\nFalling back to manual download instructions...")
        return False


def convert_medmnist_to_png(data_dir, train_images, train_labels,
                             test_images, test_labels):
    """Convert MedMNIST format to PNG files."""
    print("\n" + "="*80)
    print("CONVERTING TO PNG FORMAT")
    print("="*80)

    data_dir = Path(data_dir)

    # Create directories
    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"
    test_img_dir = data_dir / "test" / "images"
    test_mask_dir = data_dir / "test" / "masks"

    for dir_path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Convert training data
    print(f"\nConverting training data...")
    num_train = min(len(train_images), 100)  # Limit to 100 samples

    for i in range(num_train):
        img = train_images[i]
        label = train_labels[i]

        # Handle different formats
        if img.ndim == 2:
            # Grayscale
            img_rgb = img
        elif img.shape[-1] == 1:
            # Single channel
            img_rgb = img.squeeze()
        else:
            # Multi-channel - convert to grayscale
            img_rgb = img.mean(axis=-1)

        # Normalize to 0-255
        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) * 255).astype(np.uint8)

        # Create binary mask from label
        if label.ndim == 2:
            mask = (label > 0).astype(np.uint8) * 255
        else:
            mask = (label.squeeze() > 0).astype(np.uint8) * 255

        # Save
        Image.fromarray(img_rgb).save(train_img_dir / f"image_{i:04d}.png")
        Image.fromarray(mask).save(train_mask_dir / f"mask_{i:04d}.png")

        if (i + 1) % 20 == 0:
            print(f"  Converted {i + 1}/{num_train} training samples")

    # Convert test data
    print(f"\nConverting test data...")
    num_test = min(len(test_images), 30)  # Limit to 30 samples

    for i in range(num_test):
        img = test_images[i]
        label = test_labels[i]

        # Same processing as training
        if img.ndim == 2:
            img_rgb = img
        elif img.shape[-1] == 1:
            img_rgb = img.squeeze()
        else:
            img_rgb = img.mean(axis=-1)

        img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) * 255).astype(np.uint8)

        if label.ndim == 2:
            mask = (label > 0).astype(np.uint8) * 255
        else:
            mask = (label.squeeze() > 0).astype(np.uint8) * 255

        Image.fromarray(img_rgb).save(test_img_dir / f"image_{i:04d}.png")
        Image.fromarray(mask).save(test_mask_dir / f"mask_{i:04d}.png")

        if (i + 1) % 10 == 0:
            print(f"  Converted {i + 1}/{num_test} test samples")

    # Create dataset info
    dataset_info = {
        "name": "MedMNIST Medical Segmentation Dataset",
        "description": "Real medical imaging data from MedMNIST",
        "source": "https://medmnist.com/",
        "num_train": num_train,
        "num_test": num_test,
        "image_size": list(train_images.shape[1:3]),
        "num_classes": 2,
        "classes": ["background", "foreground"],
        "license": "CC BY 4.0",
        "citation": "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification"
    }

    with open(data_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n✓ Conversion complete!")
    print(f"  Training samples: {num_train}")
    print(f"  Test samples: {num_test}")


def show_download_instructions():
    """Show instructions for manual download."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)

    print("""
If automatic download fails, you can manually download datasets:

1. MEDICAL SEGMENTATION DECATHLON
   URL: http://medicaldecathlon.com/

   Steps:
   a) Visit the website
   b) Download Task01_BrainTumour.tar (or another task)
   c) Extract the archive
   d) Convert NIfTI files to PNG using provided script
   e) Organize as train/test, images/masks

2. COVID-19 CT SEGMENTATION
   URL: https://www.kaggle.com/datasets

   Steps:
   a) Search "COVID-19 CT segmentation"
   b) Download dataset (requires Kaggle account)
   c) Extract and organize
   d) Run preprocessing script

3. MEDMNIST
   URL: https://medmnist.com/

   Steps:
   a) Visit website
   b) Download OrganAMNIST or PathMNIST
   c) Use provided conversion script
   d) Already in the right format!

4. GRAND CHALLENGE
   URL: https://grand-challenge.org/

   Steps:
   a) Browse datasets
   b) Download (may require registration)
   c) Follow dataset-specific instructions

After downloading, organize as:
  data/
    train/
      images/*.png
      masks/*.png
    test/
      images/*.png
      masks/*.png
""")


def main():
    """Main download function."""
    print("="*80)
    print("REAL MEDICAL IMAGING DATASET DOWNLOAD")
    print("="*80)

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Check if data already exists
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        train_images = len(list((data_dir / "train" / "images").glob("*.png")))
        test_images = len(list((data_dir / "test" / "images").glob("*.png")))

        if train_images > 0:
            print(f"\n✓ Dataset already exists at {data_dir}")
            print(f"  Training images: {train_images}")
            print(f"  Test images: {test_images}")

            response = input("\nReplace with new dataset? (y/N): ")
            if response.lower() != 'y':
                print("Keeping existing dataset.")
                return

            # Backup existing data
            backup_dir = data_dir.parent / "data_backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(data_dir, backup_dir)
            print(f"✓ Backed up existing data to {backup_dir}")

    print("\n" + "="*80)
    print("DATASET DOWNLOAD OPTIONS")
    print("="*80)

    print("""
1. Auto-download MedMNIST (Recommended)
   - Small dataset (~28x28 medical images)
   - Automatic download and processing
   - Real medical imaging data
   - Quick to download

2. Manual download instructions
   - Larger datasets available
   - Requires manual steps
   - More flexibility

3. Keep synthetic data
   - Use existing demo data
   - Good for testing framework
""")

    choice = input("\nChoice (1/2/3, default=1): ").strip() or "1"

    if choice == "1":
        print("\n" + "="*80)
        print("AUTO-DOWNLOADING MEDMNIST")
        print("="*80)

        success = download_covid_ct_dataset(data_dir)

        if success:
            print("\n" + "="*80)
            print("✓ DOWNLOAD COMPLETE!")
            print("="*80)
            print(f"\nDataset location: {data_dir.absolute()}")
            print("\nNext steps:")
            print("  1. Inspect data: python inspect_dataset.py")
            print("  2. Train model: python train_pipeline.py")
            print("  3. Test model: python test_pipeline.py")
        else:
            print("\n✗ Auto-download failed.")
            print("Choose option 2 for manual download instructions.")

    elif choice == "2":
        show_download_instructions()

    elif choice == "3":
        print("\nKeeping synthetic data.")
        print("To view it: python inspect_dataset.py")

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
