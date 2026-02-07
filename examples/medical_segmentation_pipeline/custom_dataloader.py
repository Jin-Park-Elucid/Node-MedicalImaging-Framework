"""
Custom dataloader for medical image segmentation dataset.

This module provides a custom dataloader specifically designed for
the medical segmentation pipeline example.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType


class MedicalSegmentationDataset(Dataset):
    """
    Dataset for medical image segmentation.

    Loads images and corresponding segmentation masks.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform=None,
        target_transform=None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing train/test folders
            split: 'train' or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to masks
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Get image and mask paths
        self.image_dir = self.data_dir / split / "images"
        self.mask_dir = self.data_dir / split / "masks"

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Get list of image files
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.png")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        if len(self.image_files) != len(self.mask_files):
            print(f"Warning: Number of images ({len(self.image_files)}) "
                  f"!= number of masks ({len(self.mask_files)})")

        print(f"Loaded {len(self.image_files)} {split} samples from {self.data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get image and mask at index."""
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('L')  # Grayscale
        image = np.array(image, dtype=np.float32)

        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.float32)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Binarize mask (0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Add channel dimension: (H, W) -> (1, H, W)
        image = image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long().squeeze(0)  # Remove channel for target

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


@NodeRegistry.register('data', 'MedicalSegmentationLoader',
                      description='Load medical segmentation dataset')
class MedicalSegmentationLoaderNode(BaseNode):
    """
    Custom dataloader node for medical segmentation dataset.

    Loads the medical imaging dataset created by download_dataset.py
    and creates PyTorch DataLoader with proper batching.
    """

    def _setup_ports(self):
        self.add_output('train_loader', DataType.BATCH)
        self.add_output('test_loader', DataType.BATCH)
        self.add_output('num_train', DataType.ANY)
        self.add_output('num_test', DataType.ANY)

    def execute(self) -> bool:
        try:
            # Get configuration (convert strings to proper types)
            data_dir = self.get_config('data_dir', 'examples/medical_segmentation_pipeline/data')
            batch_size = int(self.get_config('batch_size', 4))
            num_workers = int(self.get_config('num_workers', 0))
            shuffle_train_str = self.get_config('shuffle_train', True)
            # Convert string 'True'/'False' to boolean
            if isinstance(shuffle_train_str, str):
                shuffle_train = shuffle_train_str.lower() in ['true', '1', 'yes']
            else:
                shuffle_train = bool(shuffle_train_str)

            # Create datasets
            train_dataset = MedicalSegmentationDataset(
                data_dir=data_dir,
                split='train'
            )

            test_dataset = MedicalSegmentationDataset(
                data_dir=data_dir,
                split='test'
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

            # Set outputs
            self.set_output_value('train_loader', train_loader)
            self.set_output_value('test_loader', test_loader)
            self.set_output_value('num_train', len(train_dataset))
            self.set_output_value('num_test', len(test_dataset))

            print(f"✓ Created data loaders:")
            print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
            print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

            return True

        except Exception as e:
            print(f"Error in MedicalSegmentationLoaderNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        return {
            'data_dir': {
                'type': 'text',
                'label': 'Data Directory',
                'default': 'examples/medical_segmentation_pipeline/data'
            },
            'batch_size': {
                'type': 'text',
                'label': 'Batch Size',
                'default': '4'
            },
            'num_workers': {
                'type': 'text',
                'label': 'Num Workers',
                'default': '0'
            },
            'shuffle_train': {
                'type': 'choice',
                'label': 'Shuffle Training',
                'choices': ['True', 'False'],
                'default': 'True'
            }
        }


# Test the dataloader
if __name__ == "__main__":
    print("Testing Medical Segmentation Dataloader...")

    # Test dataset
    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run download_dataset.py first!")
        sys.exit(1)

    # Create dataset
    train_dataset = MedicalSegmentationDataset(data_dir, split='train')

    print(f"\n✓ Training dataset: {len(train_dataset)} samples")

    # Test loading a sample
    image, mask = train_dataset[0]
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Mask unique values: {torch.unique(mask).tolist()}")

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(f"\n✓ DataLoader: {len(train_loader)} batches")

    # Test batch loading
    images, masks = next(iter(train_loader))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch masks shape: {masks.shape}")

    print("\n✓ Dataloader test passed!")
