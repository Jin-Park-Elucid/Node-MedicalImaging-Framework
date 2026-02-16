#!/usr/bin/env python3
"""
DataLoader for AVTE 2D segmentation dataset.

This module provides a PyTorch Dataset and DataLoader for loading
preprocessed 2D slices created by preprocess_2d_slices.py.

Compatible with multi-channel input (adjacent slices as channels).

Author: Medical Imaging Framework
Date: 2026-02-08
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Optional, Tuple, Callable
import sys


class AVTE2DDataset(Dataset):
    """
    Dataset for AVTE 2D segmentation with multi-slice context.

    Loads preprocessed .npz files containing 2D slices with adjacent
    slices as input channels. Supports automatic train/val/test splitting.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_to_memory: bool = False,
        random_seed: int = 42
    ):
        """
        Initialize AVTE 2D dataset.

        Args:
            data_dir: Root directory containing preprocessed .npz files
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of cases for training (default: 0.8)
            val_ratio: Proportion of cases for validation (default: 0.1)
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
            load_to_memory: If True, load all data to RAM (faster but memory intensive)
            random_seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.transform = transform
        self.target_transform = target_transform
        self.load_to_memory = load_to_memory
        self.random_seed = random_seed

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # Load dataset info
        info_path = self.data_dir / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}

        # Get all .npz files
        all_slice_files = sorted(list(self.data_dir.glob("*.npz")))

        if len(all_slice_files) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}")

        # Group slices by case (extract case name from filename)
        case_to_slices = {}
        for slice_file in all_slice_files:
            # Filename format: casename_###.npz
            case_name = '_'.join(slice_file.stem.split('_')[:-1])
            if case_name not in case_to_slices:
                case_to_slices[case_name] = []
            case_to_slices[case_name].append(slice_file)

        # Sort case names for reproducibility
        case_names = sorted(case_to_slices.keys())
        n_cases = len(case_names)

        # Split cases into train/val/test
        np.random.seed(random_seed)
        indices = np.random.permutation(n_cases)

        n_train = int(n_cases * train_ratio)
        n_val = int(n_cases * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Get slice files for this split
        if split == 'train':
            split_cases = [case_names[i] for i in train_indices]
        elif split == 'val':
            split_cases = [case_names[i] for i in val_indices]
        elif split == 'test':
            split_cases = [case_names[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        self.slice_files = []
        for case_name in split_cases:
            self.slice_files.extend(case_to_slices[case_name])
        self.slice_files = sorted(self.slice_files)

        if len(self.slice_files) == 0:
            raise ValueError(f"No slices found for {split} split")

        # Optionally load all data to memory
        self.data_cache = {}
        if self.load_to_memory:
            print(f"Loading {len(self.slice_files)} slices to memory...")
            for idx, file_path in enumerate(self.slice_files):
                with np.load(file_path) as data:
                    self.data_cache[idx] = {
                        'image': data['image'],
                        'label': data['label'],
                        'metadata': json.loads(str(data['metadata']))
                    }
            print(f"✓ Loaded {len(self.data_cache)} slices to memory")

        # Print dataset info
        print(f"✓ AVTE2DDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Cases: {len(split_cases)} ({n_cases} total)")
        print(f"  Slices: {len(self.slice_files)}")
        print(f"  Directory: {self.data_dir}")
        if self.dataset_info:
            print(f"  Window size: {self.dataset_info.get('window_size', 'N/A')}")
            print(f"  Total channels: {self.dataset_info.get('total_channels', 'N/A')}")
            print(f"  Padding mode: {self.dataset_info.get('padding_mode', 'N/A')}")

    def __len__(self) -> int:
        return len(self.slice_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            image: Tensor of shape (C, H, W) where C is number of channels
            label: Tensor of shape (H, W) with integer class labels
        """
        # Load from cache or disk
        if self.load_to_memory:
            data = self.data_cache[idx]
            image = data['image']
            label = data['label']
        else:
            file_path = self.slice_files[idx]
            with np.load(file_path) as data:
                image = data['image']  # Shape: (C, H, W)
                label = data['label']  # Shape: (H, W)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing metadata
        """
        if self.load_to_memory:
            return self.data_cache[idx]['metadata']
        else:
            file_path = self.slice_files[idx]
            with np.load(file_path) as data:
                metadata = json.loads(str(data['metadata']))
            return metadata

    def get_sample_info(self, idx: int) -> str:
        """
        Get human-readable information about a sample.

        Args:
            idx: Sample index

        Returns:
            String with sample information
        """
        metadata = self.get_metadata(idx)
        info = f"Sample {idx}:\n"
        info += f"  File: {self.slice_files[idx].name}\n"
        info += f"  Source: {metadata.get('source_file', 'N/A')}\n"
        info += f"  Slice: {metadata.get('slice_index', 'N/A')} / {metadata.get('total_slices', 'N/A')}\n"
        info += f"  Channels: {metadata.get('num_channels', 'N/A')}\n"
        info += f"  Shape: {metadata.get('shape', {}).get('image', 'N/A')}\n"
        return info


def create_avte_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_to_memory: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders for AVTE dataset.

    Args:
        data_dir: Root directory containing preprocessed data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        load_to_memory: Whether to load all data to RAM
        train_ratio: Proportion of cases for training (default: 0.8)
        val_ratio: Proportion of cases for validation (default: 0.1)
        random_seed: Random seed for reproducible splits (default: 42)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Create datasets with automatic splitting
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = AVTE2DDataset(
            data_dir=data_dir,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            load_to_memory=load_to_memory,
            random_seed=random_seed
        )

    # Create dataloaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Drop last incomplete batch for training
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )

    print(f"\n✓ Created dataloaders:")
    print(f"  Train: {len(datasets['train'])} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(datasets['val'])} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(datasets['test'])} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# Test the dataloader
if __name__ == "__main__":
    print("="*60)
    print("TESTING AVTE 2D DATALOADER")
    print("="*60)

    # Default data directory
    data_dir = Path("/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data")

    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        print("Run preprocess_2d_slices.py first to create the preprocessed data!")
        print("\nUsage:")
        print("  python preprocess_2d_slices.py --input_dir <input> --output_dir <output>")
        sys.exit(1)

    # Test with custom path if provided
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print(f"\nLoading data from: {data_dir}\n")

    try:
        # Create dataset
        train_dataset = AVTE2DDataset(data_dir, split='train')

        # Test loading a sample
        print("\n" + "="*60)
        print("TESTING SINGLE SAMPLE")
        print("="*60)

        image, label = train_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Label dtype: {label.dtype}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Unique labels: {torch.unique(label).tolist()}")

        # Print sample info
        print("\n" + train_dataset.get_sample_info(0))

        # Test dataloader
        print("="*60)
        print("TESTING DATALOADER")
        print("="*60)

        train_loader, val_loader, test_loader = create_avte_dataloaders(
            data_dir=str(data_dir),
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            load_to_memory=False
        )

        # Test batch loading
        images, labels = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels unique: {torch.unique(labels).tolist()}")

        print("\n" + "="*60)
        print("✓ DATALOADER TEST PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
