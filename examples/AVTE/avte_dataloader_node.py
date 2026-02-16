#!/usr/bin/env python3
"""
AVTE 2D Dataloader Node for GUI Integration.

This module provides a node that can be used in the medical imaging
framework's visual pipeline editor to load AVTE 2D segmentation data.

Author: Medical Imaging Framework
Date: 2026-02-08
"""

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType
from avte_dataloader import AVTE2DDataset


@NodeRegistry.register('data', 'AVTE2DLoader',
                      description='Load AVTE 2D segmentation dataset with multi-slice context')
class AVTE2DLoaderNode(BaseNode):
    """
    AVTE 2D Dataloader Node for GUI.

    Loads preprocessed AVTE 2D slices with multi-slice context windows
    and creates PyTorch DataLoaders with automatic train/val/test splitting.

    Features:
    - Multi-slice context windows (configurable channels)
    - Automatic case-level train/val/test splitting
    - Reproducible splits with random seed
    - Configurable batch size and workers
    - GPU memory pinning support
    """

    def _setup_ports(self):
        """Define output ports for the node."""
        self.add_output('train_loader', DataType.BATCH)
        self.add_output('val_loader', DataType.BATCH)
        self.add_output('test_loader', DataType.BATCH)
        self.add_output('num_train', DataType.ANY)
        self.add_output('num_val', DataType.ANY)
        self.add_output('num_test', DataType.ANY)
        self.add_output('num_channels', DataType.ANY)

    def execute(self) -> bool:
        """
        Execute the node to create dataloaders.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get configuration
            data_dir = self.get_config('data_dir',
                '/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data')
            batch_size = int(self.get_config('batch_size', 8))
            num_workers = int(self.get_config('num_workers', 4))
            train_ratio = float(self.get_config('train_ratio', 0.8))
            val_ratio = float(self.get_config('val_ratio', 0.1))
            random_seed = int(self.get_config('random_seed', 42))

            # Convert string 'True'/'False' to boolean
            pin_memory_str = self.get_config('pin_memory', 'True')
            if isinstance(pin_memory_str, str):
                pin_memory = pin_memory_str.lower() in ['true', '1', 'yes']
            else:
                pin_memory = bool(pin_memory_str)

            load_to_memory_str = self.get_config('load_to_memory', 'False')
            if isinstance(load_to_memory_str, str):
                load_to_memory = load_to_memory_str.lower() in ['true', '1', 'yes']
            else:
                load_to_memory = bool(load_to_memory_str)

            shuffle_train_str = self.get_config('shuffle_train', 'True')
            if isinstance(shuffle_train_str, str):
                shuffle_train = shuffle_train_str.lower() in ['true', '1', 'yes']
            else:
                shuffle_train = bool(shuffle_train_str)

            print(f"Loading AVTE 2D dataset from: {data_dir}")
            print(f"Configuration:")
            print(f"  Batch size: {batch_size}")
            print(f"  Workers: {num_workers}")
            print(f"  Train ratio: {train_ratio}")
            print(f"  Val ratio: {val_ratio}")
            print(f"  Random seed: {random_seed}")

            # Create datasets with automatic splitting
            train_dataset = AVTE2DDataset(
                data_dir=data_dir,
                split='train',
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                load_to_memory=load_to_memory,
                random_seed=random_seed
            )

            val_dataset = AVTE2DDataset(
                data_dir=data_dir,
                split='val',
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                load_to_memory=load_to_memory,
                random_seed=random_seed
            )

            test_dataset = AVTE2DDataset(
                data_dir=data_dir,
                split='test',
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                load_to_memory=load_to_memory,
                random_seed=random_seed
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True  # Drop last incomplete batch for training
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available()
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available()
            )

            # Get number of channels from dataset info
            num_channels = train_dataset.dataset_info.get('total_channels', 'Unknown')

            # Set outputs
            self.set_output_value('train_loader', train_loader)
            self.set_output_value('val_loader', val_loader)
            self.set_output_value('test_loader', test_loader)
            self.set_output_value('num_train', len(train_dataset))
            self.set_output_value('num_val', len(val_dataset))
            self.set_output_value('num_test', len(test_dataset))
            self.set_output_value('num_channels', num_channels)

            print(f"\n✓ Created AVTE 2D data loaders:")
            print(f"  Training:   {len(train_dataset)} slices, {len(train_loader)} batches")
            print(f"  Validation: {len(val_dataset)} slices, {len(val_loader)} batches")
            print(f"  Test:       {len(test_dataset)} slices, {len(test_loader)} batches")
            print(f"  Channels:   {num_channels}")

            return True

        except Exception as e:
            print(f"✗ Error in AVTE2DLoaderNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        """
        Define GUI form fields for the node.

        Returns:
            dict: Field definitions for the GUI
        """
        return {
            'data_dir': {
                'type': 'text',
                'label': 'Data Directory',
                'default': '/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
                'help': 'Path to preprocessed AVTE 2D data'
            },
            'batch_size': {
                'type': 'text',
                'label': 'Batch Size',
                'default': '8',
                'help': 'Number of samples per batch'
            },
            'train_ratio': {
                'type': 'text',
                'label': 'Train Ratio',
                'default': '0.8',
                'help': 'Proportion of cases for training (0.0-1.0)'
            },
            'val_ratio': {
                'type': 'text',
                'label': 'Validation Ratio',
                'default': '0.1',
                'help': 'Proportion of cases for validation (0.0-1.0)'
            },
            'random_seed': {
                'type': 'text',
                'label': 'Random Seed',
                'default': '42',
                'help': 'Seed for reproducible train/val/test splits'
            },
            'num_workers': {
                'type': 'text',
                'label': 'Num Workers',
                'default': '4',
                'help': 'Number of parallel data loading workers'
            },
            'shuffle_train': {
                'type': 'choice',
                'label': 'Shuffle Training',
                'choices': ['True', 'False'],
                'default': 'True',
                'help': 'Whether to shuffle training data'
            },
            'pin_memory': {
                'type': 'choice',
                'label': 'Pin Memory',
                'choices': ['True', 'False'],
                'default': 'True',
                'help': 'Pin memory for faster GPU transfer'
            },
            'load_to_memory': {
                'type': 'choice',
                'label': 'Load to Memory',
                'choices': ['False', 'True'],
                'default': 'False',
                'help': 'Load all data to RAM (faster but memory intensive)'
            }
        }


# Test the node
if __name__ == "__main__":
    print("="*60)
    print("TESTING AVTE 2D LOADER NODE")
    print("="*60)

    # Create node instance
    node = AVTE2DLoaderNode()

    # Set configuration
    node.set_config('data_dir', '/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data')
    node.set_config('batch_size', '4')
    node.set_config('train_ratio', '0.8')
    node.set_config('val_ratio', '0.1')
    node.set_config('num_workers', '0')

    # Execute node
    print("\nExecuting node...")
    success = node.execute()

    if success:
        # Get outputs
        train_loader = node.get_output_value('train_loader')
        val_loader = node.get_output_value('val_loader')
        test_loader = node.get_output_value('test_loader')
        num_train = node.get_output_value('num_train')
        num_val = node.get_output_value('num_val')
        num_test = node.get_output_value('num_test')
        num_channels = node.get_output_value('num_channels')

        print("\n" + "="*60)
        print("NODE OUTPUTS")
        print("="*60)
        print(f"Train loader: {type(train_loader).__name__}")
        print(f"Val loader: {type(val_loader).__name__}")
        print(f"Test loader: {type(test_loader).__name__}")
        print(f"Num train samples: {num_train}")
        print(f"Num val samples: {num_val}")
        print(f"Num test samples: {num_test}")
        print(f"Num channels: {num_channels}")

        # Test loading a batch
        print("\n" + "="*60)
        print("TESTING BATCH LOADING")
        print("="*60)
        images, labels = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique labels: {torch.unique(labels).tolist()}")

        print("\n" + "="*60)
        print("✓ NODE TEST PASSED!")
        print("="*60)
    else:
        print("\n✗ Node execution failed!")
        sys.exit(1)
