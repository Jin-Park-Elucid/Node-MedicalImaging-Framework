#!/usr/bin/env python3
"""
Preprocessing script for 2D segmentation training with multi-slice windows.

This script converts 3D NIfTI volumes into 2D slices with adjacent slice context,
suitable for training 2D segmentation networks on AVTE data.

Features:
- Multi-slice window: includes adjacent slices as input channels
- Border handling: zero padding, duplication, or mirroring
- Saves individual .npz files per slice for efficient data loading
- Multiprocessing support for parallel processing
- Compatible with future AVTE dataloader implementation

Author: Medical Imaging Framework
Date: 2026-02-08
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Literal, Tuple, List, Dict, Any
import json
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial


def process_single_file_worker(
    args: Tuple[Path, Path, Path, Dict[str, Any]]
) -> Tuple[int, str]:
    """
    Worker function to process a single NIfTI file.

    This function is designed to be used with multiprocessing.Pool.

    Args:
        args: Tuple containing (image_path, label_path, output_dir, config)

    Returns:
        Tuple of (num_slices, case_name)
    """
    image_path, label_path, output_dir, config = args

    try:
        # Create preprocessor with config
        preprocessor = NIfTI2DSlicePreprocessor(
            window_size=config['window_size'],
            padding_mode=config['padding_mode'],
            normalize=config['normalize'],
            clip_range=config['clip_range']
        )

        # Process the file
        output_paths = preprocessor.process_nifti_file(
            image_path, label_path, output_dir
        )

        return len(output_paths), image_path.stem

    except Exception as e:
        print(f"\nError processing {image_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return 0, image_path.stem


class NIfTI2DSlicePreprocessor:
    """
    Preprocessor for converting 3D NIfTI volumes to 2D slices with context.
    """

    def __init__(
        self,
        window_size: int = 2,
        padding_mode: Literal['zero', 'replicate', 'mirror'] = 'replicate',
        normalize: bool = True,
        clip_range: Tuple[float, float] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            window_size: Number of adjacent slices before and after center slice.
                        Total channels = 2 * window_size + 1
            padding_mode: How to handle border cases:
                - 'zero': Fill with zeros
                - 'replicate': Duplicate the first/last slice
                - 'mirror': Mirror the slices at boundaries
            normalize: Whether to apply z-score normalization per volume
            clip_range: Optional (min, max) tuple to clip HU values before normalization
        """
        self.window_size = window_size
        self.padding_mode = padding_mode
        self.normalize = normalize
        self.clip_range = clip_range
        self.total_channels = 2 * window_size + 1

    def _get_slice_with_context(
        self,
        volume: np.ndarray,
        slice_idx: int,
        depth: int
    ) -> np.ndarray:
        """
        Extract a 2D slice with adjacent slices as channels.

        Args:
            volume: 3D numpy array (D, H, W)
            slice_idx: Index of the center slice
            depth: Total depth of the volume

        Returns:
            Multi-channel 2D array (C, H, W) where C = 2*window_size + 1
        """
        channels = []

        # Get slice indices for the window
        for offset in range(-self.window_size, self.window_size + 1):
            target_idx = slice_idx + offset

            # Handle boundary cases
            if target_idx < 0:
                if self.padding_mode == 'zero':
                    slice_data = np.zeros_like(volume[0])
                elif self.padding_mode == 'replicate':
                    slice_data = volume[0]
                elif self.padding_mode == 'mirror':
                    # Mirror: [-2, -1, 0, 1, 2] -> [1, 0, 0, 1, 2]
                    mirror_idx = abs(target_idx)
                    slice_data = volume[min(mirror_idx, depth - 1)]
            elif target_idx >= depth:
                if self.padding_mode == 'zero':
                    slice_data = np.zeros_like(volume[0])
                elif self.padding_mode == 'replicate':
                    slice_data = volume[-1]
                elif self.padding_mode == 'mirror':
                    # Mirror: [D-2, D-1, D, D+1] -> [D-2, D-1, D-1, D-2]
                    mirror_idx = 2 * depth - target_idx - 1
                    slice_data = volume[max(0, mirror_idx)]
            else:
                slice_data = volume[target_idx]

            channels.append(slice_data)

        # Stack channels: (C, H, W)
        multi_channel_slice = np.stack(channels, axis=0)
        return multi_channel_slice

    def _normalize_volume(self, volume: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize the volume using z-score normalization.

        Args:
            volume: 3D numpy array

        Returns:
            Normalized volume and normalization statistics
        """
        if self.clip_range is not None:
            volume = np.clip(volume, self.clip_range[0], self.clip_range[1])

        # Compute statistics on non-zero voxels (foreground)
        foreground_mask = volume > volume.min()
        if foreground_mask.any():
            mean = volume[foreground_mask].mean()
            std = volume[foreground_mask].std()
        else:
            mean = volume.mean()
            std = volume.std()

        # Avoid division by zero
        if std < 1e-8:
            std = 1.0

        normalized = (volume - mean) / std

        stats = {
            'mean': float(mean),
            'std': float(std),
            'min': float(volume.min()),
            'max': float(volume.max())
        }

        return normalized, stats

    def process_nifti_file(
        self,
        image_path: Path,
        label_path: Path,
        output_dir: Path
    ) -> List[Path]:
        """
        Process a single NIfTI file and save 2D slices.

        Args:
            image_path: Path to input image (.nii.gz)
            label_path: Path to segmentation label (.nii.gz)
            output_dir: Directory to save processed slices

        Returns:
            List of output file paths
        """
        # Load NIfTI files
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        # Get data as numpy arrays (Z, Y, X)
        image_data = image_nii.get_fdata().astype(np.float32)
        label_data = label_nii.get_fdata().astype(np.int8)

        # Get metadata
        spacing = image_nii.header.get_zooms()

        # Normalize image if requested
        if self.normalize:
            image_data, norm_stats = self._normalize_volume(image_data)
        else:
            norm_stats = {}

        # Get filename without extension
        filename_base = image_path.stem.replace('.nii', '').replace('_0000', '')

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each slice
        depth = image_data.shape[0]
        output_paths = []

        for slice_idx in range(depth):
            # Get multi-channel image slice
            image_slice = self._get_slice_with_context(
                image_data, slice_idx, depth
            )  # Shape: (C, H, W)

            # Get corresponding label (single channel)
            label_slice = label_data[slice_idx]  # Shape: (H, W)

            # Create output filename: <case>_<slice_idx:03d>.npz
            output_filename = f"{filename_base}_{slice_idx:03d}.npz"
            output_path = output_dir / output_filename

            # Save as compressed npz
            np.savez_compressed(
                output_path,
                image=image_slice.astype(np.float32),
                label=label_slice.astype(np.int8),
                metadata=json.dumps({
                    'source_file': str(image_path.name),
                    'slice_index': int(slice_idx),
                    'total_slices': int(depth),
                    'window_size': self.window_size,
                    'padding_mode': self.padding_mode,
                    'num_channels': self.total_channels,
                    'spacing': [float(s) for s in spacing],
                    'normalized': self.normalize,
                    'norm_stats': norm_stats,
                    'shape': {
                        'image': list(image_slice.shape),
                        'label': list(label_slice.shape)
                    }
                })
            )

            output_paths.append(output_path)

        return output_paths

    def process_dataset(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        num_workers: int = 4
    ):
        """
        Process entire dataset: imagesTr and labelsTr with multiprocessing.

        Args:
            raw_data_dir: Root directory containing imagesTr/ and labelsTr/
            output_dir: Directory to save processed slices
            num_workers: Number of parallel worker processes (0 = single process)
        """
        images_dir = raw_data_dir / 'imagesTr'
        labels_dir = raw_data_dir / 'labelsTr'

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")

        # Get all image files
        image_files = sorted(list(images_dir.glob('*_0000.nii.gz')))

        if len(image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

        print(f"\nFound {len(image_files)} cases to process")

        if num_workers > 0:
            print(f"Using {num_workers} worker processes for parallel processing")
        else:
            print("Using single-process mode (num_workers=0)")

        # Prepare config dict for workers
        config = {
            'window_size': self.window_size,
            'padding_mode': self.padding_mode,
            'normalize': self.normalize,
            'clip_range': self.clip_range
        }

        print(f"\n{'='*60}")
        print(f"Processing all cases ({len(image_files)} files)")
        print('='*60)

        # Prepare arguments for each file
        file_args = []
        for image_path in image_files:
            # Get corresponding label file
            label_filename = image_path.stem.replace('.nii', '').replace('_0000', '') + '.nii.gz'
            label_path = labels_dir / label_filename

            if not label_path.exists():
                print(f"Warning: Label not found for {image_path.name}, skipping...")
                continue

            file_args.append((image_path, label_path, output_dir, config))

        # Process files with multiprocessing or single process
        if num_workers > 0:
            # Multiprocessing mode
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_file_worker, file_args),
                    total=len(file_args),
                    desc="Processing"
                ))
        else:
            # Single process mode (easier for debugging)
            results = []
            for args in tqdm(file_args, desc="Processing"):
                results.append(process_single_file_worker(args))

        # Accumulate slice counts
        total_slices = sum(num_slices for num_slices, _ in results)
        processed_cases = len([r for r in results if r[0] > 0])

        # Save dataset statistics
        dataset_info = {
            'window_size': self.window_size,
            'total_channels': self.total_channels,
            'padding_mode': self.padding_mode,
            'normalized': self.normalize,
            'clip_range': self.clip_range,
            'num_cases': processed_cases,
            'num_slices': total_slices,
            'case_names': [case_name.replace('.nii', '').replace('_0000', '')
                          for _, case_name in results if _ > 0]
        }

        info_path = output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print('='*60)
        print(f"Output directory: {output_dir}")
        print(f"Dataset info saved to: {info_path}")
        print(f"\nDataset statistics:")
        print(f"  Total cases: {processed_cases}")
        print(f"  Total slices: {total_slices}")
        print(f"  Channels per slice: {self.total_channels} (window_size={self.window_size})")
        print(f"\nNote: Train/val/test splits will be handled by the dataloader")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess 3D NIfTI volumes into 2D slices for segmentation training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default='/data/avte_training/nnUNet_raw/Dataset006_model_9_4',
        help='Input directory containing imagesTr/ and labelsTr/'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/jin.park/Code_Hendrix/Data/AVTE/Dataset006_model_9_4/2D_data',
        help='Output directory for processed 2D slices'
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=2,
        help='Number of adjacent slices before and after center (total channels = 2*window_size + 1)'
    )

    parser.add_argument(
        '--padding_mode',
        type=str,
        default='replicate',
        choices=['zero', 'replicate', 'mirror'],
        help='How to handle border slices'
    )

    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable z-score normalization'
    )

    parser.add_argument(
        '--clip_min',
        type=float,
        default=None,
        help='Minimum HU value to clip (e.g., -1024 for CT)'
    )

    parser.add_argument(
        '--clip_max',
        type=float,
        default=None,
        help='Maximum HU value to clip (e.g., 3071 for CT)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel worker processes (0 = single process, -1 = CPU count)'
    )

    args = parser.parse_args()

    # Prepare clip range
    clip_range = None
    if args.clip_min is not None and args.clip_max is not None:
        clip_range = (args.clip_min, args.clip_max)

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = mp.cpu_count()
        print(f"Auto-detected {num_workers} CPUs")
    elif num_workers < 0:
        print("Error: num_workers must be >= -1")
        sys.exit(1)

    # Initialize preprocessor
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print('='*60)
    print(f"Window size: {args.window_size} (total channels: {2*args.window_size + 1})")
    print(f"Padding mode: {args.padding_mode}")
    print(f"Normalization: {not args.no_normalize}")
    print(f"Clip range: {clip_range}")
    print(f"Parallel workers: {num_workers}")

    preprocessor = NIfTI2DSlicePreprocessor(
        window_size=args.window_size,
        padding_mode=args.padding_mode,
        normalize=not args.no_normalize,
        clip_range=clip_range
    )

    # Process dataset
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("2D SLICE PREPROCESSING FOR AVTE SEGMENTATION")
    print('='*60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print('='*60)

    try:
        preprocessor.process_dataset(
            raw_data_dir=input_dir,
            output_dir=output_dir,
            num_workers=num_workers
        )

        print("\n✓ Preprocessing completed successfully!")
        print("Note: Use the dataloader to split into train/val/test sets")

    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
