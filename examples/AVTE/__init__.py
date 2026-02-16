"""
AVTE 2D Segmentation Module

This module provides preprocessing and data loading utilities for
training 2D segmentation networks on AVTE medical imaging data.

Main components:
- preprocess_2d_slices: Convert 3D NIfTI to 2D slices with context
- avte_dataloader: PyTorch Dataset and DataLoader for preprocessed data
- avte_dataloader_node: GUI node for visual pipeline editor
- example_usage: Example training script

Author: Medical Imaging Framework
Date: 2026-02-08
"""

from pathlib import Path

__version__ = "1.2.0"
__author__ = "Medical Imaging Framework"

# Module paths
MODULE_DIR = Path(__file__).parent
EXAMPLES_DIR = MODULE_DIR.parent
DOCS_DIR = MODULE_DIR / "docs"

# Import node for auto-registration
try:
    from .avte_dataloader_node import AVTE2DLoaderNode
    __all__ = [
        "MODULE_DIR",
        "EXAMPLES_DIR",
        "DOCS_DIR",
        "AVTE2DLoaderNode",
    ]
except ImportError:
    # Framework not available
    __all__ = [
        "MODULE_DIR",
        "EXAMPLES_DIR",
        "DOCS_DIR",
    ]
