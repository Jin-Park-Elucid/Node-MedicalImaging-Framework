"""
Launch GUI with Medical Segmentation Pipeline nodes registered.

This script ensures custom nodes are registered before launching
the GUI workflow editor:
- MedicalSegmentationLoader: Synthetic data loader
- AVTE2DLoader: AVTE 2D segmentation dataloader with multi-slice context
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Framework root
sys.path.insert(0, str(Path(__file__).parent / 'medical_segmentation_pipeline'))  # For custom_dataloader
sys.path.insert(0, str(Path(__file__).parent / 'AVTE'))  # For AVTE nodes

# Import framework and nodes
from medical_imaging_framework.core import NodeRegistry
import medical_imaging_framework.nodes

# Import custom dataloaders to register them
from custom_dataloader import MedicalSegmentationLoaderNode

# Import AVTE module to register AVTE2DLoader node
try:
    from avte_dataloader_node import AVTE2DLoaderNode
    avte_available = True
except ImportError as e:
    avte_available = False
    avte_error = str(e)

# Launch GUI
from medical_imaging_framework.gui.editor import main as gui_main

if __name__ == "__main__":
    print("="*80)
    print("MEDICAL IMAGING FRAMEWORK - GUI EDITOR")
    print("="*80)
    print()
    print("Custom nodes registered:")
    print("  ✓ MedicalSegmentationLoader")
    if avte_available:
        print("  ✓ AVTE2DLoader")
    else:
        print(f"  ✗ AVTE2DLoader (not available: {avte_error})")
    print()
    print(f"Total nodes available: {len(NodeRegistry.get_all_nodes())}")
    print()
    print("Workflow files available:")

    # Medical segmentation pipeline workflows
    workflow_dir = Path(__file__).parent / 'medical_segmentation_pipeline'
    medseg_workflows = list(workflow_dir.glob("*_workflow.json"))
    if medseg_workflows:
        print("  [Medical Segmentation Pipeline]")
        for workflow_file in medseg_workflows:
            print(f"    • {workflow_file.name}")

    # AVTE workflows
    avte_workflow_dir = Path(__file__).parent / 'AVTE' / 'config'
    avte_workflows = list(avte_workflow_dir.glob("*.json"))
    if avte_workflows:
        print("  [AVTE 2D Segmentation]")
        for workflow_file in avte_workflows:
            print(f"    • {workflow_file.name}")

    if not medseg_workflows and not avte_workflows:
        print("  (No workflow files found)")

    print()
    print("="*80)
    print()

    # Launch GUI
    gui_main()
