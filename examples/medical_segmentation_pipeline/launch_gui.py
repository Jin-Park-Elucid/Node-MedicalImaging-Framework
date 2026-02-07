"""
Launch GUI with Medical Segmentation Pipeline nodes registered.

This script ensures the custom MedicalSegmentationLoader node is
registered before launching the GUI workflow editor.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import framework and nodes
from medical_imaging_framework.core import NodeRegistry
import medical_imaging_framework.nodes

# Import custom dataloader to register it
from custom_dataloader import MedicalSegmentationLoaderNode

# Launch GUI
from medical_imaging_framework.gui.editor import main as gui_main

if __name__ == "__main__":
    print("="*80)
    print("MEDICAL IMAGING FRAMEWORK - GUI EDITOR")
    print("="*80)
    print()
    print("Custom nodes registered:")
    print("  ✓ MedicalSegmentationLoader")
    print()
    print(f"Total nodes available: {len(NodeRegistry.get_all_nodes())}")
    print()
    print("Workflow files available:")
    workflow_dir = Path(__file__).parent
    for workflow_file in workflow_dir.glob("*_workflow.json"):
        print(f"  • {workflow_file.name}")
    print()
    print("="*80)
    print()

    # Launch GUI
    gui_main()
