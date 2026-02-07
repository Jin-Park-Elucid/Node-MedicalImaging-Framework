#!/usr/bin/env python3
"""
Fix for circular import issue by modifying import strategy.
"""

import os
import shutil
from pathlib import Path

print("╔══════════════════════════════════════════════════════════════════╗")
print("║        Fixing Circular Import - Alternative Approach            ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# Get script directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("Step 1: Backing up current __init__.py")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

init_file = Path("medical_imaging_framework/__init__.py")
backup_file = Path("medical_imaging_framework/__init__.py.backup")

if init_file.exists():
    shutil.copy(init_file, backup_file)
    print(f"✅ Backed up to {backup_file}")
else:
    print(f"❌ File not found: {init_file}")
    exit(1)
print()

print("Step 2: Creating new __init__.py with lazy imports")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

new_init_content = '''"""
Medical Imaging Framework - Node-based deep learning for medical imaging.
"""

__version__ = '0.1.0'

# Import core components
from .core import (
    BaseNode,
    CompositeNode,
    PyTorchModuleNode,
    NodeRegistry,
    ComputationalGraph,
    GraphExecutor,
    DataType,
    PortType
)

# Lazy import of nodes - only import when explicitly needed
# This avoids circular import issues during package initialization
_nodes_imported = False

def _ensure_nodes_imported():
    """Ensure nodes are imported and registered."""
    global _nodes_imported
    if not _nodes_imported:
        from . import nodes
        _nodes_imported = True

# Auto-import nodes for backward compatibility
# Comment out this line if you want fully lazy loading
_ensure_nodes_imported()

__all__ = [
    'BaseNode',
    'CompositeNode',
    'PyTorchModuleNode',
    'NodeRegistry',
    'ComputationalGraph',
    'GraphExecutor',
    'DataType',
    'PortType',
]
'''

with open(init_file, 'w') as f:
    f.write(new_init_content)
print("✅ New __init__.py created")
print()

print("Step 3: Testing the fix")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

try:
    # Clear any cached imports
    import sys
    for key in list(sys.modules.keys()):
        if key.startswith('medical_imaging_framework'):
            del sys.modules[key]

    # Try importing
    import medical_imaging_framework
    print("✅ Basic import successful")

    from medical_imaging_framework import NodeRegistry
    import medical_imaging_framework.nodes
    print("✅ Nodes import successful")

    nodes = NodeRegistry.list_nodes()
    print(f"✅ {len(nodes)} nodes registered")

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    ✅ FIX SUCCESSFUL!                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("The circular import issue has been resolved.")
    print()
    print("Backup saved at: medical_imaging_framework/__init__.py.backup")
    print()

except Exception as e:
    print(f"❌ Fix failed: {e}")
    print()
    print("Restoring backup...")
    shutil.copy(backup_file, init_file)
    print("✅ Backup restored")
    print()
    print("The issue persists. Running diagnostic...")
    print()
    os.system("python3 diagnose_import.py")
    exit(1)
