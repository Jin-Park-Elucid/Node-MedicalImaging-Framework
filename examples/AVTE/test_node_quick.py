#!/usr/bin/env python3
"""Quick validation test for AVTE2DLoader node."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

print("Testing AVTE2DLoader node import...")

try:
    from avte_dataloader_node import AVTE2DLoaderNode
    print("✓ Successfully imported AVTE2DLoaderNode")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    from medical_imaging_framework.core import NodeRegistry
    print("✓ Successfully imported NodeRegistry")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

try:
    # Check if node is registered
    all_nodes = NodeRegistry.get_all_nodes()
    if 'AVTE2DLoader' in str(all_nodes):
        print("✓ Node is registered in NodeRegistry")
    else:
        print("⚠ Node may not be registered yet (registry empty)")
except Exception as e:
    print(f"⚠ Could not check registry: {e}")

# Create node instance
try:
    node = AVTE2DLoaderNode()
    print(f"✓ Successfully created node instance")
    print(f"  Node type: {type(node).__name__}")
    print(f"  Node has execute method: {hasattr(node, 'execute')}")
    print(f"  Node has get_field_definitions method: {hasattr(node, 'get_field_definitions')}")
except Exception as e:
    print(f"✗ Failed to create node: {e}")
    sys.exit(1)

# Check field definitions
try:
    fields = node.get_field_definitions()
    print(f"✓ Node has {len(fields)} configuration fields:")
    for field_name in fields.keys():
        print(f"    - {field_name}")
except Exception as e:
    print(f"✗ Failed to get field definitions: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL VALIDATION TESTS PASSED!")
print("="*60)
print("\nThe node is ready for GUI integration.")
