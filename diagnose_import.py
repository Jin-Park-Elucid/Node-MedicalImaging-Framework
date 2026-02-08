#!/usr/bin/env python3
"""
Diagnostic script to identify the circular import issue.
"""

import sys
import os
from pathlib import Path

print("╔══════════════════════════════════════════════════════════════════╗")
print("║           Import Diagnostic Tool                                ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Check current directory
print(f"Current directory: {os.getcwd()}")
print()

# Check if package is installed
print("Checking package installation...")
try:
    import pkg_resources
    try:
        version = pkg_resources.get_distribution('Node-MedicalImaging-Framework').version
        location = pkg_resources.get_distribution('Node-MedicalImaging-Framework').location
        print(f"✅ Package installed: version {version}")
        print(f"   Location: {location}")
    except pkg_resources.DistributionNotFound:
        print("❌ Package not installed via pip")
except ImportError:
    print("⚠️  Cannot check (pkg_resources not available)")
print()

# Check sys.path
print("Python import search path:")
for i, p in enumerate(sys.path[:5], 1):
    print(f"  {i}. {p}")
print()

# Try importing step by step
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Step 1: Import core")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from medical_imaging_framework.core import BaseNode, NodeRegistry, DataType
    print("✅ Core imports successful")
    print(f"   BaseNode: {BaseNode}")
    print(f"   NodeRegistry: {NodeRegistry}")
except Exception as e:
    print(f"❌ Core import failed: {e}")
    sys.exit(1)
print()

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Step 2: Import nodes package")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    import medical_imaging_framework.nodes
    print("✅ Nodes package imported")
except Exception as e:
    print(f"❌ Nodes package import failed: {e}")
    print()
    print("Attempting to import sub-modules individually...")
    print()

    # Try each submodule
    submodules = ['data', 'networks', 'training', 'inference', 'visualization']
    for submod in submodules:
        try:
            module_name = f'medical_imaging_framework.nodes.{submod}'
            __import__(module_name)
            print(f"✅ {submod} module imported successfully")
        except Exception as e2:
            print(f"❌ {submod} module failed: {e2}")

    sys.exit(1)
print()

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Step 3: Check node registration")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    nodes = NodeRegistry.get_all_nodes()
    print(f"✅ {len(nodes)} nodes registered")
    print()
    print("Registered nodes by category:")
    categories = NodeRegistry.get_categories()
    for category in sorted(categories):
        node_names = NodeRegistry.get_nodes_by_category(category)
        print(f"  {category}: {len(node_names)} nodes")
        for node_name in sorted(node_names):
            print(f"    - {node_name}")
except Exception as e:
    print(f"❌ Cannot list nodes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Step 4: Test node creation")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
try:
    from medical_imaging_framework import ComputationalGraph
    graph = ComputationalGraph("Test")
    node = NodeRegistry.create_node('UNet2D', 'test', config={
        'in_channels': 1,
        'out_channels': 2
    })
    graph.add_node(node)
    print("✅ Can create nodes and graphs")
    print(f"   Created UNet2D node: {node.name}")
except Exception as e:
    print(f"❌ Node creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("╔══════════════════════════════════════════════════════════════════╗")
print("║                 ✅ ALL DIAGNOSTICS PASSED                        ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()
print("The framework is working correctly!")
print()
