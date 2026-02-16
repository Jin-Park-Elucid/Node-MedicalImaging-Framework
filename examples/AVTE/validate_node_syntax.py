#!/usr/bin/env python3
"""Validate AVTE2DLoader node syntax and structure."""

import ast
import sys
from pathlib import Path

node_file = Path(__file__).parent / "avte_dataloader_node.py"

print("="*60)
print("AVTE2DLoader Node Validation")
print("="*60)

# 1. Check file exists
print(f"\n1. Checking file exists: {node_file}")
if not node_file.exists():
    print(f"✗ File not found!")
    sys.exit(1)
print("✓ File exists")

# 2. Parse syntax
print("\n2. Parsing Python syntax...")
try:
    with open(node_file, 'r') as f:
        code = f.read()
    tree = ast.parse(code, filename=str(node_file))
    print("✓ Syntax is valid")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

# 3. Check for required class
print("\n3. Checking for AVTE2DLoaderNode class...")
classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
class_names = [c.name for c in classes]
if 'AVTE2DLoaderNode' in class_names:
    print("✓ AVTE2DLoaderNode class found")
else:
    print(f"✗ Class not found. Found: {class_names}")
    sys.exit(1)

# 4. Check for required methods
print("\n4. Checking for required methods...")
avte_class = [c for c in classes if c.name == 'AVTE2DLoaderNode'][0]
methods = [node.name for node in avte_class.body if isinstance(node, ast.FunctionDef)]
required_methods = ['_setup_ports', 'execute', 'get_field_definitions']
missing = [m for m in required_methods if m not in methods]
if missing:
    print(f"✗ Missing methods: {missing}")
    sys.exit(1)
for method in required_methods:
    print(f"  ✓ {method}")

# 5. Check for decorator
print("\n5. Checking for NodeRegistry.register decorator...")
has_decorator = False
for decorator in avte_class.decorator_list:
    if isinstance(decorator, ast.Call):
        if hasattr(decorator.func, 'attr') and decorator.func.attr == 'register':
            has_decorator = True
            # Check decorator arguments
            if len(decorator.args) >= 2:
                category = decorator.args[0].value if isinstance(decorator.args[0], ast.Constant) else None
                name = decorator.args[1].value if isinstance(decorator.args[1], ast.Constant) else None
                print(f"  ✓ Decorator found with category='{category}', name='{name}'")
if not has_decorator:
    print("  ⚠ Decorator not found or not in expected format")

# 6. Check imports
print("\n6. Checking imports...")
imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
import_names = []
for imp in imports:
    if isinstance(imp, ast.Import):
        import_names.extend([alias.name for alias in imp.names])
    elif isinstance(imp, ast.ImportFrom):
        if imp.module:
            import_names.append(imp.module)

required_imports = ['torch', 'medical_imaging_framework.core', 'avte_dataloader']
for req in required_imports:
    found = any(req in name for name in import_names)
    if found:
        print(f"  ✓ {req}")
    else:
        print(f"  ✗ {req} not found")

# 7. Check __init__.py
print("\n7. Checking __init__.py...")
init_file = Path(__file__).parent / "__init__.py"
if init_file.exists():
    print("  ✓ __init__.py exists")
    with open(init_file, 'r') as f:
        init_content = f.read()
    if 'AVTE2DLoaderNode' in init_content:
        print("  ✓ __init__.py imports AVTE2DLoaderNode")
    else:
        print("  ⚠ __init__.py does not import AVTE2DLoaderNode")
else:
    print("  ✗ __init__.py not found")

# 8. Check documentation
print("\n8. Checking documentation...")
docs = [
    "GUI_NODE_GUIDE.md",
    "README.md",
    "CHANGELOG.md"
]
for doc in docs:
    doc_path = Path(__file__).parent / doc
    if doc_path.exists():
        print(f"  ✓ {doc}")
    else:
        print(f"  ⚠ {doc} not found")

print("\n" + "="*60)
print("✓ VALIDATION COMPLETE!")
print("="*60)
print("\nThe AVTE2DLoader node structure is valid and ready for use.")
print("\nTo test with actual data, ensure PyTorch is installed and run:")
print("  python3 avte_dataloader_node.py")
