#!/bin/bash
# Quick fix for circular import issue

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           Fixing Circular Import Issue                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Cleaning cached files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✅ Removed __pycache__ directories"

# Remove all .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "✅ Removed .pyc files"

# Remove all .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✅ Removed .pyo files"

# Remove egg-info if exists
rm -rf medical_imaging_framework.egg-info 2>/dev/null
rm -rf *.egg-info 2>/dev/null
echo "✅ Removed egg-info directories"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Activating virtual environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found at venv/bin/activate"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Reinstalling framework"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Uninstall existing installation
pip uninstall -y medical-imaging-framework 2>/dev/null
echo "✅ Uninstalled old package"

# Install in development mode
pip install -e . --quiet
if [ $? -eq 0 ]; then
    echo "✅ Framework reinstalled successfully"
else
    echo "❌ Failed to reinstall framework"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Testing import"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test basic import
python3 -c "import medical_imaging_framework; print('✅ Basic import successful')" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Basic import failed"
    exit 1
fi

# Test nodes import
python3 -c "import medical_imaging_framework.nodes; print('✅ Nodes import successful')" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Nodes import failed"
    exit 1
fi

# Test node registry
python3 -c "from medical_imaging_framework import NodeRegistry; print('✅ NodeRegistry import successful')" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ NodeRegistry import failed"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Running quick test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'EOF'
try:
    from medical_imaging_framework import NodeRegistry, ComputationalGraph
    import medical_imaging_framework.nodes

    # Try to create a simple node
    graph = ComputationalGraph("Test")
    node = NodeRegistry.create_node('UNet2D', 'test', config={'in_channels': 1, 'out_channels': 2})
    graph.add_node(node)

    print("✅ Framework fully functional!")
    print(f"✅ Can create nodes and graphs")
    print(f"✅ All systems operational")
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ FIX SUCCESSFUL!                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "The circular import issue has been resolved."
    echo ""
    echo "Next steps:"
    echo "  1. Test with: python examples/simple_test.py"
    echo "  2. Launch GUI: python -m medical_imaging_framework.gui.editor"
    echo "  3. Read guide: cat docs/getting-started/TROUBLESHOOTING_INSTALL.md"
    echo ""
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    ❌ FIX FAILED                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Please check:"
    echo "  1. Python version: python3 --version (need 3.8+)"
    echo "  2. Check logs above for specific errors"
    echo "  3. Try: pip install -r requirements.txt"
    echo ""
    exit 1
fi
