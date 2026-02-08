#!/bin/bash
# Test script to verify automatic environment activation

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Testing Automatic Environment Activation                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check 1: direnv installed
echo "1. Checking if direnv is installed..."
if command -v direnv &> /dev/null; then
    echo "   ✅ direnv is installed: $(which direnv)"
else
    echo "   ❌ direnv is NOT installed"
    echo "   Install with: sudo apt-get install direnv"
    exit 1
fi
echo ""

# Check 2: direnv hook in .bashrc
echo "2. Checking if direnv hook is configured..."
if grep -q "direnv hook bash" ~/.bashrc; then
    echo "   ✅ direnv hook found in ~/.bashrc"
else
    echo "   ❌ direnv hook NOT found in ~/.bashrc"
    echo "   Add this line to ~/.bashrc:"
    echo "   eval \"\$(direnv hook bash)\""
    exit 1
fi
echo ""

# Check 3: .envrc exists
echo "3. Checking if .envrc file exists..."
if [ -f ".envrc" ]; then
    echo "   ✅ .envrc file exists"
else
    echo "   ❌ .envrc file NOT found"
    exit 1
fi
echo ""

# Check 4: .envrc is allowed
echo "4. Checking if .envrc is allowed..."
direnv status | grep -q "Found RC allowed true"
if [ $? -eq 0 ]; then
    echo "   ✅ .envrc is allowed"
else
    echo "   ⚠️  .envrc needs to be allowed"
    echo "   Running: direnv allow"
    direnv allow
fi
echo ""

# Check 5: Virtual environment exists
echo "5. Checking if virtual environment exists..."
if [ -d "venv" ]; then
    echo "   ✅ venv/ directory exists"
else
    echo "   ❌ venv/ directory NOT found"
    echo "   Create with: python -m venv venv"
    exit 1
fi
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Reload your shell to activate direnv hook:"
echo "   exec bash"
echo "   OR"
echo "   source ~/.bashrc"
echo ""
echo "2. Test automatic activation:"
echo "   cd .."
echo "   cd Node-MedicalImaging-Framework"
echo ""
echo "3. You should see this message:"
echo "   ✅ Node-MedicalImaging-Framework environment activated"
echo "   📁 Project root: /home/jin.park/Codes/Node-MedicalImaging-Framework"
echo "   🐍 Python: Python 3.10.12"
echo "   📦 Virtual env: venv/"
echo ""
echo "4. Verify environment is active:"
echo "   which python  # Should show: .../venv/bin/python"
echo "   echo \$PROJECT_ROOT  # Should show project path"
echo ""
echo "💡 If auto-activation still doesn't work, use manual activation:"
echo "   source activate.sh"
echo ""
