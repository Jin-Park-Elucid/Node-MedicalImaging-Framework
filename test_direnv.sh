#!/bin/bash
# Test script to verify direnv automatic activation

echo "============================================"
echo "Testing direnv automatic activation"
echo "============================================"
echo ""

# Test 1: Check direnv is installed
echo "Test 1: Check direnv installation"
if command -v direnv &> /dev/null; then
    echo "✅ direnv is installed: $(which direnv)"
else
    echo "❌ direnv is NOT installed"
    exit 1
fi
echo ""

# Test 2: Check .envrc exists
echo "Test 2: Check .envrc file"
if [ -f "/home/jinhyeongpark/Codes/Node_DL_MedicalImaging/.envrc" ]; then
    echo "✅ .envrc file exists"
else
    echo "❌ .envrc file NOT found"
    exit 1
fi
echo ""

# Test 3: Check .envrc is allowed
echo "Test 3: Check .envrc is allowed by direnv"
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
if direnv status 2>&1 | grep -q "Found RC allowed true"; then
    echo "✅ .envrc is allowed"
else
    echo "⚠️  .envrc is NOT allowed yet"
    echo "   Run: direnv allow"
fi
echo ""

# Test 4: Check venv exists
echo "Test 4: Check virtual environment"
if [ -d "/home/jinhyeongpark/Codes/Node_DL_MedicalImaging/venv" ]; then
    echo "✅ venv directory exists"
    if [ -f "/home/jinhyeongpark/Codes/Node_DL_MedicalImaging/venv/bin/activate" ]; then
        echo "✅ venv/bin/activate exists"
    else
        echo "❌ venv/bin/activate NOT found"
    fi
else
    echo "❌ venv directory NOT found"
fi
echo ""

# Test 5: Check direnv hook in shell
echo "Test 5: Check direnv hook in shell config"
if grep -q "direnv hook" ~/.bashrc 2>/dev/null; then
    echo "✅ direnv hook found in ~/.bashrc"
else
    echo "⚠️  direnv hook NOT found in ~/.bashrc"
    echo "   Add this line to ~/.bashrc:"
    echo "   eval \"\$(direnv hook bash)\""
fi
echo ""

# Test 6: Manual activation test
echo "Test 6: Manual activation test"
cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging
eval "$(direnv export bash)" 2>&1
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ VIRTUAL_ENV is set: $VIRTUAL_ENV"
    echo "✅ Python location: $(which python)"
    echo "✅ Python version: $(python --version)"
else
    echo "⚠️  VIRTUAL_ENV is not set"
    echo "   This is normal in non-interactive shells"
fi
echo ""

echo "============================================"
echo "Summary:"
echo "============================================"
echo "To test automatic activation manually:"
echo "1. Open a new terminal"
echo "2. cd /home/jinhyeongpark/Codes/Node_DL_MedicalImaging"
echo "3. You should see the activation message"
echo "4. Run: echo \$VIRTUAL_ENV"
echo "5. It should show the venv path"
echo ""
