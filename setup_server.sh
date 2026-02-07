#!/bin/bash
# Setup script for Node-MedicalImaging-Framework on server
# This script sets up the environment, installs dependencies, and configures the framework

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "   Node-MedicalImaging-Framework - Server Setup Script"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📁 Project directory: $SCRIPT_DIR${NC}"
echo ""

# Check Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python version"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or later"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✅ Found Python: $PYTHON_VERSION${NC}"

# Check if Python version is 3.8+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python 3.8 or later is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo ""

# Create virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Creating virtual environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Using existing venv"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Using existing virtual environment${NC}"
fi
echo ""

# Activate virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Activating virtual environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Upgrading pip"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# Install dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Installing dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Install the framework in editable mode
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Installing Node-MedicalImaging-Framework"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install -e .
echo -e "${GREEN}✅ Node-MedicalImaging-Framework installed in editable mode${NC}"
echo ""

# Create .envrc for direnv (optional)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Setting up environment activation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v direnv &> /dev/null; then
    echo "direnv detected - creating .envrc file"

    cat > .envrc << 'EOF'
# Automatically activate virtual environment when entering this directory
# Used by direnv (https://direnv.net/)

# Activate the virtual environment
source venv/bin/activate

# Set the project root
export PROJECT_ROOT="$PWD"

# Add the project to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Optional: Set environment variables for development
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONUNBUFFERED=1

echo "✅ Node-MedicalImaging-Framework environment activated"
echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python: $(python --version)"
echo "📦 Virtual env: venv/"
EOF

    direnv allow
    echo -e "${GREEN}✅ .envrc created and allowed${NC}"
    echo -e "${BLUE}   Automatic activation will work when you cd into this directory${NC}"
else
    echo -e "${YELLOW}⚠️  direnv not found${NC}"
    echo -e "${BLUE}   Use 'source activate.sh' to manually activate the environment${NC}"
fi
echo ""

# Test the installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Testing installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing package import..."
python3 << EOF
try:
    import medical_imaging_framework
    print("✅ Package imports successfully")
    print(f"   Version: {medical_imaging_framework.__version__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Installation test passed${NC}"
else
    echo -e "${RED}❌ Installation test failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════"
echo "   Setup Complete! 🎉"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ Virtual environment created at: venv/${NC}"
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo -e "${GREEN}✅ Node-MedicalImaging-Framework installed${NC}"
echo -e "${GREEN}✅ Environment configured${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
if command -v direnv &> /dev/null; then
    echo -e "     ${BLUE}cd .. && cd $(basename $SCRIPT_DIR)${NC}"
    echo "     (direnv will auto-activate)"
else
    echo -e "     ${BLUE}source activate.sh${NC}"
fi
echo ""
echo "  2. Run tests:"
echo -e "     ${BLUE}python examples/simple_test.py${NC}"
echo ""
echo "  3. Launch GUI (requires X11 forwarding):"
echo -e "     ${BLUE}python -m medical_imaging_framework.gui.editor${NC}"
echo ""
echo "  4. View documentation:"
echo -e "     ${BLUE}cat docs/INDEX.md${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Deactivate for clean exit (will be reactivated on next cd if direnv is used)
deactivate 2>/dev/null || true
