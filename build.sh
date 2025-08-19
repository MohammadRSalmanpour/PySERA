#!/bin/bash

# PySera Build Script
# This script helps build and distribute the PySera library

set -e  # Exit on any error

echo "🚀 PySera Build Script"
echo "====================="

# Function to print colored output
print_step() {
    echo -e "\n📋 $1"
}

print_success() {
    echo -e "✅ $1"
}

print_error() {
    echo -e "❌ $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    print_error "Error: setup.py or pyproject.toml not found. Run this from the PySera root directory."
    exit 1
fi

# Check Python version
print_step "Checking Python version..."
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python version is compatible (3.8+)"
else
    print_error "Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Clean previous builds
print_step "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
print_success "Cleaned build directories"

# Install build dependencies
print_step "Installing build dependencies..."
python -m pip install --upgrade pip setuptools wheel build twine
print_success "Build dependencies installed"

# Build source distribution
print_step "Building source distribution..."
python -m build --sdist
print_success "Source distribution built"

# Build wheel distribution
print_step "Building wheel distribution..."
python -m build --wheel
print_success "Wheel distribution built"

# List built files
print_step "Built files:"
ls -la dist/

# Verify the built packages
print_step "Verifying built packages..."
python -m twine check dist/*
print_success "Package verification passed"

# Optional: Install in development mode for testing
if [ "$1" = "--install" ]; then
    print_step "Installing in development mode..."
    python -m pip install -e .
    print_success "Development installation completed"
    
    # Test the installation
    print_step "Testing installation..."
    python -c "import pysera; print(f'✓ PySera {pysera.__version__} imported successfully')"
    print_success "Installation test passed"
fi

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "📦 Distribution files are in the 'dist/' directory:"
echo "   - Source distribution (.tar.gz)"
echo "   - Wheel distribution (.whl)"
echo ""
echo "🚀 To install the built package:"
echo "   pip install dist/PySera-*.whl"
echo ""
echo "📤 To upload to PyPI (when ready):"
echo "   python -m twine upload dist/*"
echo ""
echo "🔍 To install in development mode:"
echo "   pip install -e ."
echo ""
echo "📚 Next steps:"
echo "   1. Test the package: python -c 'import pysera; print(pysera.__version__)'"
echo "   2. Run examples: cd examples && python basic_usage.py"
echo "   3. Read documentation: cat README.md"