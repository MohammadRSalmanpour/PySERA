#!/bin/bash

# PySera Development Environment Setup Script
# This script sets up a local development environment for PySera

set -e  # Exit on any error

echo "🚀 PySera Development Environment Setup"
echo "======================================"

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

print_warning() {
    echo -e "⚠️  $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    print_error "Error: setup.py or pyproject.toml not found. Run this from the PySera root directory."
    exit 1
fi

# Check Python version
print_step "Checking Python version..."
if command -v python >/dev/null 2>&1; then
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python version: $python_version"
    
    if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version is compatible (3.8+)"
    else
        print_error "Python 3.8+ required. Current version: $python_version"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
print_step "Setting up virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel
print_success "Pip upgraded"

# Install dependencies
print_step "Installing dependencies..."
if [ -f "requirements-library.txt" ]; then
    pip install -r requirements-library.txt
    print_success "Library dependencies installed"
else
    print_warning "requirements-library.txt not found, installing basic dependencies..."
    pip install numpy pandas scipy matplotlib openpyxl
fi

# Install PySera in development mode
print_step "Installing PySera in development mode..."
pip install -e .
print_success "PySera installed in development mode"

# Install optional development tools
print_step "Installing development tools..."
pip install pytest black isort flake8 mypy
print_success "Development tools installed"

# Run basic tests
print_step "Running basic tests..."
python dev_test.py
test_result=$?

if [ $test_result -eq 0 ]; then
    print_success "Basic tests passed"
else
    print_warning "Some tests failed, but setup is complete"
fi

# Create test data directory
print_step "Creating test data directory..."
mkdir -p test_data
echo "# Add your test images and masks here" > test_data/README.md
print_success "Test data directory created"

# Display summary
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Summary:"
echo "   • Virtual environment: venv/"
echo "   • PySera installed in development mode"
echo "   • Dependencies installed"
echo "   • Development tools installed"
echo "   • Test data directory created"
echo ""
echo "🚀 Next steps:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Run tests: python dev_test.py"
echo "   3. Try examples: cd examples && python basic_usage.py"
echo "   4. Start developing!"
echo ""
echo "📚 Useful commands:"
echo "   • Run tests: python dev_test.py --verbose"
echo "   • Test CLI: python radiomics_standalone.py --help"
echo "   • Format code: black pysera/"
echo "   • Check style: flake8 pysera/"
echo "   • Build package: ./build.sh"
echo ""
echo "📖 Documentation:"
echo "   • Development guide: DEVELOPMENT.md"
echo "   • Main README: README.md"
echo "   • Installation guide: INSTALL.md"
echo ""

# Create activation reminder
cat > activate_dev.sh << 'EOF'
#!/bin/bash
# Quick activation script for PySera development environment
source venv/bin/activate
echo "🚀 PySera development environment activated!"
echo "Run 'python dev_test.py' to verify everything works."
EOF

chmod +x activate_dev.sh
print_success "Created activation script: activate_dev.sh"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    print_step "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Test files
test_output/
test_data/*.nii*
test_data/*.dcm
test_data/*.npy

# Logs
*.log
EOF
    print_success ".gitignore created"
fi

echo "✨ Setup complete! Happy developing with PySera! ✨"