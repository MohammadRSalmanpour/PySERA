# PySera Installation Guide

This guide provides step-by-step instructions for installing the PySera library.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation Methods

### Method 1: Direct Installation

```bash
# Create virtual environment (recommended)
python3 -m venv pysera-env
source pysera-env/bin/activate

# Install PySera
pip install pysera

# Development tools
pip install pysera[dev]   
```

### Method 2: Local Installation from Source

If you have the source code locally:

```bash
# Navigate to the PySera directory
cd /path/to/pysera

# Create virtual environment
python3 -m venv pysera-env
source pysera-env/bin/activate

# Install dependencies
pip install numpy pandas scipy SimpleITK nibabel pydicom pynrrd scikit-image opencv-python PyWavelets mahotas scikit-learn matplotlib Pillow openpyxl psutil pyhull vtk itk rt-utils dcmrtstruct2nii ReliefF sklearn-relief scikit-optimize scikit-learn-extra kmodes

# Install PySera
pip install .
```

## Dependency Installation

### Core Dependencies
```bash
pip install numpy==1.26.4 pandas>=2.3.1 scipy>=1.15.3 SimpleITK>=2.5.2 nibabel>=5.3.2 pydicom>=3.0.1 pynrrd>=1.1.3 opencv-python>=4.11.0.86 scikit-image>=0.25.2 PyWavelets>=1.8.0 mahotas>=1.4.18 scikit-learn>=1.7 matplotlib>=3.10.5 Pillow>=11.3.0 openpyxl>=3.1.5 psutil>=7.0.0 pyhull>=2015.2.1
```


## Testing Installation

### Basic Test
```bash
python -c "import pysera; print('✓ Import successful')"
```

### Functionality Test
```python
import pysera

# Test basic functionality
processor = pysera.RadiomicsProcessor(output_path="./test_output")
print(f"✓ RadiomicsProcessor created")

# Test logging
logger, handler = pysera.setup_logging()
print(f"✓ Logging setup successful")

# Test configuration
print(f"✓ Available feature modes: {len(pysera.FEATURE_EXTRACTION_MODES)}")
```

### Run Examples
```bash
cd library_examples
python basic_usage.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   pip install -r requirements-library.txt
   ```

2. **Permission Error**: Use virtual environment
   ```bash
   python3 -m venv pysera-env
   source pysera-env/bin/activate
   pip install -e .
   ```

3. **Import Error**: Ensure you're in the correct directory and virtual environment

4. **System Package Manager Warning**: Use virtual environment or add `--break-system-packages` (not recommended)

### Environment-Specific Instructions

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-venv python3-pip

# Continue with standard installation
```

#### macOS
```bash
# Using Homebrew
brew install python

# Continue with standard installation
```

#### Windows
```bash
# Use Python installer from python.org
# Then continue with standard installation
```

## Verification

After installation, verify PySera is working:

```python
#!/usr/bin/env python3
import pysera

print(f"PySera version: {pysera.__version__}")
print(f"Available classes: {pysera.__all__[:5]}...")  # Show first 5 exports

# Test processor creation
processor = pysera.RadiomicsProcessor()
print("✓ RadiomicsProcessor works")

# Test logging
memory_handler = pysera.MemoryLogHandler()
logger, handler = pysera.setup_logging(memory_handler)
print("✓ Logging system works")

print("🎉 PySera is ready to use!")
```

## Support

If you encounter installation issues:

1. Check Python version: `python3 --version` (should be 3.8+)
2. Check pip version: `pip --version`
3. Try installing in a fresh virtual environment
4. Check the GitHub issues page for known problems
5. Create a new issue with your error details

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage examples
2. Check the `examples/` directory for code samples
3. Review the API documentation
4. Try processing your first radiomics dataset!

---

*Happy radiomics processing with PySera! 🚀*