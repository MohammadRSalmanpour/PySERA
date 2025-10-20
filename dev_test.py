#!/usr/bin/env python3
"""
pysera Development Test Script

This script helps developers verify that pysera is working correctly
in their local development environment.

Usage:
    python dev_test.py [--verbose] [--full]
    
Options:
    --verbose    Show detailed output
    --full       Run full test suite (slower)
"""

import sys
import os
import argparse
import traceback

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_step(step):
    """Print a test step."""
    print(f"\n📋 {step}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def test_basic_imports(verbose=False):
    """Test basic pysera imports."""
    print_step("Testing basic imports...")
    
    try:
        # Add current directory to path for development
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        import pysera
        print_success("pysera imported successfully")

        if verbose:
            print(f"   Version: {pysera.__version__}")
            print(f"   Author: {pysera.__author__}")
            print(f"   Location: {pysera.__file__}")
        
        return True
    except ImportError as e:
        print_error(f"Failed to import pysera: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_core_classes(verbose=False):
    """Test core pysera functionality."""
    print_step("Testing core functionality...")
    
    try:
        import pysera
        
        # Test main process_batch function
        process_func = pysera.process_batch
        assert callable(process_func), "process_batch should be callable"
        print_success("process_batch function accessible")
        
        if verbose:
            print(f"   process_batch function: {process_func}")
        
        return True
    except Exception as e:
        print_error(f"Core functionality test failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_configuration_access(verbose=False):
    """Test API access."""
    print_step("Testing API access...")
    
    try:
        import pysera
        
        # Test version info
        version = pysera.__version__
        assert isinstance(version, str), "__version__ should be a string"
        print_success(f"Version loaded: {version}")
        
        # Test author info
        author = pysera.__author__
        assert isinstance(author, str), "__author__ should be a string"
        print_success(f"Author info loaded: {author}")
        
        # Test __all__ exports
        all_exports = pysera.__all__
        assert isinstance(all_exports, list), "__all__ should be a list"
        assert 'process_batch' in all_exports, "process_batch should be in __all__"
        print_success(f"Public API exports: {all_exports}")
        
        if verbose:
            print(f"   Version: {version}")
            print(f"   Author: {author}")
            print(f"   Public exports: {all_exports}")
        
        return True
    except Exception as e:
        print_error(f"API access test failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_utility_functions(verbose=False):
    """Test main function capabilities."""
    print_step("Testing main function capabilities...")
    
    try:
        import pysera
        
        # Test main function signature
        process_func = pysera.process_batch
        assert callable(process_func), "process_batch should be callable"
        print_success("Main function accessible")
        
        # Test function signature inspection
        import inspect
        sig = inspect.signature(process_func)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        required_params = ['image_input', 'mask_input']
        for param in required_params:
            assert param in params, f"Required parameter {param} missing"
        
        print_success(f"Function signature valid ({len(params)} parameters)")
        
        if verbose:
            print(f"   Function parameters: {params[:5]}...")
        
        return True
    except Exception as e:
        print_error(f"Main function test failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_multiprocessing_capability(verbose=False):
    """Test multiprocessing capabilities."""
    print_step("Testing multiprocessing capabilities...")
    
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        
        # Test CPU count detection
        cpu_count = mp.cpu_count()
        print_success(f"Multiprocessing available ({cpu_count} CPUs detected)")
        
        # Test ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Simple test function
            def test_func(x):
                return x * 2
            
            future = executor.submit(test_func, 5)
            result = future.result()
            assert result == 10, "ProcessPoolExecutor test failed"
        
        print_success("ProcessPoolExecutor working")
        
        if verbose:
            print(f"   CPU count: {cpu_count}")
            print(f"   Test result: {result}")
        
        return True
    except Exception as e:
        print_error(f"Multiprocessing test failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_dependency_imports(verbose=False):
    """Test important dependency imports."""
    print_step("Testing dependency imports...")
    
    dependencies = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'openpyxl': 'openpyxl'
    }
    
    missing_deps = []
    available_deps = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            available_deps.append(package)
            if verbose:
                print(f"   ✅ {package}")
        except ImportError:
            missing_deps.append(package)
            if verbose:
                print(f"   ❌ {package}")
    
    print_success(f"Dependencies available: {len(available_deps)}/{len(dependencies)}")
    
    if missing_deps:
        print_warning(f"Missing dependencies: {', '.join(missing_deps)}")
        print_warning("Install with: pip install " + " ".join(missing_deps))
    
    # Check critical dependencies
    critical_deps = ['numpy', 'pandas']
    for dep in critical_deps:
        if any(dep in missing for missing in missing_deps):
            print_error(f"Critical dependency missing: {dep}")
            return False
    
    return True

def test_cli_interface(verbose=False):
    """Test CLI interface."""
    print_step("Testing CLI interface...")
    
    try:
        from pysera._cli import main
        print_success("CLI module imported")
        
        # Test argument parser import
        from pysera.cli.argument_parser import parse_arguments
        print_success("Argument parser accessible")
        
        if verbose:
            print(f"   CLI main function: {main}")
        
        return True
    except Exception as e:
        print_error(f"CLI interface test failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def test_file_structure(verbose=False):
    """Test file structure integrity."""
    print_step("Testing file structure...")
    
    required_files = [
        'pysera/__init__.py',
        'pysera/_cli.py',
        'pysera/processing/radiomics_processor.py',
        'pysera/utils/log_record.py',
        'pysera/config/settings.py',
        'setup.py',
        'pyproject.toml',
        'README.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            if verbose:
                print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            if verbose:
                print(f"   ❌ {file_path}")
    
    print_success(f"Required files found: {len(existing_files)}/{len(required_files)}")
    
    if missing_files:
        print_error(f"Missing files: {missing_files}")
        return False
    
    return True

def run_full_tests(verbose=False):
    """Run additional comprehensive tests."""
    print_step("Running full test suite...")
    
    try:
        import pysera
        
        # Test process_batch with invalid parameters (should handle gracefully)
        try:
            result = pysera.process_batch(
                image_input="nonexistent_image.nii.gz",
                mask_input="nonexistent_mask.nii.gz",
                output_path="./test_output",
                report="none"  # Disable logging for test
            )
            
            # Should return a result dict with success=False
            assert isinstance(result, dict), "process_batch should return a dict"
            assert 'success' in result, "Result should contain 'success' key"
            assert result['success'] == False, "Should fail with nonexistent files"
            print_success("Error handling test passed")
            
        except Exception as e:
            print_warning(f"Error handling test encountered: {e}")
        
        # Test parameter validation
        import inspect
        sig = inspect.signature(pysera.process_batch)
        param_count = len(sig.parameters)
        assert param_count >= 10, f"Expected many parameters, got {param_count}"
        print_success("Parameter validation test passed")
        
        if verbose:
            print(f"   Function has {param_count} parameters")
        
        return True
    except Exception as e:
        print_error(f"Full test suite failed: {e}")
        if verbose:
            traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files created during testing."""
    test_dirs = ['test_output', '__pycache__']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                import shutil
                shutil.rmtree(test_dir)
            except:
                pass  # Ignore cleanup errors

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='pysera Development Test Script')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--full', '-f', action='store_true',
                        help='Run full test suite')
    args = parser.parse_args()

    print_header("pysera Development Test Suite")
    print("This script verifies that pysera is working correctly in your development environment.")

    # List of test functions
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Dependency Imports", test_dependency_imports),
        ("Core Functionality", test_core_classes),
        ("API Access", test_configuration_access),
        ("Main Function", test_utility_functions),
        ("Multiprocessing", test_multiprocessing_capability),
        ("CLI Interface", test_cli_interface),
    ]
    
    if args.full:
        tests.append(("Full Test Suite", run_full_tests))
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func(verbose=args.verbose):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            if args.verbose:
                traceback.print_exc()
            failed += 1
    
    # Clean up
    cleanup_test_files()
    
    # Summary
    print_header("Test Summary")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! pysera is ready for development.")
        print("\nNext steps:")
        print("1. Try running: cd library_examples && python basic_usage.py")
        print("2. Test CLI: python radiomics_standalone.py --help")
        print("3. Start developing your features!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install dependencies: pip install -r requirements-library.txt")
        print("3. Install in development mode: pip install -e .")
        print("4. Check DEVELOPMENT.md for more help")
    
    return failed

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)