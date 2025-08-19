import logging
import numpy as np
from typing import Tuple, Optional
import os
import psutil
import gc
import tempfile
import pandas as pd
import SimpleITK as sitk
import uuid

ALLOWED_EXTENSIONS = [".nii.gz", ".nii", ".dcm", ".dicom", ".nrrd"]

logger = logging.getLogger("Dev_logger")

def handle_math_operations(feature_vector, feature_value_mode, mode='divide', epsilon=1e-30):  # toto
    if feature_value_mode == 'REAL_VALUE':
        return feature_vector
    
    # feature_value_mode == 'APPROXIMATE_VALUE'
    # using epsilon instead of zero to prevent division by zero and sqrt of negative values
    if mode == 'divide':
        mask = (feature_vector == 0.)
    elif mode == 'sqrt':
        mask = (feature_vector < 0.)
    elif mode == 'both':
        mask = (feature_vector <= 0.)

    feature_vector[mask] = epsilon
    logger.warning(f"Using epsilon = {epsilon} to prevent mathematical errors.")

    # Clean RAM
    del mask
    gc.collect()
    return feature_vector


# def synthesis_small_RoI(array, array_shape, target_shape=(2, 2, 2)):  # toto  unused

#     # Check if original shape is smaller in all dimensions
#     if all(orig < target for orig, target in zip(array_shape, target_shape)):
#         # Repeat the values to reach desired shape
#         reps = [t // s for s, t in zip(array_shape, target_shape)]
#         new_array = np.tile(array, reps)
#     else:
#         new_array = array  # no expansion needed
#     new_shape = new_array.shape

#     return new_array, new_shape


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_available(required_mb: float) -> bool:
    """
    Check if required memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if memory is available, False otherwise
    """
    available_memory = psutil.virtual_memory().available / 1024 / 1024
    return available_memory > required_mb * 1.5  # 50% buffer


def estimate_array_memory(array_path: str) -> float:
    """
    Estimate memory usage for an array.

    Args:
        array_path: Path to array on disk

    Returns:
        Estimated memory usage in MB
    """
    array = np.load(array_path, mmap_mode='r')
    shape = array.shape
    dtype = array.dtype
    # Clean RAM
    del array
    gc.collect()

    element_size = dtype.itemsize
    total_elements = np.prod(shape)
    return (total_elements * element_size) / (1024 * 1024)


def safe_array_operation(func, *args, **kwargs):
    """
    Safely execute array operations with memory monitoring.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    try:
        return func(*args, **kwargs)
    except MemoryError as e:
        logging.error(f"Memory error in {func.__name__}: {e}")
        # Force garbage collection
        gc.collect()
        raise


def optimize_array_dtype(array_path: str, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Optimize array data type for memory efficiency while preserving precision.

    Args:
        array: Input array
        target_dtype: Target data type (if None, auto-detect)

    Returns:
        Optimized array
    """
    array = np.load(array_path, mmap_mode='r')
    if target_dtype is None:
        # Auto-detect optimal dtype with careful precision preservation
        if array.dtype == np.float64:
            # Check if float32 is sufficient - be more conservative
            if np.all(np.isfinite(array)) and np.max(np.abs(array)) < 3.4e37:
                # Test conversion to ensure no precision loss for the data range
                test_converted = array.astype(np.float32).astype(np.float64)
                relative_error = np.max(np.abs((array - test_converted) / (array + 1e-10)))
                if relative_error < 1e-6:  # Less than 1 part per million error
                    target_dtype = np.float32
        elif array.dtype == np.int64:
            # Check if int32 is sufficient
            if np.max(np.abs(array)) < 2 ** 31:
                target_dtype = np.int32

    if target_dtype and target_dtype != array.dtype:
        try:
            new_array = array.astype(target_dtype)
            new_array_path = save_numpy_on_disk(new_array, prefix='optimized_np', suffix='.npy')
            # Clean RAM
            del array, new_array
            gc.collect()
            # Clean disk
            remove_temp_file(array_path)
            return new_array_path
        except (OverflowError, ValueError):
            logging.warning(f"Could not convert array to {target_dtype}, keeping original dtype")
    new_array_path = save_numpy_on_disk(array, prefix='optimized_np', suffix='.npy')
    # Clean RAM
    del array
    gc.collect()
    # Clean disk
    remove_temp_file(array_path)
    return new_array_path


def log_memory_usage(operation_name: str):
    """
    Log memory usage for debugging.

    Args:
        operation_name: Name of the operation
    """
    memory_mb = get_memory_usage()
    logging.info(f"MEMORY - {operation_name}: {memory_mb:.1f} MB")


def chunked_astype(array, dtype, chunk_size=1000000):
    """Memory-efficient type conversion using memory mapping for large arrays with multiprocessing safety"""
    # For arrays that are large but not extremely large, use in-memory chunking
    if array.size > chunk_size and array.size < 50000000:  # 50M threshold for in-memory processing
        return _chunked_astype_in_memory(array, dtype, chunk_size)
    elif array.size > chunk_size:
        return _chunked_astype_with_tempfile(array, dtype, chunk_size)
    else:
        return array.astype(dtype)


def _chunked_astype_in_memory(array, dtype, chunk_size):
    """In-memory chunked type conversion to avoid file conflicts in multiprocessing."""
    actual_chunk_size = min(chunk_size, array.size // 4)

    # Pre-allocate result array
    result = np.empty(array.shape, dtype=dtype)

    # Process in chunks
    for i in range(0, array.size, actual_chunk_size):
        end_idx = min(i + actual_chunk_size, array.size)
        chunk = array.flat[i:end_idx]
        converted_chunk = chunk.astype(dtype)
        result.flat[i:end_idx] = converted_chunk

        # Force memory cleanup after each chunk
        del chunk, converted_chunk
        gc.collect()

    return result


def _chunked_astype_with_tempfile(array, dtype, chunk_size):
    """File-based chunked type conversion with multiprocessing safety."""
    actual_chunk_size = min(chunk_size, array.size // 4)

    # Create process-safe temporary file
    _, temp_filename = create_process_safe_tempfile("chunked_astype", ".tmp")

    try:
        out = np.memmap(temp_filename, dtype=dtype, mode='w+', shape=array.shape)

        # Process in chunks to avoid memory spikes, ensuring exact conversion
        for i in range(0, array.size, actual_chunk_size):
            end_idx = min(i + actual_chunk_size, array.size)
            chunk = array.flat[i:end_idx]
            # Ensure exact type conversion without loss
            converted_chunk = chunk.astype(dtype)
            out.flat[i:end_idx] = converted_chunk

            # Force memory cleanup after each chunk
            del chunk, converted_chunk
            gc.collect()

        # Return a regular array copy to avoid memory mapping issues
        result = np.array(out)

        # Properly close and delete the memory map
        del out
        gc.collect()

        # Safe file deletion with retry mechanism
        _safe_delete_temp_file(temp_filename)

        return result

    except Exception as e:
        # Ensure cleanup even if error occurs
        try:
            _safe_delete_temp_file(temp_filename)
        except:
            pass
        raise e

def save_ml_result(structure, base_path):
    """
    Recursively creates folders and files based on the given structure.

    Parameters:
    - structure (dict): The output from the create_output() function.
    - base_path (str): The base directory where the folder/file structure will be created.
    """

    def process_node(node, current_path):
        node_name = node.get("name", "unnamed")
        node_type = node.get("type", "xlsx")

        # Process folder nodes
        if node_type == "folder":
            folder_path = os.path.join(current_path, node_name)
            os.makedirs(folder_path, exist_ok=True)
            children = node.get("children", [])
            # In case children is a dict instead of a list, convert it to list format
            if isinstance(children, dict):
                children = [{"name": key, "value": value} for key, value in children.items()]
            for child in children:
                process_node(child, folder_path)

        # Process xlsx file nodes
        elif node_type == "xlsx":
            file_name = f"{node_name}.xlsx"
            file_path = os.path.join(current_path, file_name)
            children = node.get("children", None)

            with pd.ExcelWriter(file_path) as writer:
                if not children:
                    if "value" in node:
                        try:
                            df = pd.DataFrame(node["value"])
                            df.to_excel(writer, index=False)
                        except Exception:
                            if type(node["value"]) != list:
                                pd.DataFrame([node["value"]]).to_excel(writer, sheet_name="sheet1", index=False)
                            else:
                                pd.DataFrame(node["value"]).to_excel(writer, sheet_name="sheet1", index=False)
                    else:
                        logger.log(300, f"{node_name} has empty result or not found.")
                        # pd.DataFrame().to_excel(writer, index=False)
                elif isinstance(children, list):
                    # Multiple sheets: each child in the list is a sheet.
                    for sheet in children:
                        sheet_name = sheet.get("name", "Sheet1")
                        sheet_value = sheet.get("value", None)
                        if sheet_value is not None:
                            if isinstance(sheet_value, pd.DataFrame):
                                sheet_value.to_excel(writer, sheet_name=sheet_name, index=False)
                            else:
                                try:
                                    df = pd.DataFrame(sheet_value)
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                                except Exception:
                                    if type(sheet_value) != list:
                                        pd.DataFrame([sheet_value]).to_excel(writer, sheet_name=sheet_name, index=False)
                                    else:
                                        pd.DataFrame(sheet_value).to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(children, dict):
                    # If children is a dict, treat it as data for a single sheet.
                    # Use the node's name as the sheet name.
                    sheet_name = node.get("name", "Sheet1")

                    # Helper to convert a dict to a DataFrame
                    def dict_to_df(d):
                        # If the values are scalars, wrap in a list to form a single row.
                        if all(not isinstance(v, list) for v in d.values()):
                            return pd.DataFrame([d])
                        else:
                            return pd.DataFrame(d)

                    try:
                        df = dict_to_df(children)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception:
                        if type(children) != list:
                            pd.DataFrame([children]).to_excel(writer, sheet_name=sheet_name, index=False)
                        else:
                            pd.DataFrame(children).to_excel(writer, sheet_name=sheet_name, index=False)

        # Process other file types as text files
        else:
            file_name = f"{node_name}.txt"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, "w") as f:
                f.write(str(node.get("value", "")))

    root = structure[0].get("out")
    if root:
        process_node(root, base_path)
    else:
        raise ValueError("The structure does not contain an 'out' key.")

def _safe_delete_temp_file(filename: str, max_retries: int = 5, delay: float = 0.1):
    """
    Safely delete temporary file with retry mechanism for Windows multiprocessing.

    Args:
        filename: Path to temporary file
        max_retries: Maximum number of deletion attempts
        delay: Delay between attempts in seconds
    """
    import time

    for attempt in range(max_retries):
        try:
            if os.path.exists(filename):
                # Force garbage collection before deletion
                gc.collect()
                time.sleep(delay)  # Small delay to ensure file handles are released
                os.unlink(filename)
                return  # Success
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                # Wait longer on each retry
                time.sleep(delay * (attempt + 1))
                continue
            else:
                # On final attempt, log warning but don't raise exception
                logging.warning(f"Could not delete temporary file {filename} after {max_retries} attempts: {e}")
                logging.warning("File will be cleaned up by OS eventually")
                # Register for cleanup at exit as last resort
                try:
                    import atexit
                    atexit.register(lambda: _cleanup_file_at_exit(filename))
                except:
                    pass


def _cleanup_file_at_exit(filename: str):
    """Cleanup temporary file at program exit."""
    try:
        if os.path.exists(filename):
            os.unlink(filename)
    except:
        pass  # Silent cleanup at exit


def memory_efficient_unique(array_path: str) -> np.ndarray:
    """
    Memory-efficient unique value detection that preserves exact results.

    Args:
        array: Input array
        preserve_exact_results: If True, ensures exact same results as np.unique

    Returns:
        Unique values (exact same as np.unique when preserve_exact_results=True)
    """
    array = np.load(array_path, mmap_mode='r')
    # Always preserve exact results by default to maintain consistency
    uniques = np.unique(array)
    # Clean RAM
    del array
    gc.collect()
    return uniques

def validate_data_integrity(original_path: str, optimized_path: str,
                            tolerance: float = 1e-10) -> bool:
    """
    Validate that optimized array maintains data integrity.

    Args:
        original: Original array
        optimized: Optimized array
        tolerance: Maximum allowed relative difference

    Returns:
        True if data integrity is preserved
    """
    original = np.load(original_path, mmap_mode='r')
    optimized = np.load(optimized_path, mmap_mode='r')

    if original.shape != optimized.shape:
        logging.error(f"Shape mismatch: original {original.shape} vs optimized {optimized.shape}")
        # Clean RAM
        del original, optimized
        gc.collect()
        return False

    # Check for exact equality first
    if np.array_equal(original, optimized):
        # Clean RAM
        del original, optimized
        gc.collect()
        return True

    # Check relative differences for floating point arrays
    if original.dtype.kind in ['f', 'c'] or optimized.dtype.kind in ['f', 'c']:
        # Avoid division by zero
        denominator = np.abs(original) + 1e-15
        relative_diff = np.abs(original - optimized) / denominator
        max_diff = np.max(relative_diff)

        if max_diff > tolerance:
            logging.warning(f"Data integrity check failed: max relative difference {max_diff} > tolerance {tolerance}")
            # Clean RAM
            del original, optimized
            gc.collect()
            return False
    else:
        # For integer arrays, check exact equality
        if not np.array_equal(original, optimized):
            logging.warning("Integer arrays are not exactly equal after optimization")
            # Clean RAM
            del original, optimized
            gc.collect()
            return False
    # Clean RAM
    del original, optimized
    gc.collect()
    return True


def get_process_safe_temp_dir(prefix):
    """
    Get a process-safe temporary directory for multiprocessing.

    Returns:
        Path to process-safe temporary directory
    """
    # base_temp_dir = tempfile.gettempdir()
    base_temp_dir = create_tmp_dir()        # toto
    process_temp_dir = os.path.join(
        base_temp_dir, f"radiomics_proc_{os.getpid()}"
    )

    # Create directory if it doesn't exist
    os.makedirs(process_temp_dir, exist_ok=True)

    # Register for cleanup at exit
    import atexit
    atexit.register(lambda: _cleanup_temp_dir(process_temp_dir))

    return process_temp_dir

# create tmp folder in the root directory
def create_tmp_dir() -> str:
    """
    Create a tmp folder in the root directory.

    Returns:
        Path to the tmp folder
    """
    # Get the main directory
    base_temp_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(base_temp_dir, exist_ok=True)
    return base_temp_dir

# def create_process_safe_tempfile(prefix="", suffix=".tmp"):
#     """
#     Create a unique filename inside a process-safe temp directory.
#     Returns:
#         the full path as a string.
#     """
#     temp_dir = get_process_safe_temp_dir()
#     unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
#     full_path = os.path.join(temp_dir, unique_name)
#     return full_path

def _cleanup_temp_dir(temp_dir):
    """Clean up temporary directory at exit."""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass  # Silent cleanup


def create_process_safe_tempfile(prefix="temp", suffix=".tmp"):
    """
    Create a process-safe temporary file that won't conflict with other processes.

    Args:
        prefix: File prefix
        suffix: File suffix

    Returns:
        Tuple of (file_handle, filename)
    """
    temp_dir = get_process_safe_temp_dir(prefix)
    filename = os.path.join(
        temp_dir,
        f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    )
    # Create and immediately close the file
    with open(filename, 'wb') as f:
        pass

    return None, filename

def save_numpy_on_disk(array, prefix="", suffix=".npy", custom_path=None):         #toto
    # Save mask on disk
    if custom_path:     # Saves the array on desired file
        # Get tempprary path
        _, temp_path = create_process_safe_tempfile(prefix='temp', suffix=suffix)

        # Safe overwrite
        np.save(temp_path, array)
        os.replace(temp_path, custom_path)

        return custom_path
    else:
        _, array_path = create_process_safe_tempfile(prefix, suffix)
        np.save(array_path, array)

        return array_path


def save_rt_on_disk(rt: sitk.Image, prefix="", suffix=".nii.gz", custom_path=None):         # toto
    if custom_path:
        _, temp_path = create_process_safe_tempfile(prefix="temp", suffix=suffix)
        sitk.WriteImage(rt, temp_path)
        os.replace(temp_path, custom_path)
        return custom_path
    else:
        _, temp_path = create_process_safe_tempfile(prefix=prefix, suffix=suffix)
        sitk.WriteImage(rt, temp_path)
        return temp_path

def remove_temp_file(file_path):

    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error occurred while trying to remove the file: {e}")