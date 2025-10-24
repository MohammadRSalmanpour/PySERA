""" functions to synthesize small RoIs (4<= , <=10)"""
import logging
import numpy as np
import random

logger = logging.getLogger("Dev_logger")

def synthesis_small_RoI(volume, mask, background='NaN', target_num=10):
    """
    Expand a small region of interest (RoI) by adding jittered points
    until the desired number of points is reached.

    Args:
        volume (int): Volume of the mask.
        mask (np.ndarray): 3D binary mask (non-zero values indicate RoI).
        target_num (int): Desired total number of points in the RoI. Default is 10.
        jitter (int): Jitter range to apply around existing points. Default is 1.
        on_disk : If the mask is saved on disk use True. Default is False, which means the mask is on RAM.

    Returns:
        tuple: (new volume count, updated mask)
    """
    required_points = target_num - volume
    if np.size(mask) < target_num:
        mask = pad_to_target(mask, background, target_num)
    
    # Get coordinates of voxels with value
    if background=='NaN':
        roi = np.where(~np.isnan(mask))
        empties = np.where(np.isnan(mask))
    else:
        roi = np.where(mask>0.)
        empties = np.where(mask==0.)
    roi_coords = list(zip(*roi))
    empty_coords = list(zip(*empties))


    added = 0
    attempts = 0
    max_attempts = 6 * required_points
    base_idx = 0
    while added < required_points  and attempts < max_attempts:
        # Get an already existing coord
        (x0, y0, z0) = roi_coords[base_idx % len(roi_coords)]

        # Find potential coord in the neighbourhood of (x0, y0, z0)
        (x1, y1, z1) = get_jittered_coords((x0, y0, z0))

        if (x1, y1, z1) in empty_coords:
            mask[x1, y1, z1] = mask[x0, y0, z0]

            roi_coords.append((x1, y1, z1))
            empty_coords.pop(empty_coords.index((x1, y1, z1)))

            added += 1

        attempts += 1
        base_idx += 1
    
    new_volume = volume+added

    return mask

def pad_to_target(arr, background, target_size):
    shape = list(arr.shape)

    def size(s): return s[0] * s[1] * s[2]

    while size(shape) < target_size:
        # Find dimension with smallest current size (ties broken by index)
        dim_to_increase = min(range(3), key=lambda d: shape[d])
        shape[dim_to_increase] += 1
    # Create new array filled with nan
    if background == 'NaN':
        new_arr = np.full(shape, np.nan, dtype=float)
    elif background == 0:
        new_arr = np.zeros(shape, dtype=float)
    # Copy old values into the new array
    slices = tuple(slice(0, old) for old in arr.shape)
    new_arr[slices] = arr
    return new_arr

def get_jittered_coords(coord0):
    (x0, y0, z0) = coord0
    random.seed(42)
    jitter = random.choices(range(0, 2), k=3)
    (x1, y1, z1) = (x0+jitter[0], y0+jitter[1], z0+jitter[2])       # new possible coordinate
    
    return (x1, y1, z1)


def synthesize_coords(arr: np.ndarray, num_coords, dim, target_num=4):
    extra_needed = target_num - num_coords
    # Offsets per dimension
    random.seed(42)
    base_offsets = random.choices(range(0, 2), k=dim)
    
    synthetic_points = []
    for i in range(extra_needed):
        offset = [(i + 1) * base_offsets[d] for d in range(dim)]
        new_point = arr.max(axis=0) + offset
        synthetic_points.append(new_point)
    return np.vstack([arr, np.array(synthetic_points, dtype=float)])
    

def synthesize_values(arr: np.ndarray, target_num=2):

    arr = np.asarray(arr)
    if len(arr) < target_num:
        last_val = arr[-1]
        num_to_add = target_num - len(arr)
        eps = np.finfo(np.float32).eps
        new_vals = last_val + eps * np.arange(1, num_to_add + 1)
        arr = np.concatenate([arr, new_vals])

    return arr
