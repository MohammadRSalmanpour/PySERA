# -*- coding: utf-8 -*-
# data_manager/resampling/base_resizer.py

from __future__ import annotations

import numpy as np
from typing import Sequence, Union
from abc import ABC, abstractmethod


class BaseResizer(ABC):
    """
    Abstract 3D resampler interface (IBSI-friendly).

    Notes
    -----
    - Accept/return NumPy arrays only (RAM-only implementation).
    - Interpolation must support at least: 'nearest', 'linear', 'cubic'.
    """

    def __init__(self, save_to_disk: bool = False) -> None:
        self.save_to_disk = bool(save_to_disk)

    @staticmethod
    def _validate_input(volume: Union[str, np.ndarray]) -> np.ndarray:
        if not isinstance(volume, np.ndarray):
            raise TypeError("This resizer expects a 3D numpy.ndarray input.")
        if volume.ndim != 3 or volume.size == 0:
            raise ValueError("Input array must be a non-empty 3D array.")
        return volume

    @abstractmethod
    def resize(
        self,
        volume: Union[str, np.ndarray],
        original_spacing: Sequence[float],
        target_spacing: Sequence[float],
        interpolation: str = "linear",
        align_to_center: bool = True,
    ) -> np.ndarray:
        """
        Resample a 3D volume from original_spacing to target_spacing.

        Parameters
        ----------
        volume : ndarray
            3D input array.
        original_spacing : (sx, sy, sz)
            Spacing in mm.
        target_spacing : (tx, ty, tz)
            Desired spacing in mm.
        interpolation : str
            'nearest' | 'linear' | 'cubic'
        align_to_center : bool
            Use 'align grid centers' (IBSI recommendation).
        """
        raise NotImplementedError
