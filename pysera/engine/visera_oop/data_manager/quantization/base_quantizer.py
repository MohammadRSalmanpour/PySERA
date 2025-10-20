# -*- coding: utf-8 -*-
# data_manager/quantization/base_quantizer.py

from __future__ import annotations

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """
    Base class for ROI quantizers (IBSI):

      - Input: ROI array with NaNs outside INTENSITY mask
      - Output: quantized volume with NaNs preserved
      - Levels must be ascending, contiguous integers starting at 1
    """

    def __init__(self, roi: np.ndarray) -> None:
        self.roi = self._validate(roi)

    @staticmethod
    def _validate(roi: np.ndarray) -> np.ndarray:
        if roi is None or not isinstance(roi, np.ndarray) or roi.ndim != 3 or roi.size == 0:
            raise ValueError("ROI must be a non-empty 3D numpy.ndarray.")
        return roi.astype(np.float32, copy=False)

    @abstractmethod
    def quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        quantized_volume : float32 ndarray
            Same shape as ROI; NaNs preserved outside intensity mask.
        levels : int32 1D ndarray
            Ascending, contiguous labels [1 ... N].
        """
        raise NotImplementedError
