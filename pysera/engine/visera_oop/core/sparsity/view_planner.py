# -*- coding: utf-8 -*-
# core/sparsity/view_planner.py

from __future__ import annotations

import logging
from enum import IntFlag, auto
from dataclasses import dataclass
from typing import Iterable, Tuple, Type, List

from ..base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class DataView(IntFlag):
    """Bitmask of materialized views that an extractor can consume."""
    NONE = 0
    ROI_VECTOR = auto()    # 1D packed finite ROI values
    DENSE_BLOCK = auto()   # Cropped dense 2D/3D block
    SPARSE_COORDS = auto() # COO-like (coords, values) within ROI
    RUN_LENGTHS = auto()   # Compact run encoding (for RL/zone families)


@dataclass
class ViewPlan:
    """Planner output describing which views to build for a set of extractors."""
    required: DataView = DataView.NONE   # Must build
    preferred: DataView = DataView.NONE  # Nice to have (if cheap)


def _as_dataview(mask_value) -> DataView:
    """Coerce arbitrary values (int or enum) into a DataView bitmask."""
    if isinstance(mask_value, DataView):
        return mask_value
    try:
        return DataView(int(mask_value))
    except (ValueError, TypeError):
        return DataView.NONE


def _get_accepts_prefers(extractor_cls: Type[BaseFeatureExtractor]) -> Tuple[DataView, DataView]:
    """
    Read capabilities from an extractor class.

    Defaults:
      ACCEPTS  -> DENSE_BLOCK
      PREFERS  -> ACCEPTS (if provided) else DENSE_BLOCK
    """
    accepts = _as_dataview(getattr(extractor_cls, "ACCEPTS", DataView.DENSE_BLOCK))
    prefers = _as_dataview(getattr(extractor_cls, "PREFERS", accepts or DataView.DENSE_BLOCK))

    # Never return NONE for accepts; dense is universal fallback
    if accepts == DataView.NONE:
        accepts = DataView.DENSE_BLOCK

    return accepts, prefers


class ViewPlanner:
    """
    Determine which views to materialize for a set of extractors.

    This class aggregates extractor requirements and decides which views are
    required and preferred, optionally enabling sparse pathways when allowed.
    """

    def __init__(self, **kwargs: object) -> None:
        # Keep kwargs for forward-compatibility (e.g., thresholds/toggles)
        self._config = kwargs

    @staticmethod
    def build_plan(
        extractor_classes: Iterable[Type[BaseFeatureExtractor]],
        use_sparse: bool
    ) -> ViewPlan:
        """
        Aggregate extractor requirements and produce a plan for materialized views.

        Parameters
        ----------
        extractor_classes : Iterable[Type[BaseFeatureExtractor]]
            Classes of extractors to consider.
        use_sparse : bool
            Whether to allow sparse representations.

        Returns
        -------
        ViewPlan
            Object indicating required and preferred views.
        """
        classes_list: List[Type[BaseFeatureExtractor]] = list(extractor_classes)
        if not classes_list:
            return ViewPlan(required=DataView.NONE, preferred=DataView.NONE)

        required_views = DataView.NONE
        preferred_views = DataView.NONE
        needs_dense = False

        # Aggregate capabilities and check for dense requirement in one loop
        for extractor_cls in classes_list:
            accepts_mask, prefers_mask = _get_accepts_prefers(extractor_cls)
            required_views |= accepts_mask
            preferred_views |= prefers_mask
            if accepts_mask & DataView.DENSE_BLOCK:
                needs_dense = True

        if not use_sparse:
            # Force dense-only path
            required_views = (required_views | DataView.DENSE_BLOCK) & ~(
                DataView.ROI_VECTOR | DataView.SPARSE_COORDS | DataView.RUN_LENGTHS
            )
            preferred_views = DataView.DENSE_BLOCK
        else:
            # Sparse allowed: still honor hard dense requirements
            if needs_dense:
                required_views |= DataView.DENSE_BLOCK
            preferred_views = preferred_views or DataView.DENSE_BLOCK

        # Fallback if nothing is required/preferred
        if required_views == DataView.NONE and preferred_views == DataView.NONE:
            return ViewPlan(required=DataView.DENSE_BLOCK, preferred=DataView.DENSE_BLOCK)

        return ViewPlan(required=required_views, preferred=preferred_views)
