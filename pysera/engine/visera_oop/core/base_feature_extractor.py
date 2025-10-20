# -*- coding: utf-8 -*-
# core/base_feature_extractor.py

from __future__ import annotations

import gc
import time
import psutil
import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, Type, Union

logger = logging.getLogger("Dev_logger")


class BaseFeatureExtractor:
    """Base class for feature extractors with registry + per-feature profiling."""

    EXTRACTOR_REGISTRY: Dict[str, Type["BaseFeatureExtractor"]] = {}
    _ENABLED_NAMES: Optional[set[str]] = None
    _ENABLED_ORDER: Optional[list[str]] = None

    feature_dependencies: Dict[str, list] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls is BaseFeatureExtractor:
            return

        BaseFeatureExtractor.EXTRACTOR_REGISTRY[cls.__name__] = cls
        alias = getattr(cls, "NAME", None)

        if isinstance(alias, str) and alias and alias not in BaseFeatureExtractor.EXTRACTOR_REGISTRY:
            BaseFeatureExtractor.EXTRACTOR_REGISTRY[alias] = cls

    @classmethod
    def enable_extractors_from_list(cls, names: Optional[list[str]]) -> None:
        """
        Enable extractors honoring caller-provided order.
        If names is None -> enable all. If [] -> disable all.
        """
        if names is None:
            # enable all (keep registration order)
            cls._ENABLED_NAMES = None
            cls._ENABLED_ORDER = None
            return

        # Explicit list supplied (possibly empty): honor exactly, in order
        enabled_list: list[str] = []
        seen = set()

        for name in names:
            # Exact class name
            if name in cls.EXTRACTOR_REGISTRY and name not in seen:
                enabled_list.append(name); seen.add(name)
                continue

            # Also allow alias/prefix matches (preserve order; avoid dups)
            name_lower = name.lower()
            for reg_name, reg_cls in cls.EXTRACTOR_REGISTRY.items():
                if reg_name in seen:  # already included
                    continue
                reg_lower = reg_name.lower()
                reg_alias = getattr(reg_cls, "NAME", "").lower()
                if (
                    name_lower == reg_lower
                    or name_lower == reg_alias
                    or reg_lower.startswith(name_lower)
                    or reg_alias.startswith(name_lower)
                ):
                    enabled_list.append(reg_name); seen.add(reg_name)

        # Store both a set (fast membership) and the *ordered* list
        cls._ENABLED_ORDER = enabled_list
        cls._ENABLED_NAMES = set(enabled_list)

    @classmethod
    def get_enabled_extractors(cls) -> Dict[str, Type["BaseFeatureExtractor"]]:
        # If no filter -> return all in registration order
        if cls._ENABLED_ORDER is None and cls._ENABLED_NAMES is None:
            return dict(cls.EXTRACTOR_REGISTRY)

        # If list is explicitly empty -> disable all
        if cls._ENABLED_ORDER is not None and len(cls._ENABLED_ORDER) == 0:
            return {}

        # Otherwise, return an *ordered* dict following _ENABLED_ORDER
        ordered: Dict[str, Type["BaseFeatureExtractor"]] = {}
        if cls._ENABLED_ORDER is not None:
            for n in cls._ENABLED_ORDER:
                if n in cls.EXTRACTOR_REGISTRY:
                    ordered[n] = cls.EXTRACTOR_REGISTRY[n]
            return ordered

        # (fallback: set was provided without order; keep registry order)
        return {n: cls.EXTRACTOR_REGISTRY[n] for n in cls.EXTRACTOR_REGISTRY if n in cls._ENABLED_NAMES}

    def __init__(
        self,
        feature_value_mode: str = "REAL_VALUE",
        cache: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:

        self.feature_value_mode = feature_value_mode
        self.cache: Dict[str, Any] = cache if cache is not None else {}

        self._feature_cache: Dict[int, Dict[str, Any]] = {}
        self.feature_names = self._discover_feature_names()

        self.data: Dict[int, Dict[str, Any]] = {}
        self.last_feature_perf: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.last_perf: Dict[int, Dict[str, Any]] = {}
        self._init_subclass(**kwargs)

    def _init_subclass(self, **kwargs: Any) -> None:
        pass

    # -------- data --------

    def _prepare_inputs(
        self,
        *,
        image: Optional[np.ndarray] = None,
        roi: Optional[np.ndarray] = None,
        views: Optional[Dict[str, Any]] = None,
        roi_index: int = 0,
        **_: Any,
    ) -> None:

        self.data = {
            int(roi_index): {
                "image": image,
                "roi": roi,
                "views": views or {},
            }
        }

    def get_image(self, roi_index: int) -> Optional[np.ndarray]:
        return self.data.get(roi_index, {}).get("image", None)

    def get_roi(self, roi_index: int) -> Optional[np.ndarray]:
        return self.data.get(roi_index, {}).get("roi", None)

    def get_views(self, roi_index: int) -> Dict[str, Any]:
        return self.data.get(roi_index, {}).get("views", {}) or {}

    # -------- discovery --------

    def _discover_feature_names(self) -> list[str]:
        feats = []
        for name, func in self.__class__.__dict__.items():

            if callable(func) and name.startswith("get_"):
                feats.append(name.replace("get_", ""))

        return feats

    # -------- profiling --------

    @staticmethod
    def _profile_snapshot() -> Dict[str, Any]:
        rss_kb = psutil.Process().memory_info().rss / 1024.0
        return {"time": time.perf_counter(), "wall": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), "rss_kb": rss_kb}

    @staticmethod
    def _profile_delta(start: Dict[str, Any]) -> Dict[str, Any]:
        end = BaseFeatureExtractor._profile_snapshot()

        return {
            "start_time": start["wall"],
            "end_time": end["wall"],
            "total_time_sec": round(end["time"] - start["time"], 6),
            "start_memory_KB": round(start["rss_kb"], 6),
            "end_memory_KB": round(end["rss_kb"], 6),
            "total_memory_KB": round(end["rss_kb"] - start["rss_kb"], 6),
        }

    def _store_feature_perf(self, roi_index: int, feature_name: str, perf: Dict[str, Any]) -> None:
        self.last_feature_perf.setdefault(int(roi_index), {})[feature_name] = perf

    def _compute_feature(self, feature_name: str, roi_index: int) -> Any:
        for dep in self.feature_dependencies.get(feature_name, []):
            self._get_or_compute_feature(dep, roi_index)

        getter = getattr(self, f"get_{feature_name}", None)
        prof = self._profile_snapshot()

        try:
            value = getter(roi_index) if callable(getter) else np.nan

        except Exception as exc:
            logger.error("Error computing '%s' (ROI %d): %s", feature_name, roi_index, exc)
            value = np.nan

        finally:
            self._store_feature_perf(roi_index, feature_name, self._profile_delta(prof))

        return value

    def _get_or_compute_feature(self, feature_name: str, roi_index: int) -> Any:
        cache = self._feature_cache.setdefault(roi_index, {})

        if feature_name in cache:
            return cache[feature_name]

        val = self._compute_feature(feature_name, roi_index)
        cache[feature_name] = val

        return val

    def _compute_all_features_for_roi(self, roi_index: int, selected_features: Optional[list[str]] = None) -> Dict[str, Any]:
        feats = selected_features if selected_features else self.feature_names
        return {n: self._get_or_compute_feature(n, roi_index) for n in feats}

    # -------- entry --------

    def extract(self, *, selected_features: Optional[list[str]] = None, **kwargs: Any) -> Dict[int, Dict[str, Any]]:
        self._prepare_inputs(**kwargs)
        roi_index = int(kwargs.get("roi_index", 0))

        if not self.data or (self.get_image(roi_index) is None and not self.get_views(roi_index)):
            self.last_perf[roi_index] = self._profile_delta(self._profile_snapshot())
            return {roi_index: {}}

        prof = self._profile_snapshot()
        out: Dict[int, Dict[str, Any]] = {}

        for index in sorted(self.data.keys()):
            out[index] = self._compute_all_features_for_roi(index, selected_features)

        self.last_perf[roi_index] = self._profile_delta(prof)

        gc.collect()
        return out

def safe_divide(a: Union[np.ndarray, float, int],
                b: Union[np.ndarray, float, int]
                ) -> Union[np.ndarray, float]:
    """
    Safely divide a by b, avoiding division by zero.
    Replaces zeros in b with np.finfo(np.float64).eps.
    """
    if isinstance(b, np.ndarray):
        if isinstance(a, np.ndarray):
            if a.shape != b.shape:
                raise ValueError("Shape of 'a' and 'b' must be equal")
        b_safe = np.where(b == 0, np.finfo(np.float64).eps, b)
    else:
        b_safe = np.finfo(np.float64).eps if b == 0 else b

    return a / b_safe

def safe_sqrt(x: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    """
    Safely compute square root, avoiding sqrt of negative numbers.
    Replaces negative values with np.finfo(np.float64).eps.
    """
    # --- Scalar case ---
    if np.isscalar(x):
        x_safe = np.finfo(np.float64).eps if x < 0 else x
        return float(np.sqrt(x_safe))

    # --- Array case ---
    if isinstance(x, np.ndarray):
        x_safe = np.where(x < 0, np.finfo(np.float64).eps, x)
        return np.sqrt(x_safe)

    # --- Unsupported type ---
    raise TypeError(f"Unsupported input type: {type(x)}")