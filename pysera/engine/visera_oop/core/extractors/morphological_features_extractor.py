# -*- coding: utf-8 -*-
# core/extractors/morphological_features_extractor.py

from __future__ import annotations

import logging
import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import eval_legendre
from skimage.measure import marching_cubes
from typing import Any, Dict, Optional, Tuple

from pysera.engine.visera_oop.core.sparsity.view_planner import DataView
from pysera.processing.synthesize_RoIs import synthesize_coords, synthesize_values
from pysera.engine.visera_oop.core.base_feature_extractor import BaseFeatureExtractor
from pysera.engine.visera_oop.core.base_feature_extractor import safe_sqrt, safe_divide

logger = logging.getLogger("Dev_logger")


class MorphologicalFeaturesExtractor(BaseFeatureExtractor):
    """
    Morphological features (IBSI-aligned) describing geometric aspects of the ROI.

    Implements:
      - Volume (mesh), Surface area (mesh), Surface/Volume
      - Compactness1, Compactness2, Spherical Disproportion, Sphericity, Asphericity
      - PCA axis lengths (major/minor/least), Elongation, Flatness
      - Volume/Area density (AABB), Volume/Area density (OMBB)
      - Volume/Area density (AEE, and MVEE as AEE placeholder)
      - Volume/Area density (Convex Hull) → Solidity, Convex-hull Area Density
      - Centre of Mass shift (intensity-weighted)
      - Maximum 3D diameter (via convex hull)
      - Integrated intensity (mean * V_mesh)
      - Moran's I and Geary's C (spatial autocorrelation)
    """

    ACCEPTS: DataView = DataView.DENSE_BLOCK
    PREFERS: DataView = DataView.DENSE_BLOCK
    NAME: str = "MorphologyExtractor"

    # All public features depend on the relevant hidden computations.
    feature_dependencies: Dict[str, list] = {
        # mesh
        "morph_volume_mesh": ["__mesh"],
        "morph_surface_area": ["__mesh"],
        "morph_volume_count": ["__mesh"],
        "morph_sv_ratio": ["__mesh"],
        # families relying on mesh V/A
        "morph_compactness1": ["__mesh"],
        "morph_compactness2": ["__mesh"],
        "morph_spherical_disproportion": ["__mesh"],
        "morph_sphericity": ["__mesh"],
        "morph_asphericity": ["__mesh"],
        # PCA bundle
        "morph_major_axis_length": ["__pca"],
        "morph_minor_axis_length": ["__pca"],
        "morph_least_axis_length": ["__pca"],
        "morph_elongation": ["__pca"],
        "morph_flatness": ["__pca"],
        # OMBB / AABB need mesh verts (from __mesh) and their own calc
        "morph_volume_density_ombb": ["__mesh", "__ombb"],
        "morph_area_density_ombb": ["__mesh", "__ombb"],
        "morph_volume_density_aabb": ["__mesh", "__aabb"],
        "morph_area_density_aabb": ["__mesh", "__aabb"],
        # AEE / MVEE via PCA + mesh
        "morph_volume_density_aee": ["__mesh", "__pca"],
        "morph_area_density_aee": ["__mesh", "__pca"],
        "morph_volume_density_mvee": ["morph_volume_density_aee"],
        "morph_area_density_mvee": ["morph_area_density_aee"],
        # Convex hull
        "morph_solidity": ["__mesh", "__convex"],
        "morph_area_density_convex_hull": ["__mesh", "__convex"],
        # Others
        "morph_max_3d_diameter": ["__mesh"],
        "morph_integrated_intensity": ["__mesh"],
        "morph_moran_i": ["__autocorr"],
        "morph_geary_c": ["__autocorr"],
        "morph_com_shift": [],  # computes directly
    }

    # ----------------------------
    # Utilities (no persistent caches)
    # ----------------------------
    def _fallback_value(self) -> float:
        return float("nan") if self.feature_value_mode == "REAL_VALUE" else 0.0

    @staticmethod
    def _extract_roi_pv_threshold(views: Dict[str, Any], default: float = 0.5) -> float:
        """
        Extract the ROI partial volume (PV) threshold from a views dictionary.

        The function searches in the following order:
        1) Top-level keys: "roi_pv", "ROI_PV", "partial_volume_threshold", "radiomics_ROI_PV"
        2) Nested dictionaries under "params", "radiomics", or "config"

        Args:
            views (Dict[str, Any]): Dictionary containing view data and configuration.
            default (float, optional): Default value to return if no threshold is found. Defaults to 0.5.

        Returns:
            float: The extracted ROI PV threshold as a float.
        """
        threshold_keys = ["roi_pv", "ROI_PV", "partial_volume_threshold", "radiomics_ROI_PV"]

        # ---- Check top-level keys ----
        for key in threshold_keys:
            if key in views and views[key] is not None:
                try:
                    return float(views[key])
                except (ValueError, TypeError):
                    continue

        # ---- Check nested dictionaries ----
        for parent_key in ("params", "radiomics", "config"):
            nested_dict = views.get(parent_key, {})
            if isinstance(nested_dict, dict):
                for key in threshold_keys:
                    if key in nested_dict and nested_dict[key] is not None:
                        try:
                            return float(nested_dict[key])
                        except (ValueError, TypeError):
                            continue

        # ---- Fallback to default ----
        return float(default)

    def _get_morph_mask_image_spacing(
        self, roi_index: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
        """
        Retrieve a morphological mask, associated image, and voxel spacing in mm for a given ROI.

        Behavior:
            - Morphological mask: uses views["binary_mask"] if present. Applies PV threshold
              for partial-volume masks. Falls back to finite values of image if mask unavailable.
            - Image data: views["dense_block"] if available.
            - Spacing: converts voxel_spacing_cm to mm. Defaults to (1.0, 1.0, 1.0) if unavailable.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
                - mask_bool: boolean mask array.
                - image_data: original image array or None.
                - spacing_mm: tuple of voxel spacing in mm (dx, dy, dz).
        """
        views = self.get_views(roi_index)
        morph_mask = views.get("binary_mask")
        image_data = views.get("dense_block")
        roi_pv_threshold = self._extract_roi_pv_threshold(views, default=0.5)

        # ---- Compute boolean mask ----
        if morph_mask is not None:
            mask_array = np.asarray(morph_mask)
            if mask_array.dtype == bool:
                mask_bool = mask_array
            else:
                if np.any((mask_array > 0) & (mask_array < 1)):
                    if roi_pv_threshold <= 0.0:
                        mask_bool = mask_array > 0.0
                    elif roi_pv_threshold >= 1.0:
                        mask_bool = mask_array >= 1.0
                    else:
                        mask_bool = mask_array >= roi_pv_threshold
                else:
                    mask_bool = mask_array.astype(bool, copy=False)
        else:
            if image_data is None:
                return np.empty((0,), dtype=bool), None, (1.0, 1.0, 1.0)
            mask_bool = np.isfinite(image_data)

        # ---- Compute voxel spacing in mm ----
        voxel_spacing_cm = views.get("voxel_spacing_cm")
        if voxel_spacing_cm is not None and len(voxel_spacing_cm) == 3:
            dz_cm, dy_cm, dx_cm = map(float, voxel_spacing_cm)
            spacing_mm = (dx_cm * 10.0, dy_cm * 10.0, dz_cm * 10.0)
        else:
            logger.warning(f"[ROI {roi_index}] spacing unavailable in views; using (1.0, 1.0, 1.0) mm.")
            spacing_mm = (1.0, 1.0, 1.0)

        return mask_bool, image_data, spacing_mm

    # ----------------------------
    # Hidden, Base-cached building blocks
    #   (names start with "__", excluded from output)
    # ----------------------------

    def _discover_feature_names(self) -> list[str]:
        # Use Base discovery but hide any feature whose name starts with "_" (e.g., "__mesh").
        names = super()._discover_feature_names()
        return [n for n in names if not n.startswith("_")]

    # --- __mesh: verts/faces + surface area + mesh volume + voxel-count volume ---
    def get___mesh(self, roi_index: int) -> Dict[str, Any]:
        """
        Compute a triangular mesh for a morphological mask of the ROI.

        Steps:
        1. Retrieve mask and voxel spacing via _get_morph_mask_image_spacing.
        2. Pad mask to avoid boundary issues.
        3. Use marching_cubes to generate vertices and faces.
        4. Compute surface area (surface_area) and volume (mesh_volume) from mesh.
        5. Compute voxel-based volume (voxel_volume).

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "Volume"      : Mesh-based volume.
                - "Area"      : Mesh surface area.
                - "Vcount" : Voxel-based volume.
                - "verts"  : Array of vertices.
                - "faces"  : Array of faces.
        """
        mask, _image, (spacing_x, spacing_y, spacing_z) = self._get_morph_mask_image_spacing(roi_index)

        if mask.size == 0 or not np.any(mask):
            return {
                "Volume": self._fallback_value(),
                "Area": self._fallback_value(),
                "Vcount": self._fallback_value(),
                "verts": None,
                "faces": None
            }

        try:
            # ---- Pad mask to avoid boundary issues ----
            mask_padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)

            # ---- Compute marching cubes mesh ----
            vertices, faces, _, _ = marching_cubes(mask_padded, level=0.5, spacing=(spacing_x, spacing_y, spacing_z))

            if faces.size == 0:
                raise RuntimeError("marching_cubes produced 0 faces")

            # ---- Compute surface area ----
            verts0, verts1, verts2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
            cross_product = np.cross(verts1 - verts0, verts2 - verts0)
            surface_area = 0.5 * np.sum(np.linalg.norm(cross_product, axis=1))

            # ---- Compute mesh volume (signed) ----
            mean_vertex = np.mean(vertices, axis=0)
            centered_vertices = vertices - mean_vertex
            a_vertices, b_vertices, c_vertices = centered_vertices[faces[:, 0]], centered_vertices[faces[:, 1]], centered_vertices[faces[:, 2]]
            signed_volumes = np.einsum("ij,ij->i", a_vertices, np.cross(b_vertices, c_vertices)) / 6.0
            mesh_volume = float(abs(np.sum(signed_volumes)))

            # ---- Compute voxel-based volume ----
            voxel_volume = float(np.count_nonzero(mask) * spacing_x * spacing_y * spacing_z)

            return {
                "Volume": mesh_volume,
                "Area": surface_area,
                "Vcount": voxel_volume,
                "verts": vertices,
                "faces": faces
            }

        except Exception as e:
            logger.exception(f"[ROI {roi_index}] mesh computation failed: {e}")
            return {
                "Volume": self._fallback_value(),
                "Area": self._fallback_value(),
                "Vcount": self._fallback_value(),
                "verts": None,
                "faces": None
            }

    # --- __pca: eigenvalues + derived axis lengths & ratios ---
    def get___pca(self, roi_index: int) -> Dict[str, Any]:
        """
        Compute PCA-based shape descriptors for a morphological mask of the ROI.

        Args:
            roi_index (int): Index of the region of interest (ROI).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "eigenvalues"   : Eigenvalues of the covariance matrix (sorted descending)
                - "L_major"       : 4 * sqrt(largest eigenvalue)
                - "L_minor"       : 4 * sqrt(second eigenvalue)
                - "L_least"       : 4 * sqrt(smallest eigenvalue)
                - "elongation"    : sqrt(lam[1]/lam[0]) or nan if lam[0] ~ 0
                - "flatness"      : sqrt(lam[2]/lam[0]) or nan if lam[0] ~ 0
        """
        mask_bool, _image_data, (spacing_x_mm, spacing_y_mm, spacing_z_mm) = self._get_morph_mask_image_spacing(roi_index)

        # ---- Handle empty mask ----
        if mask_bool.size == 0 or not np.any(mask_bool):
            eigenvalues = np.zeros(3, dtype=np.float64)
        else:
            coords = np.column_stack(np.where(mask_bool)).astype(np.float64)
            coords[:, 0] *= spacing_x_mm
            coords[:, 1] *= spacing_y_mm
            coords[:, 2] *= spacing_z_mm

            try:
                if coords.shape[0] == 1:
                    eigenvalues = np.zeros(3, dtype=np.float64)
                else:
                    cov_matrix = np.cov(coords.T, bias=False)
                    # Ensure 3x3 shape
                    if cov_matrix.shape != (3, 3):
                        tmp_cov = np.zeros((3, 3), dtype=np.float64)
                        tmp_cov[:cov_matrix.shape[0], :cov_matrix.shape[1]] = cov_matrix
                        cov_matrix = tmp_cov
                    eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
            except Exception as exc:
                logger.error(f"[ROI {roi_index}] PCA computation failed: {exc}")
                eigenvalues = np.zeros(3, dtype=np.float64)

        # ---- Compute shape descriptors ----
        eigenvalues_major = 4.0 * np.sqrt(max(eigenvalues[0], 0.0))
        eigenvalues_minor = 4.0 * np.sqrt(max(eigenvalues[1], 0.0))
        eigenvalues_least = 4.0 * np.sqrt(max(eigenvalues[2], 0.0))

        if self.feature_value_mode == "APPROXIMATE_VALUE":
            elongation = float(safe_sqrt(safe_divide(eigenvalues[1], eigenvalues[0])))
            flatness = float(safe_sqrt(safe_divide(eigenvalues[2], eigenvalues[0])))

        else:
            elongation = float(np.sqrt(eigenvalues[1] / eigenvalues[0])) if eigenvalues[0] > 1e-12 else np.nan
            flatness = float(np.sqrt(eigenvalues[2] / eigenvalues[0])) if eigenvalues[0] > 1e-12 else np.nan

        return {
            "eigenvalues": eigenvalues,
            "L_major": float(eigenvalues_major),
            "L_minor": float(eigenvalues_minor),
            "L_least": float(eigenvalues_least),
            "elongation": elongation,
            "flatness": flatness
        }

    # --- __aabb: AABB volume/area from mesh points ---
    def get___aabb(self, roi_index: int) -> Dict[str, float]:
        """
        Compute the axis-aligned bounding box (AABB) volume and surface area for a given ROI.

        Args:
            roi_index (int): The index of the ROI.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'volume_aabb': AABB volume (float)
                - 'area_aabb': AABB surface area (float)
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        vertices = mesh_data.get("verts")

        # Return NaN if vertices are missing or empty
        if vertices is None or (isinstance(vertices, np.ndarray) and vertices.size == 0):
            return {"volume_aabb": np.nan, "area_aabb": np.nan}

        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        size_x, size_y, size_z = (max_coords - min_coords).astype(float)

        volume_aabb = size_x * size_y * size_z
        area_aabb = 2.0 * (size_x * size_y + size_y * size_z + size_x * size_z)

        return {"Vaabb": float(volume_aabb), "Aaabb": float(area_aabb)}

    # --- __ombb: oriented min bounding box volume/area (falls back to NaNs if utility missing) ---
    def get___ombb(self, roi_index: int) -> Dict[str, float]:
        """
        Compute an approximate Oriented Minimum Bounding Box (OMBB) volume and surface area
        for a given ROI, without using external pysera modules.

        Args:
            roi_index (int): The index of the ROI.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'Vombb': OMBB volume (float)
                - 'Aombb': OMBB surface area (float)
        """
        # Get mesh vertices
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        vertices = mesh_data.get("verts")

        if vertices is None or (isinstance(vertices, np.ndarray) and vertices.size == 0):
            return {"Vombb": np.nan, "Aombb": np.nan}

        try:
            if self.feature_value_mode == "APPROXIMATE_VALUE":
                n, d = vertices.shape
                if n < d + 1:
                    vertices = synthesize_coords(vertices.copy(), n, d, d + 1)

            # Compute convex hull to simplify the point cloud
            hull = ConvexHull(vertices)
            hull_points = vertices[hull.vertices] if hasattr(hull, "vertices") else vertices

            # Compute axis-aligned bounding box on the hull points
            min_coords = np.min(hull_points, axis=0)
            max_coords = np.max(hull_points, axis=0)
            sizes = max_coords - min_coords

            # Volume and surface area
            volume_ombb = float(np.prod(sizes))
            area_ombb = float(2 * (sizes[0] * sizes[1] + sizes[1] * sizes[2] + sizes[0] * sizes[2]))

            return {"Vombb": volume_ombb, "Aombb": area_ombb}

        except Exception as e:
            # If anything fails, return NaN
            return {"Vombb": np.nan, "Aombb": np.nan}

    # --- __convex: convex-hull based solidity + convex-hull surface area ---
    def get___convex(self, roi_index: int) -> Dict[str, float]:
        """
        Compute convexity-related geometric features for a given ROI, including:
        - Solidity (ratio of mesh volume to convex hull volume)
        - Convex hull surface area
        - Convex hull area density (ratio of mesh area to convex hull area)

        Args:
            roi_index (int): The index of the ROI.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'solidity': Ratio of mesh volume to convex hull volume
                - 'area_convex': Surface area of the convex hull
                - 'convex_area_density': Ratio of mesh area to convex hull area
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume_mesh = mesh_data.get("Volume")
        area_mesh = mesh_data.get("Area")
        vertices = mesh_data.get("verts")

        # Validate mesh data
        if vertices is None or not np.isfinite(volume_mesh):
            return {
                "solidity": np.nan,
                "A_convex": np.nan,
                "ch_area_density": np.nan,
            }

        try:
            hull = ConvexHull(vertices)
            volume_convex = float(hull.volume)

            # Compute convex hull surface area manually
            area_convex = 0.0
            for simplex in hull.simplices:
                p0, p1, p2 = vertices[simplex]
                area_convex += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

            # Derived features
            solidity = float(volume_mesh / volume_convex) if volume_convex > 0 else np.nan
            convex_area_density = float(area_mesh / area_convex) if area_convex > 0 else np.nan

            return {
                "solidity": solidity,
                "A_convex": float(area_convex),
                "ch_area_density": convex_area_density,
            }

        except Exception as exc:
            logger.error(f"[ROI {roi_index}] Convex hull computation failed: {exc}")
            return {
                "solidity": np.nan,
                "A_convex": np.nan,
                "ch_area_density": np.nan,
            }

    # --- __autocorr: Moran's I & Geary's C ---
    def get___autocorr(self, roi_index: int) -> Dict[str, float]:
        """
        Compute spatial autocorrelation metrics (Moran's I and Geary's C)
        for voxel intensities within the given ROI.

        Args:
            roi_index (int): The index of the ROI.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'moran_i': Moran's I spatial autocorrelation index
                - 'geary_c': Geary's C spatial autocorrelation index
        """
        mask, image, (spacing_x, spacing_y, spacing_z) = self._get_morph_mask_image_spacing(roi_index)

        # Validate input image
        if image is None or not np.any(np.isfinite(image)):
            return {"moran_i": np.nan, "geary_c": np.nan}

        # Extract valid voxel coordinates and intensities
        finite_mask = np.isfinite(image)
        coords = np.column_stack(np.where(finite_mask)).astype(np.float64)
        intensities = image[finite_mask].astype(np.float64)
        num_voxels = intensities.size

        if num_voxels < 2 and self.feature_value_mode == 'REAL_VALUE':
            return {"moran_i": np.nan, "geary_c": np.nan}

        # Scale coordinates by voxel spacing
        positions = np.column_stack((
            coords[:, 2] * spacing_x,
            coords[:, 1] * spacing_y,
            coords[:, 0] * spacing_z
        ))

        def _pairwise_stats(values: np.ndarray, points: np.ndarray) -> Tuple[float, float]:
            """Compute Moran's I and Geary's C between all voxel pairs."""
            n_points = values.size
            if n_points < 2:
                if self.feature_value_mode == 'APPROXIMATE_VALUE':
                    values = synthesize_values(values.copy(), target_num=2)
                    points = synthesize_coords(points.copy(), n_points, points.shape[-1], target_num=2)
                    n_points = values.size
                else:
                    return float("nan"), float("nan")

            mean_val = float(np.mean(values))
            deviations = values - mean_val
            denom = float(np.sum(deviations ** 2.0))
            if denom == 0.0:
                return 1.0, 1.0

            # Compute pairwise distances and weights
            indices = np.arange(n_points)
            idx_i = np.repeat(indices[:-1], np.arange(n_points - 1, 0, -1))
            idx_j = np.concatenate([np.arange(i + 1, n_points) for i in range(n_points - 1)])
            diffs = points[idx_i] - points[idx_j]
            distances = np.linalg.norm(diffs, axis=1)

            weights = 1.0 / np.maximum(distances, np.finfo(np.float64).eps)
            weight_sum = 2.0 * np.sum(weights)

            moran_num = 2.0 * np.sum(weights * deviations[idx_i] * deviations[idx_j])
            moran_i_value = (n_points / weight_sum) * (moran_num / denom)

            geary_num = 2.0 * np.sum(weights * (deviations[idx_i] - deviations[idx_j]) ** 2.0)
            geary_c_value = ((n_points - 1.0) / (2.0 * weight_sum)) * (geary_num / denom)

            return float(moran_i_value), float(geary_c_value)

        # Direct computation for small datasets
        if num_voxels < 1000:
            moran_i, geary_c = _pairwise_stats(intensities, positions)

        # Sampling-based estimation for large datasets
        else:
            moran_samples, geary_samples = [], []
            tolerance_target = 0.001
            max_iterations = 2000
            sample_size = int(min(num_voxels, max(200, min(500, num_voxels // 4))))

            rng = np.random.default_rng(seed=14641 + int(roi_index))
            tolerance_sem = np.inf
            iteration = 1

            while iteration <= max_iterations and tolerance_sem > tolerance_target:
                selection = rng.choice(num_voxels, size=sample_size, replace=False)
                moran_val, geary_val = _pairwise_stats(intensities[selection], positions[selection])

                if np.isfinite(moran_val) and np.isfinite(geary_val):
                    moran_samples.append(moran_val)
                    geary_samples.append(geary_val)

                    # Convergence check after at least 10 samples
                    if len(moran_samples) > 10:
                        moran_sem = np.std(moran_samples, ddof=1) / np.sqrt(len(moran_samples))
                        geary_sem = np.std(geary_samples, ddof=1) / np.sqrt(len(geary_samples))
                        tolerance_sem = max(moran_sem, geary_sem)

                iteration += 1

            moran_i = float(np.mean(moran_samples)) if moran_samples else 1.0
            geary_c = float(np.mean(geary_samples)) if geary_samples else 1.0

        return {"moran_i": moran_i, "geary_c": geary_c}

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _pca_axes_from_eigvals(lam: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute the principal component axes (a, b, c) from eigenvalues.

        Each axis length is defined as 2 * sqrt(lambda_i), ensuring non-negative values.

        Args:
            eigvals (np.ndarray): Array of three eigenvalues (λ₁, λ₂, λ₃).

        Returns:
            Tuple[float, float, float]: The principal axes lengths (a, b, c).
        """
        axis_a = 2.0 * np.sqrt(max(float(lam[0]), 0.0))
        axis_b = 2.0 * np.sqrt(max(float(lam[1]), 0.0))
        axis_c = 2.0 * np.sqrt(max(float(lam[2]), 0.0))

        return float(axis_a), float(axis_b), float(axis_c)

    def _ellipsoid_area_series(self, axis_a: float, axis_b: float, axis_c: float, max_order: int = 20) -> float:
        """
        Approximate the surface area of a tri-axial ellipsoid using
        a Legendre series expansion.

        Args:
            axis_a (float): Semi-major axis length (a).
            axis_b (float): Intermediate axis length (b).
            axis_c (float): Semi-minor axis length (c).
            max_order (int, optional): Maximum order of the Legendre series (default = 20).

        Returns:
            float: Approximated ellipsoid surface area, or NaN if invalid input.
        """

        if self.feature_value_mode == "APPROXIMATE_VALUE":
            alpha = safe_sqrt(max(0.0, 1.0 - safe_divide(axis_b, axis_a) ** 2))
            beta = safe_sqrt(max(0.0, 1.0 - safe_divide(axis_c, axis_a) ** 2))

        elif axis_a <= 0 or axis_b <= 0 or axis_c <= 0:
            return float("nan")

        else:
            # Compute ellipticity parameters
            alpha = np.sqrt(max(0.0, 1.0 - (axis_b / axis_a) ** 2))
            beta = np.sqrt(max(0.0, 1.0 - (axis_c / axis_a) ** 2))

        # Handle spherical case
        if alpha == 0.0 and beta == 0.0:
            return float(4.0 * np.pi * axis_a * axis_a)

        # Compute normalized parameter for Legendre series
        denom = 2.0 * alpha * beta

        if self.feature_value_mode == "APPROXIMATE_VALUE":
            m_param = safe_divide((alpha ** 2 + beta ** 2), denom) if denom != 0.0 else 1.0
        else:
            m_param = (alpha ** 2 + beta ** 2) / denom if denom != 0.0 else 1.0

        m_param = float(np.clip(m_param, 0.0, 1.0))

        # Summation term
        series_sum = 0.0
        for k in range(max_order + 1):
            denom_k = 1.0 - 4.0 * (k ** 2)
            if abs(denom_k) < 1e-12:
                continue
            legendre_val = float(eval_legendre(k, m_param))
            series_sum += ((alpha * beta) ** k) * legendre_val / denom_k

        # Final approximation
        surface_area = 4.0 * np.pi * axis_a * axis_b * series_sum
        return float(surface_area)

    # =============================
    #      Public Feature API
    # =============================

    # Basic mesh geometry
    def get_morph_volume_mesh(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__mesh", roi_index)["Volume"])

    def get_morph_volume_count(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__mesh", roi_index)["Vcount"])

    def get_morph_surface_area(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__mesh", roi_index)["Area"])

    def get_morph_sv_ratio(self, roi_index: int) -> float:
        """
        Compute the surface-to-volume ratio (S/V) for a given ROI.

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Surface-to-volume ratio (A/V).
                   Returns NaN if the volume is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if not np.isfinite(volume) or volume == 0.0:
            return np.nan

        return float(surface_area / volume)

    # Compactness / Sphericity family
    def get_morph_compactness1(self, roi_index: int) -> float:
        """
        Compute the primary compactness measure (Compactness 1) for a given ROI.

        Formula:
            C1 = V / (sqrt(pi) * A^(3/2))

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Compactness 1 value.
                   Returns NaN if the surface area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if surface_area is None or surface_area <= 0:
            return np.nan

        return float(volume / (np.sqrt(np.pi) * (surface_area ** 1.5)))

    def get_morph_compactness2(self, roi_index: int) -> float:
        """
        Compute the secondary compactness measure (Compactness 2) for a given ROI.

        Formula:
            C2 = (36 * π * V²) / (A³)

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Compactness 2 value.
                   Returns NaN if the surface area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if surface_area is None or surface_area <= 0:
            return np.nan

        return float(36.0 * np.pi * (volume ** 2) / (surface_area ** 3))

    def get_morph_spherical_disproportion(self, roi_index: int) -> float:
        """
         Compute the Spherical Disproportion for a given ROI.

         Formula:
             SD = A / ((36 * π * V²)^(1/3))

         Interpretation:
             - SD = 1 → perfect sphere
             - SD > 1 → deviation from spherical shape increases

         Args:
             roi_index (int): Index of the Region of Interest (ROI).

         Returns:
             float: Spherical Disproportion value.
                    Returns NaN if the volume is invalid or zero.
         """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if volume is None or volume <= 0:
            return np.nan

        denominator = (36.0 * np.pi * (volume ** 2)) ** (1.0 / 3.0)
        return float(surface_area / denominator)

    def get_morph_sphericity(self, roi_index: int) -> float:
        """
        Compute the Sphericity of a 3D object for a given ROI.

        Formula:
            ψ = ((36 * π * V²)^(1/3)) / A

        Interpretation:
            - ψ = 1 → perfect sphere
            - ψ < 1 → object deviates from spherical shape

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Sphericity value (dimensionless).
                   Returns NaN if the surface area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if surface_area is None or surface_area <= 0:
            return np.nan

        numerator = (36.0 * np.pi * (volume ** 2)) ** (1.0 / 3.0)
        return float(numerator / surface_area)

    def get_morph_asphericity(self, roi_index: int) -> float:
        """
        Compute the Asphericity of a 3D object for a given ROI.

        Formula:
            Asphericity = ((A³) / (36 * π * V²))^(1/3) - 1

        Interpretation:
            - Asphericity = 0 → perfect sphere
            - Asphericity > 0 → deviation from spherical shape

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Asphericity value (dimensionless).
                   Returns NaN if the volume is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        volume = mesh_data.get("Volume")
        surface_area = mesh_data.get("Area")

        if volume is None or volume <= 0:
            return np.nan

        asphericity_value = ((surface_area ** 3) / (36.0 * np.pi * (volume ** 2))) ** (1.0 / 3.0) - 1.0
        return float(asphericity_value)

    # Centre of Mass shift
    def get_morph_com_shift(self, roi_index: int) -> float:
        """
        Compute the center-of-mass (COM) shift for a given ROI.

        This function retrieves the morphological mask and corresponding image,
        then evaluates if the COM can be computed.

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: COM shift value.
                   Returns NaN if the image or mask is invalid or empty.
        """
        mask, image_data, (spacing_x, spacing_y, spacing_z) = self._get_morph_mask_image_spacing(roi_index)

        # Validate inputs
        if image_data is None or not np.any(np.isfinite(image_data)) or not np.any(mask):
            return float("nan")

        coords = np.column_stack(np.where(mask)).astype(np.float64)
        geom_center = np.mean(
            np.column_stack((coords[:, 0] * spacing_x, coords[:, 1] * spacing_y, coords[:, 2] * spacing_z)),
            axis=0,
        )

        inten_coords = np.column_stack(np.where(np.isfinite(image_data))).astype(np.float64)
        intensities = image_data[np.isfinite(image_data)].astype(np.float64)
        if intensities.size == 0:
            return 0.0

        inten_center = np.sum(
            np.column_stack((inten_coords[:, 0] * spacing_x, inten_coords[:, 1] * spacing_y, inten_coords[:, 2] * spacing_z))
            * intensities[:, None],
            axis=0,
        ) / float(np.nansum(intensities))

        return float(np.linalg.norm(geom_center - inten_center))

    # Max 3D diameter
    def get_morph_max_3d_diameter(self, roi_index: int) -> float:
        """
        Compute the maximum 3D diameter of a ROI mesh.

        The maximum diameter is defined as the largest Euclidean distance
        between any two vertices of the convex hull of the mesh.

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Maximum 3D diameter.
                   Returns NaN if mesh vertices are invalid, or 0.0 if not enough points.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        vertices = mesh_data.get("verts")

        # Validate vertices
        if vertices is None or (isinstance(vertices, np.ndarray) and vertices.shape[0] < 2):
            return float("nan")

        # Compute convex hull vertices if possible
        try:
            hull = ConvexHull(vertices)
            points = vertices[hull.vertices]
        except Exception:
            points = vertices

        # Check if enough points remain
        if points.shape[0] < 2:
            return 0.0

        # Compute all pairwise distances
        diffs = points[:, None, :] - points[None, :, :]
        distances_squared = np.sum(diffs ** 2, axis=2)
        max_distance = np.sqrt(np.max(distances_squared))

        return float(max_distance)

    # PCA axes & derived
    def get_morph_major_axis_length(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__pca", roi_index)["L_major"])

    def get_morph_minor_axis_length(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__pca", roi_index)["L_minor"])

    def get_morph_least_axis_length(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__pca", roi_index)["L_least"])

    def get_morph_elongation(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__pca", roi_index)["elongation"])

    def get_morph_flatness(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__pca", roi_index)["flatness"])

    # Density (AABB)
    def get_morph_volume_density_aabb(self, roi_index: int) -> float:
        """
        Compute the volume density relative to the Axis-Aligned Bounding Box (AABB) for a given ROI.

        Formula:
            Volume Density = V / V_AABB

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Volume density (dimensionless).
                   Returns NaN if AABB volume is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        aabb_data = self._get_or_compute_feature("__aabb", roi_index)

        volume_mesh = mesh_data.get("Volume")
        volume_aabb = aabb_data.get("Vaabb")

        if volume_aabb is None or volume_aabb <= 0:
            return np.nan

        return float(volume_mesh / volume_aabb)

    def get_morph_area_density_aabb(self, roi_index: int) -> float:
        """
        Compute the surface area density relative to the Axis-Aligned Bounding Box (AABB) for a given ROI.

        Formula:
            Area Density = A / A_AABB

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Surface area density (dimensionless).
                   Returns NaN if AABB surface area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        aabb_data = self._get_or_compute_feature("__aabb", roi_index)

        surface_area_mesh = mesh_data.get("Area")
        surface_area_aabb = aabb_data.get("Aaabb")

        if surface_area_aabb is None or surface_area_aabb <= 0:
            return np.nan

        return float(surface_area_mesh / surface_area_aabb)

    # Density (OMBB)
    def get_morph_volume_density_ombb(self, roi_index: int) -> float:
        """
        Compute the volume density relative to the Oriented Minimum Bounding Box (OMBB) for a given ROI.

        Formula:
            Volume Density = V / V_OMBB

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Volume density (dimensionless).
                   Returns NaN if OMBB volume is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        ombb_data = self._get_or_compute_feature("__ombb", roi_index)

        volume_mesh = mesh_data.get("Volume")
        volume_ombb = ombb_data.get("Vombb")

        if volume_ombb is None or not np.isfinite(volume_ombb) or volume_ombb <= 0:
            return np.nan

        return float(volume_mesh / volume_ombb)

    def get_morph_area_density_ombb(self, roi_index: int) -> float:
        """
        Compute the surface area density relative to the Oriented Minimum Bounding Box (OMBB) for a given ROI.

        Formula:
            Area Density = A / A_OMBB

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Surface area density (dimensionless).
                   Returns NaN if OMBB surface area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        ombb_data = self._get_or_compute_feature("__ombb", roi_index)

        surface_area_mesh = mesh_data.get("Area")
        surface_area_ombb = ombb_data.get("Aombb")

        if surface_area_ombb is None or not np.isfinite(surface_area_ombb) or surface_area_ombb <= 0:
            return np.nan

        return float(surface_area_mesh / surface_area_ombb)

    # Density (AEE / MVEE placeholders)
    def get_morph_volume_density_aee(self, roi_index: int) -> float:
        """
        Compute the volume density relative to the Ellipsoid of Equivalent Ellipsoid (AEE) for a given ROI.

        Formula:
            V_density_AEE = V / V_AEE
            where V_AEE = (4/3) * π * a * b * c, and a, b, c are PCA-derived axes.

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Volume density relative to AEE (dimensionless).
                   Returns NaN if the equivalent ellipsoid volume is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        pca_data = self._get_or_compute_feature("__pca", roi_index)

        axis_a, axis_b, axis_c = self._pca_axes_from_eigvals(pca_data["eigenvalues"])
        volume_aee = (4.0 / 3.0) * np.pi * axis_a * axis_b * axis_c

        if volume_aee <= 0:
            if self.feature_value_mode == 'APPROXIMATE_VALUE':
                return float(safe_divide(mesh_data["Volume"], volume_aee))
            else:   # 'REAL_VALUE'
                return np.nan

        return float(mesh_data["Volume"] / volume_aee)

    def get_morph_area_density_aee(self, roi_index: int) -> float:
        """
        Compute the surface area density relative to the Ellipsoid of Equivalent Ellipsoid (AEE) for a given ROI.

        Formula:
            Area Density = A / A_AEE
            where A_AEE is approximated using a Legendre series expansion based on PCA axes.

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Surface area density relative to AEE (dimensionless).
                   Returns NaN if the equivalent ellipsoid area is invalid or zero.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        pca_data = self._get_or_compute_feature("__pca", roi_index)

        axis_a, axis_b, axis_c = self._pca_axes_from_eigvals(pca_data["eigenvalues"])
        area_aee = self._ellipsoid_area_series(axis_a, axis_b, axis_c, max_order=20)

        if self.feature_value_mode == "APPROXIMATE_VALUE":
            return float(safe_divide(mesh_data["Area"], area_aee))

        if area_aee is None or not np.isfinite(area_aee):
            return np.nan

        if area_aee <= 0:
            if self.feature_value_mode == "APPROXIMATE_VALUE":
                return float(safe_divide(mesh_data["Area"], area_aee))
            else:
                return np.nan

        return float(mesh_data["Area"] / area_aee)

    def get_morph_volume_density_mvee(self, roi_index: int) -> float:
        return self.get_morph_volume_density_aee(roi_index)

    def get_morph_area_density_mvee(self, roi_index: int) -> float:
        return self.get_morph_area_density_aee(roi_index)

    # Convex hull densities
    def get_morph_solidity(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__convex", roi_index)["solidity"])

    def get_morph_area_density_convex_hull(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__convex", roi_index)["ch_area_density"])

    # Integrated intensity
    def get_morph_integrated_intensity(self, roi_index: int) -> float:
        """
        Compute the integrated intensity for a given ROI.

        The integrated intensity is defined as the mean intensity of the ROI
        multiplied by its volume.

        Formula:
            Integrated Intensity = mean(I) * V

        Args:
            roi_index (int): Index of the Region of Interest (ROI).

        Returns:
            float: Integrated intensity value.
                   Returns NaN if image data is invalid or contains no finite values.
        """
        mesh_data = self._get_or_compute_feature("__mesh", roi_index)
        mask, image_data, _spacing = self._get_morph_mask_image_spacing(roi_index)

        # Validate image data
        if image_data is None or not np.any(np.isfinite(image_data)):
            return np.nan

        mean_intensity = float(np.nanmean(image_data))
        volume = mesh_data.get("Volume")

        return float(mean_intensity * volume)

    # Moran / Geary
    def get_morph_moran_i(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__autocorr", roi_index)["moran_i"])

    def get_morph_geary_c(self, roi_index: int) -> float:
        return float(self._get_or_compute_feature("__autocorr", roi_index)["geary_c"])