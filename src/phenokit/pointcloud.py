"""3-D phenotyping from LiDAR / laser-scanner / photogrammetric point clouds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree


def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Keep the centroid of points in each occupied voxel."""
    xyz = np.asarray(xyz, float)
    keys = np.floor((xyz - xyz.min(0)) / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.bincount(inv)
    return np.vstack([np.bincount(inv, xyz[:, d]) / counts for d in range(3)]).T


def statistical_outlier_removal(xyz: np.ndarray, k: int = 10, std_ratio: float = 2.5) -> np.ndarray:
    """Boolean inlier mask: mean k-NN distance within mean + std_ratio·σ."""
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    md = d[:, 1:].mean(1)
    return md <= md.mean() + std_ratio * md.std()


def ransac_plane(xyz: np.ndarray, threshold: float = 0.02, iterations: int = 500, seed: int = 0,
                 candidate_mask: np.ndarray | None = None):
    """Fit the dominant plane (soil/pot surface). Returns ``(normal, d, inlier_mask)`` with n·p + d = 0, nz > 0."""
    rng = np.random.default_rng(seed)
    pts = xyz if candidate_mask is None else xyz[candidate_mask]
    best, best_n = None, -1
    for _ in range(iterations):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -n @ s[0]
        cnt = int((np.abs(pts @ n + d) < threshold).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    # least-squares refinement on inliers
    inl = np.abs(pts @ n + d) < threshold
    c = pts[inl].mean(0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    d = -n @ c
    return n, d, np.abs(xyz @ n + d) < threshold


def height_above_plane(xyz, normal, d):
    return np.asarray(xyz, float) @ normal + d


def grid_dtm(xyz: np.ndarray, res: float = 0.5, percentile: float = 2.0, origin=None, shape=None):
    """Digital terrain model: low percentile of z per cell, gaps filled by nearest neighbour."""
    xyz = np.asarray(xyz, float)
    o = xyz[:, :2].min(0) if origin is None else np.asarray(origin, float)
    ij = np.floor((xyz[:, :2] - o) / res).astype(int)
    shp = tuple(ij.max(0) + 1) if shape is None else shape
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shp[0]) & (ij[:, 1] < shp[1])
    ij, z = ij[ok], xyz[ok, 2]
    dtm = np.full(shp, np.nan)
    flat = np.ravel_multi_index(ij.T, shp)
    order = np.argsort(flat)
    flat, z = flat[order], z[order]
    uniq, start = np.unique(flat, return_index=True)
    for u, s, e in zip(uniq, start, np.r_[start[1:], len(flat)]):
        dtm.flat[u] = np.percentile(z[s:e], percentile)
    if np.isnan(dtm).any():
        idx = ndimage.distance_transform_edt(np.isnan(dtm), return_distances=False, return_indices=True)
        dtm = dtm[tuple(idx)]
    # smooth to suppress cells where no ground was visible (dense canopy)
    return ndimage.median_filter(dtm, size=3), o


def normalise_heights(xyz: np.ndarray, res: float = 0.5, percentile: float = 2.0) -> np.ndarray:
    """Height above ground for each point using a gridded DTM (bilinear-free nearest-cell lookup)."""
    dtm, o = grid_dtm(xyz, res, percentile)
    ij = np.clip(np.floor((xyz[:, :2] - o) / res).astype(int), 0, np.array(dtm.shape) - 1)
    return xyz[:, 2] - dtm[ij[:, 0], ij[:, 1]]


def canopy_height_model(xyz: np.ndarray, heights: np.ndarray, transform, shape) -> np.ndarray:
    """Rasterise max height-above-ground into the grid defined by a GeoTransform."""
    r, c = transform.map_to_pixel(xyz[:, 0], xyz[:, 1])
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < shape[0]) & (c < shape[1])
    chm = np.full(shape, -np.inf)
    np.maximum.at(chm, (r[ok], c[ok]), heights[ok])
    chm[np.isinf(chm)] = np.nan
    return chm


def plot_height_metrics(xyz: np.ndarray, heights: np.ndarray, layout, buffer_m: float = 0.1,
                        canopy_threshold: float = 0.05) -> pd.DataFrame:
    """Per-plot LiDAR metrics: height percentiles, canopy cover, density and profile volume."""
    idx = layout.assign_points(xyz[:, :2], buffer_m)
    rows = []
    for k, p in enumerate(layout.table.itertuples()):
        h = heights[idx == k]
        area = (p.xmax - p.xmin - 2 * buffer_m) * (p.ymax - p.ymin - 2 * buffer_m)
        if len(h) < 10:
            rows.append({"plot_id": p.plot_id, "lidar_n": len(h)})
            continue
        veg = h[h > canopy_threshold]
        rows.append({"plot_id": p.plot_id, "lidar_n": len(h), "lidar_density": len(h) / area,
                     "height_p50": float(np.percentile(veg, 50)) if len(veg) else 0.0,
                     "height_p90": float(np.percentile(veg, 90)) if len(veg) else 0.0,
                     "height_p99": float(np.percentile(veg, 99)) if len(veg) else 0.0,
                     "height_max": float(h.max()),
                     "canopy_cover_lidar": len(veg) / len(h),
                     "height_cv": float(veg.std() / veg.mean()) if len(veg) > 1 and veg.mean() > 0 else np.nan,
                     "profile_volume": float(np.percentile(veg, 90) * len(veg) / len(h)) if len(veg) else 0.0})
    return pd.DataFrame(rows)


def plant_traits_3d(xyz: np.ndarray, labels: np.ndarray | None = None, ground_label: int = 0,
                    stem_label: int = 1, up_axis: int = 2) -> dict:
    """Single-plant traits from a (laser-scanned) point cloud in consistent units.

    If ``labels`` are given (e.g. Pheno4D: 0 = soil, 1 = stem, ≥2 = leaf instances) leaf
    count and per-organ stats are returned; otherwise the soil is removed by RANSAC.
    """
    xyz = np.asarray(xyz, float)
    out = {}
    if labels is not None:
        labels = np.asarray(labels).astype(int)
        soil = labels == ground_label
        plant = ~soil
        if soil.sum() > 50:
            n, d, _ = ransac_plane(xyz[soil], threshold=np.ptp(xyz[soil][:, up_axis]) * 0.05 + 1e-6)
            ground_h = height_above_plane(xyz, n, d)
        else:
            ground_h = xyz[:, up_axis] - xyz[plant, up_axis].min()
        leaves = np.unique(labels[labels > stem_label])
        out["leaf_count"] = int(len(leaves))
        out["leaf_points"] = {int(l): int((labels == l).sum()) for l in leaves}
    else:
        z = xyz[:, up_axis]
        cand = z < np.percentile(z, 30)
        n, d, ground = ransac_plane(xyz, threshold=max(np.ptp(z) * 0.01, 1e-6), candidate_mask=cand)
        ground_h = height_above_plane(xyz, n, d)
        plant = ground_h > max(np.ptp(z) * 0.01, 1e-6) * 2
    p = xyz[plant]
    h = ground_h[plant]
    out["n_points"] = int(plant.sum())
    out["height"] = float(np.percentile(h, 99.5)) if len(h) else 0.0
    horiz = [a for a in range(3) if a != up_axis]
    if len(p) >= 4:
        try:
            out["convex_hull_volume"] = float(ConvexHull(p).volume)
            out["projected_area_hull"] = float(ConvexHull(p[:, horiz]).volume)
        except Exception:
            out["convex_hull_volume"] = out["projected_area_hull"] = np.nan
        ext = np.ptp(p[:, horiz], axis=0)
        out["width_major"], out["width_minor"] = float(ext.max()), float(ext.min())
    return out
