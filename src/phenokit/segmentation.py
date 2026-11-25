"""Classical computer-vision segmentation of vegetation and plant objects."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .indices import rgb_index


def otsu_threshold(values, bins: int = 256) -> float:
    """Otsu (1979) threshold maximising between-class variance."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    # for well-separated modes the criterion is flat across the empty gap; take the centre of the
    # tied plateau rather than its first bin, giving the widest margin to both classes
    best = np.flatnonzero(between >= between.max() * (1 - 1e-12))
    run = best[(best >= best[0]) & (best <= best[0] + np.flatnonzero(np.diff(np.r_[best, best[-1] + 2]) > 1)[0])]
    return float(centres[run].mean())


def clean_mask(mask, min_size: int = 20, opening: int = 1, fill_holes: bool = True):
    m = np.asarray(mask, bool)
    if opening:
        m = ndimage.binary_opening(m, iterations=opening)
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    if min_size:
        lab, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sizes >= min_size
            m = keep[lab]
    return m


def segment_vegetation_rgb(rgb, method: str = "exg_otsu", **kw):
    """Vegetation mask from RGB.

    * ``exg_otsu``: Excess Green + Otsu (Woebbecke et al., 1995).
    * ``exgr``: ExG − ExR > 0 (Meyer & Neto, 2008) — threshold-free.
    * ``kmeans_lab``: 2-cluster k-means on CIE a*/b*, vegetation = most negative a* (greenest).
    """
    rgb = np.asarray(rgb)
    if method == "exg_otsu":
        exg = rgb_index("ExG", rgb)
        m = exg > otsu_threshold(exg)
    elif method == "exgr":
        m = rgb_index("ExGR", rgb) > kw.get("threshold", 0.0)
    elif method == "kmeans_lab":
        from sklearn.cluster import KMeans

        lab = rgb_to_lab(rgb)
        X = lab[..., 1:].reshape(-1, 2)
        rng = np.random.default_rng(0)
        sample = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
        km = KMeans(2, n_init=4, random_state=0).fit(sample)
        veg = int(np.argmin(km.cluster_centers_[:, 0]))
        m = (km.predict(X) == veg).reshape(rgb.shape[:2])
    else:
        raise ValueError(method)
    return clean_mask(m, **{k: v for k, v in kw.items() if k in ("min_size", "opening", "fill_holes")})


def segment_vegetation_spectral(index_img, threshold: float | None = None):
    v = np.asarray(index_img, float)
    return v > (otsu_threshold(v) if threshold is None else threshold)


def rgb_to_lab(rgb):
    """sRGB (uint8 or 0–1) → CIE L*a*b* (D65)."""
    c = np.asarray(rgb, float)[..., :3]
    if c.max() > 1.0:
        c = c / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def segmentation_metrics(pred, truth) -> dict:
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    tn = (~p & ~t).sum()
    return {"iou": tp / max(tp + fp + fn, 1), "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "accuracy": (tp + tn) / p.size}


def object_features(rgb, mask, min_size: int = 30, nir=None):
    """Label connected plant objects and compute shape/colour features per object.

    Returns ``(labels, DataFrame)``; useful for crop/weed discrimination and plant counting.
    """
    import pandas as pd

    lab, n = ndimage.label(np.asarray(mask, bool))
    rgb = np.asarray(rgb, float)
    exg = rgb_index("ExG", rgb)
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        obj = lab[sl] == i
        area = int(obj.sum())
        if area < min_size:
            continue
        rr, cc = np.nonzero(obj)
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        perim = int((obj & ~ndimage.binary_erosion(obj)).sum())
        cov = np.cov(np.vstack([rr, cc])) if area > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ecc = float(np.sqrt(1 - ev[1] / ev[0])) if ev[0] > 0 else 0.0
        h, w = obj.shape
        pix = rgb[rr, cc, :3]
        row = {"label": i, "area_px": area, "perimeter_px": perim,
               "compactness": 4 * np.pi * area / max(perim, 1) ** 2, "eccentricity": ecc,
               "extent": area / (h * w), "bbox_aspect": max(h, w) / max(min(h, w), 1),
               "mean_r": pix[:, 0].mean(), "mean_g": pix[:, 1].mean(), "mean_b": pix[:, 2].mean(),
               "std_g": pix[:, 1].std(), "mean_exg": exg[rr, cc].mean(), "centroid_r": rr.mean(),
               "centroid_c": cc.mean()}
        if nir is not None:
            row["mean_nir"] = float(np.asarray(nir, float)[rr, cc].mean())
        rows.append(row)
    return lab, pd.DataFrame(rows)
