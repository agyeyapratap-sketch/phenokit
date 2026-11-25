"""Quality control and data-integrity checks for phenotyping data.

Every check returns :class:`QCResult` objects with a PASS / WARN / FAIL status so results
can be aggregated into a report, stored in the catalogue and used to gate the pipeline.
Thresholds are explicit arguments: tune them per sensor and document them in the SOP.
"""
from __future__ import annotations

import base64
import html
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_ORDER = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class QCResult:
    target: str
    check: str
    status: str
    value: float | None = None
    threshold: str = ""
    message: str = ""


def _grade(value, warn, fail, higher_is_bad=True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return FAIL
    if higher_is_bad:
        return FAIL if value > fail else WARN if value > warn else PASS
    return FAIL if value < fail else WARN if value < warn else PASS


def worst(results) -> str:
    return max((r.status for r in results), key=_ORDER.get, default=PASS)


# --------------------------------------------------------------------------- RGB / frame images


def to_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, float)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    return img


def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness: variance of the Laplacian (low = blurred). Scale to 0–255 grey first."""
    g = to_gray(img)
    if g.max() <= 1.0:
        g = g * 255
    return float(ndimage.laplace(g).var())


def noise_sigma(img: np.ndarray) -> float:
    """Immerkær (1996) fast noise variance estimator, returns sigma on the image's scale."""
    g = to_gray(img)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    conv = ndimage.convolve(g, k, mode="reflect")
    H, W = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (W - 2) * (H - 2)))


def qc_rgb(img: np.ndarray, target: str = "rgb", blur_warn: float = 100.0, blur_fail: float = 30.0,
           clip_warn: float = 0.01, clip_fail: float = 0.05, dark_fail: float = 40.0) -> list[QCResult]:
    """Sharpness, clipping (over/under-exposure), brightness and colour-cast checks for 8-bit RGB."""
    img = np.asarray(img)
    rgb = img[..., :3].astype(float)
    res = []
    lv = laplacian_variance(rgb)
    res.append(QCResult(target, "sharpness_laplacian_var", _grade(lv, blur_warn, blur_fail, False), lv,
                        f"warn<{blur_warn}, fail<{blur_fail}", "low values indicate motion/focus blur"))
    hi = float((rgb >= 254).any(-1).mean())
    lo = float((rgb <= 1).all(-1).mean())
    res.append(QCResult(target, "overexposed_fraction", _grade(hi, clip_warn, clip_fail), hi,
                        f"warn>{clip_warn}, fail>{clip_fail}"))
    res.append(QCResult(target, "underexposed_fraction", _grade(lo, clip_warn, clip_fail), lo,
                        f"warn>{clip_warn}, fail>{clip_fail}"))
    mean = float(rgb.mean())
    res.append(QCResult(target, "mean_brightness", _grade(mean, dark_fail * 1.5, dark_fail, False), mean,
                        f"warn<{dark_fail * 1.5}, fail<{dark_fail}"))
    ch = rgb.reshape(-1, 3).mean(0)
    cast = float(ch.max() / max(ch.min(), 1e-6))
    res.append(QCResult(target, "colour_cast_ratio", _grade(cast, 2.0, 3.0), cast, "warn>2, fail>3",
                        "max/min channel mean; vegetation scenes are typically 1.1–1.8"))
    res.append(QCResult(target, "noise_sigma", PASS, noise_sigma(rgb), "info"))
    return res


def average_hash(img: np.ndarray, size: int = 8) -> int:
    """64-bit perceptual (average) hash for duplicate/near-duplicate image detection."""
    from PIL import Image

    g = to_gray(img)
    g = (255 * (g - g.min()) / max(np.ptp(g), 1e-9)).astype(np.uint8)
    small = np.asarray(Image.fromarray(g).resize((size, size), Image.BILINEAR), float)
    bits = (small > small.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def find_near_duplicates(hashes: dict[str, int], max_hamming: int = 4) -> list[tuple[str, str, int]]:
    keys = list(hashes)
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = bin(hashes[keys[i]] ^ hashes[keys[j]]).count("1")
            if d <= max_hamming:
                out.append((keys[i], keys[j], d))
    return out


# --------------------------------------------------------------------------- spectral cubes / spectra


def band_snr(cube: np.ndarray, mask: np.ndarray | None = None, max_pixels: int = 50_000,
             seed: int = 0) -> np.ndarray:
    """Per-band SNR = mean signal / noise, noise by spectral–spatial decorrelation (SSDC).

    Each band is regressed on its two spectral neighbours and its spatial neighbour
    (Roger & Arnold, 1996); the residual is dominated by sensor noise because scene
    structure is shared across adjacent bands/pixels. This works on heterogeneous canopies
    where simple pixel differencing would mistake crop-row texture for noise. A uniform
    target (panel) gives a complementary homogeneous-area estimate.
    """
    c = np.asarray(cube, float)
    H, W, B = c.shape
    right = np.concatenate([c[:, 1:], c[:, -1:]], 1)
    valid = np.ones((H, W), bool) if mask is None else np.asarray(mask, bool)
    valid[:, -1] = False
    idx = np.flatnonzero(valid)
    if len(idx) > max_pixels:
        idx = np.random.default_rng(seed).choice(idx, max_pixels, replace=False)
    X = c.reshape(-1, B)[idx]
    R = right.reshape(-1, B)[idx]
    snr = np.empty(B)
    for b in range(B):
        nb = [b - 1, b + 1] if 0 < b < B - 1 else ([1, 2] if b == 0 else [B - 2, B - 3])
        A = np.c_[np.ones(len(X)), X[:, nb], R[:, b]]
        coef, *_ = np.linalg.lstsq(A, X[:, b], rcond=None)
        resid = X[:, b] - A @ coef
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        snr[b] = X[:, b].mean() / max(noise, 1e-12)
    return snr


def detect_bad_bands(cube: np.ndarray, wavelengths=None, snr_min: float = 10.0,
                     sat_value: float | None = None, sat_frac: float = 0.01,
                     known_absorption: bool = True) -> dict:
    """Flag bands that are noisy, dead, saturated, non-finite or inside atmospheric water absorption."""
    c = np.asarray(cube)
    B = c.shape[-1]
    flat = c.reshape(-1, B).astype(float)
    reasons: dict[int, list[str]] = {b: [] for b in range(B)}
    finite = np.isfinite(flat)
    for b in np.where(finite.mean(0) < 0.999)[0]:
        reasons[b].append("non-finite")
    std = np.nanstd(flat, 0)
    for b in np.where(std < 1e-9)[0]:
        reasons[b].append("dead (zero variance)")
    snr = band_snr(np.nan_to_num(c))
    for b in np.where(snr < snr_min)[0]:
        reasons[b].append(f"low SNR ({snr[b]:.1f})")
    if sat_value is not None:
        for b in np.where((flat >= sat_value).mean(0) > sat_frac)[0]:
            reasons[b].append("saturated")
    if known_absorption and wavelengths is not None:
        wl = np.asarray(wavelengths)
        for lo, hi in ((1340, 1460), (1790, 1960), (2450, 2600)):
            for b in np.where((wl >= lo) & (wl <= hi))[0]:
                reasons[b].append(f"atmospheric water absorption {lo}-{hi} nm")
    bad = [b for b, r in reasons.items() if r]
    return {"bad_bands": bad, "reasons": {b: r for b, r in reasons.items() if r}, "snr": snr}


def detect_spectral_spikes(spectra: np.ndarray, z: float = 6.0) -> np.ndarray:
    """Boolean mask of spikes in finely sampled point spectra (e.g. 1 nm spectroradiometer data)
    using second-difference robust z-scores. For imaging cubes use :func:`detect_spatial_spikes`."""
    s = np.atleast_2d(spectra).astype(float)
    d2 = np.zeros_like(s)
    d2[:, 1:-1] = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
    med = np.median(d2, 1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(d2 - med), 1, keepdims=True) + 1e-12
    return np.abs(d2 - med) / mad > z


def detect_spatial_spikes(cube: np.ndarray, z: float = 8.0, size: int = 3) -> np.ndarray:
    """Impulse noise / hot pixels in imaging cubes: robust z of each pixel vs its spatial median (per band).

    A spike must be (i) a spatial outlier in its band and (ii) spectrally isolated — the same
    pixel is *not* anomalous in the adjacent bands. Condition (ii) stops field boundaries and
    small objects (anomalous in every band) being counted. Genuine narrow absorption features
    are spatially coherent and are therefore not flagged either.
    """
    c = np.asarray(cube, float)
    med = ndimage.median_filter(c, size=(size, size, 1), mode="reflect")
    r = c - med
    mad = 1.4826 * np.median(np.abs(r - np.median(r, (0, 1))), (0, 1)) + 1e-12
    zz = np.abs(r) / mad
    out = zz > z
    calm = zz < z / 3
    out[..., 1:-1] &= calm[..., :-2] & calm[..., 2:]
    out[..., 0] &= calm[..., 1]
    out[..., -1] &= calm[..., -2]
    return out


def qc_reflectance(cube_or_spectra: np.ndarray, target: str = "reflectance", lo: float = -0.02,
                   hi: float = 1.2, frac_warn: float = 0.001, frac_fail: float = 0.01) -> list[QCResult]:
    """Physical plausibility of calibrated reflectance (0–1, small tolerance)."""
    a = np.asarray(cube_or_spectra, float)
    out_frac = float(((a < lo) | (a > hi) | ~np.isfinite(a)).mean())
    return [QCResult(target, "reflectance_out_of_range", _grade(out_frac, frac_warn, frac_fail), out_frac,
                     f"[{lo},{hi}] warn>{frac_warn} fail>{frac_fail}")]


def qc_panel(measured: np.ndarray, expected: np.ndarray, target: str = "panel",
             tol_warn: float = 0.02, tol_fail: float = 0.05) -> list[QCResult]:
    """Check calibrated reflectance of a reference panel (or validation panel) against its certificate."""
    err = float(np.nanmean(np.abs(np.asarray(measured) - np.asarray(expected))))
    return [QCResult(target, "panel_abs_error", _grade(err, tol_warn, tol_fail), err,
                     f"warn>{tol_warn}, fail>{tol_fail}", "mean |measured − certified| reflectance")]


def qc_spectrum_splice(wl: np.ndarray, spectra: np.ndarray, joins=(1000.0, 1800.0), target: str = "spectra",
                       warn: float = 0.01, fail: float = 0.02) -> list[QCResult]:
    """Detector-join step check for 3-detector spectroradiometers (VNIR/SWIR1/SWIR2)."""
    wl = np.asarray(wl)
    s = np.atleast_2d(spectra)
    out = []
    for j in joins:
        i = int(np.searchsorted(wl, j))
        if 3 <= i < len(wl) - 3:
            # step = discontinuity relative to linear extrapolation from either side
            left = s[:, i - 1] + (s[:, i - 1] - s[:, i - 3]) / 2
            step = float(np.median(np.abs(s[:, i] - left)))
            out.append(QCResult(target, f"splice_step_{int(j)}nm", _grade(step, warn, fail), step,
                                f"warn>{warn}, fail>{fail}"))
    return out


# --------------------------------------------------------------------------- thermal


def radial_trend(temp_c: np.ndarray, covariate: np.ndarray | None = None):
    """Fit T = a + b·covariate + c·r² (r = normalised distance from image centre, r=1 at edge centres).

    ``c`` estimates vignetting / non-uniformity in °C; including a land-cover covariate (e.g.
    NDVI or vegetation fraction) stops hot soil margins being mistaken for vignetting.
    Returns ``(c, r2_map)``.
    """
    t = np.asarray(temp_c, float)
    H, W = t.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((yy - (H - 1) / 2) / (H / 2)) ** 2 + ((xx - (W - 1) / 2) / (W / 2)) ** 2
    ok = np.isfinite(t)
    cols = [np.ones(ok.sum())]
    if covariate is not None:
        cv = np.asarray(covariate, float)
        ok &= np.isfinite(cv)
        cols = [np.ones(ok.sum()), cv[ok]]
    A = np.c_[tuple(cols) + (r2[ok],)]
    coef, *_ = np.linalg.lstsq(A, t[ok], rcond=None)
    return float(coef[-1]), r2


def qc_thermal(temp_c: np.ndarray, target: str = "thermal", valid=(-10.0, 70.0), nan_warn=0.001,
               nan_fail=0.02, covariate: np.ndarray | None = None, vignetting_warn=0.5,
               vignetting_fail=1.0) -> list[QCResult]:
    t = np.asarray(temp_c, float)
    bad = float(((t < valid[0]) | (t > valid[1]) | ~np.isfinite(t)).mean())
    res = [QCResult(target, "invalid_temperature_fraction", _grade(bad, nan_warn, nan_fail), bad,
                    f"valid {valid} °C")]
    c, _ = radial_trend(np.where((t > valid[0]) & (t < valid[1]), t, np.nan), covariate)
    res.append(QCResult(target, "radial_nonuniformity_c", _grade(abs(c), vignetting_warn, vignetting_fail), c,
                        f"warn>{vignetting_warn}, fail>{vignetting_fail}",
                        "°C rise at image edge relative to centre (cover-adjusted); suggests vignetting/NUC drift"))
    res.append(QCResult(target, "temperature_range_c", PASS, float(np.nanpercentile(t, 99) - np.nanpercentile(t, 1)),
                        "info"))
    return res


def qc_thermal_references(ref_series_c: np.ndarray, target: str = "thermal_refs", drift_warn=0.5,
                          drift_fail=1.0) -> list[QCResult]:
    """Drift of a stable reference target imaged repeatedly through a flight."""
    r = np.asarray(ref_series_c, float)
    drift = float(np.ptp(r))
    return [QCResult(target, "reference_drift_c", _grade(drift, drift_warn, drift_fail), drift,
                     f"warn>{drift_warn}, fail>{drift_fail}")]


# --------------------------------------------------------------------------- LiDAR / point clouds


def qc_pointcloud(xyz: np.ndarray, area_m2: float | None = None, target: str = "pointcloud",
                  min_density: float = 100.0, k: int = 8, outlier_std: float = 3.0) -> list[QCResult]:
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, float)
    n = len(xyz)
    res = []
    if area_m2 is None:
        from scipy.spatial import ConvexHull

        area_m2 = ConvexHull(xyz[:, :2]).volume if n >= 3 else np.nan
    dens = n / area_m2 if area_m2 else np.nan
    res.append(QCResult(target, "point_density_per_m2", _grade(dens, min_density * 1.5, min_density, False), dens,
                        f"fail<{min_density}"))
    uniq = len(np.unique(np.round(xyz, 4), axis=0))
    dup = 1 - uniq / max(n, 1)
    res.append(QCResult(target, "duplicate_fraction", _grade(dup, 0.01, 0.05), dup, "warn>0.01, fail>0.05"))
    sample = xyz if n <= 1_000_000 else xyz[np.random.default_rng(0).choice(n, 1_000_000, replace=False)]
    d, _ = cKDTree(sample).query(sample, k=k + 1)
    md = d[:, 1:].mean(1)
    thr = md.mean() + outlier_std * md.std()
    of = float((md > thr).mean())
    res.append(QCResult(target, "statistical_outlier_fraction", _grade(of, 0.02, 0.05), of, "warn>0.02, fail>0.05"))
    res.append(QCResult(target, "z_range_m", PASS, float(np.ptp(xyz[:, 2])), "info"))
    # isolated points (birds, dust, multipath): k-NN distance far above the typical spacing
    iso = md > 10 * np.median(md)
    res.append(QCResult(target, "isolated_points", WARN if iso.sum() else PASS, int(iso.sum()), ">0 → inspect",
                        f"max z of isolated points {sample[iso, 2].max():.1f}" if iso.any() else ""))
    # coverage gaps (missed swath, occlusion): fraction of 1 m cells far below median density
    if area_m2 and area_m2 > 4:
        ij = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / 1.0).astype(int)
        shape = ij.max(0) + 1
        counts = np.bincount(np.ravel_multi_index(ij.T, shape), minlength=int(np.prod(shape)))
        med = np.median(counts[counts > 0])
        low = float((counts < 0.25 * med).mean())
        res.append(QCResult(target, "low_density_cell_fraction", _grade(low, 0.01, 0.05), low,
                            "cells <25% of median density; warn>0.01, fail>0.05", "gaps / missed swaths"))
    return res


# --------------------------------------------------------------------------- positioning


def qc_gnss_track(t_s: np.ndarray, xy: np.ndarray, fix: np.ndarray | None = None, target: str = "gnss",
                  max_gap_s: float = 1.0, max_speed_ms: float = 2.0, rtk_frac_min: float = 0.999) -> list[QCResult]:
    """Ground-vehicle / UAV position log: time gaps, speed limits and RTK-fixed fraction."""
    t = np.asarray(t_s, float)
    xy = np.asarray(xy, float)
    dt = np.diff(t)
    res = [QCResult(target, "max_time_gap_s", _grade(float(dt.max()), max_gap_s, 3 * max_gap_s), float(dt.max()),
                    f"warn>{max_gap_s}")]
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(dt, 1e-9)
    over = float((sp > max_speed_ms).mean())
    res.append(QCResult(target, "overspeed_fraction", _grade(over, 0.01, 0.05), over, f"speed>{max_speed_ms} m/s"))
    if fix is not None:
        frac = float(np.mean(np.asarray(fix) == "RTK_FIXED"))
        n_bad = int(np.sum(np.asarray(fix) != "RTK_FIXED"))
        res.append(QCResult(target, "rtk_fixed_fraction", _grade(frac, rtk_frac_min, 0.8, False), frac,
                            f"warn<{rtk_frac_min}, fail<0.8", f"{n_bad} records without RTK fix" if n_bad else ""))
    return res


# --------------------------------------------------------------------------- annotations & splits


def qc_split_leakage(splits: dict[str, list], target: str = "dataset_split") -> list[QCResult]:
    """Detect items appearing in more than one split (train/val/test leakage)."""
    names = list(splits)
    res = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sorted(set(splits[names[i]]) & set(splits[names[j]]))
            res.append(QCResult(target, f"overlap_{names[i]}_{names[j]}", FAIL if inter else PASS, len(inter), "0",
                                f"shared items: {inter}" if inter else ""))
    return res


def qc_mask_consistency(mask_a: np.ndarray, mask_b: np.ndarray, target: str = "labels",
                        iou_warn: float = 0.9, iou_fail: float = 0.75) -> list[QCResult]:
    """Agreement between two label sources (e.g. vegetation mask vs. union of crop/weed annotation)."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    iou = float((a & b).sum() / max((a | b).sum(), 1))
    return [QCResult(target, "label_iou", _grade(iou, iou_warn, iou_fail, False), iou,
                     f"warn<{iou_warn}, fail<{iou_fail}")]


def qc_plot_values(df, columns, id_col="plot_id", z: float = 4.0, target: str = "plot_table") -> list[QCResult]:
    """Missing plots and robust-z outliers in a plot-level trait table."""
    res = []
    for c in columns:
        v = df[c].to_numpy(float)
        miss = float(np.isnan(v).mean())
        res.append(QCResult(target, f"{c}_missing_fraction", _grade(miss, 0.0, 0.05), miss, "warn>0, fail>0.05"))
        med = np.nanmedian(v)
        mad = 1.4826 * np.nanmedian(np.abs(v - med)) + 1e-12
        out = df.loc[np.abs(v - med) / mad > z, id_col].tolist()
        res.append(QCResult(target, f"{c}_outliers", WARN if out else PASS, len(out), f"|robust z|>{z}",
                            f"plots: {out[:20]}" if out else ""))
    return res


# --------------------------------------------------------------------------- report


class QCReport:
    def __init__(self, title: str = "QC report"):
        self.title = title
        self.results: list[QCResult] = []
        self.figures: list[tuple[str, str]] = []

    def extend(self, results):
        self.results.extend(results)
        return self

    def add_figure(self, caption: str, fig) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        self.figures.append((caption, base64.b64encode(buf.getvalue()).decode()))

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in self.results])

    def summary(self) -> dict:
        df = self.to_frame()
        if df.empty:
            return {"overall": PASS, "counts": {}}
        return {"overall": worst(self.results), "counts": df.status.value_counts().to_dict()}

    def to_html(self, path: str | Path) -> Path:
        df = self.to_frame()
        colours = {PASS: "#d9f2d9", WARN: "#fff2cc", FAIL: "#f8d0d0"}
        rows = "".join(
            f"<tr style='background:{colours[r.status]}'><td>{html.escape(r.target)}</td><td>{html.escape(r.check)}</td>"
            f"<td><b>{r.status}</b></td><td>{'' if r.value is None else f'{r.value:.4g}'}</td>"
            f"<td>{html.escape(r.threshold)}</td><td>{html.escape(r.message)}</td></tr>" for r in self.results)
        s = self.summary()
        figs = "".join(f"<figure><img src='data:image/png;base64,{b}'/><figcaption>{html.escape(c)}</figcaption></figure>"
                       for c, b in self.figures)
        doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(self.title)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}th{{background:#eee}}figure{{display:inline-block;margin:1rem}}
img{{max-width:520px}}</style></head><body><h1>{html.escape(self.title)}</h1>
<p>Overall: <b>{s['overall']}</b> — {s['counts']}</p>
<table><tr><th>Target</th><th>Check</th><th>Status</th><th>Value</th><th>Threshold</th><th>Note</th></tr>{rows}</table>
{figs}</body></html>"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
        if not df.empty:
            df.to_csv(p.with_suffix(".csv"), index=False)
        return p
