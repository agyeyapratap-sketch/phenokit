"""Field-trial plot layouts and plot-level (zonal) statistics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import GeoTransform


@dataclass
class PlotLayout:
    """Rectangular plots in field/map coordinates (metres). One row per plot."""

    table: pd.DataFrame  # columns: plot_id, xmin, xmax, ymin, ymax (+ any design columns)

    @classmethod
    def grid(cls, n_rows: int, n_cols: int, plot_w: float, plot_l: float, gap_w: float, gap_l: float,
             x0: float = 0.0, y0: float = 0.0, design: pd.DataFrame | None = None) -> "PlotLayout":
        """Serpentine-free row/column grid starting at the south-west corner (x0, y0)."""
        recs = []
        for r in range(n_rows):
            for c in range(n_cols):
                xmin = x0 + c * (plot_w + gap_w)
                ymin = y0 + r * (plot_l + gap_l)
                recs.append({"plot_id": f"P{r + 1:02d}{c + 1:02d}", "row": r + 1, "col": c + 1,
                             "xmin": xmin, "xmax": xmin + plot_w, "ymin": ymin, "ymax": ymin + plot_l})
        t = pd.DataFrame(recs)
        if design is not None:
            t = t.merge(design, on="plot_id", how="left")
        return cls(t)

    @classmethod
    def from_csv(cls, path) -> "PlotLayout":
        t = pd.read_csv(path)
        need = {"plot_id", "xmin", "xmax", "ymin", "ymax"}
        if not need <= set(t.columns):
            raise ValueError(f"plot layout needs columns {need}")
        return cls(t)

    def to_csv(self, path):
        self.table.to_csv(path, index=False)

    def windows(self, transform: GeoTransform, buffer_m: float = 0.0, shape=None):
        """Pixel windows (r0, r1, c0, c1) per plot; ``buffer_m`` shrinks plots to avoid edge pixels."""
        out = {}
        for _, p in self.table.iterrows():
            r0, c0 = transform.map_to_pixel(p.xmin + buffer_m, p.ymax - buffer_m)
            r1, c1 = transform.map_to_pixel(p.xmax - buffer_m, p.ymin + buffer_m)
            r0, r1 = int(np.ceil(r0 - 1e-6)), int(np.floor(r1 + 1e-6))
            c0, c1 = int(np.ceil(c0 - 1e-6)), int(np.floor(c1 + 1e-6))
            if shape is not None:
                r0, c0 = max(r0, 0), max(c0, 0)
                r1, c1 = min(r1, shape[0]), min(c1, shape[1])
            out[p.plot_id] = (r0, r1, c0, c1)
        return out

    def assign_points(self, xy: np.ndarray, buffer_m: float = 0.0) -> np.ndarray:
        """Plot index (row in table) for each XY point, −1 if outside all plots."""
        xy = np.asarray(xy, float)
        idx = np.full(len(xy), -1, int)
        for k, p in enumerate(self.table.itertuples()):
            inside = ((xy[:, 0] >= p.xmin + buffer_m) & (xy[:, 0] < p.xmax - buffer_m) &
                      (xy[:, 1] >= p.ymin + buffer_m) & (xy[:, 1] < p.ymax - buffer_m))
            idx[inside] = k
        return idx


def zonal_stats(raster: np.ndarray, layout: PlotLayout, transform: GeoTransform, mask: np.ndarray | None = None,
                stats=("mean", "median", "std", "p90"), buffer_m: float = 0.0, prefix: str = "",
                band_names=None) -> pd.DataFrame:
    """Per-plot statistics of a 2-D raster or each band of a 3-D raster.

    ``mask`` (same H×W) restricts statistics to e.g. vegetation pixels; ``n_px`` and
    ``valid_frac`` report support so that sparse plots can be flagged downstream.
    """
    r = np.asarray(raster, float)
    if r.ndim == 2:
        r = r[..., None]
    names = band_names or ([prefix.rstrip("_")] if r.shape[-1] == 1 and prefix else
                           [f"{prefix}b{b}" for b in range(r.shape[-1])])
    funcs = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd, "min": np.nanmin, "max": np.nanmax,
             "p10": lambda a, axis: np.nanpercentile(a, 10, axis=axis),
             "p90": lambda a, axis: np.nanpercentile(a, 90, axis=axis),
             "p99": lambda a, axis: np.nanpercentile(a, 99, axis=axis)}
    rows = []
    for pid, (r0, r1, c0, c1) in layout.windows(transform, buffer_m, r.shape[:2]).items():
        win = r[r0:r1, c0:c1].reshape(-1, r.shape[-1])
        total = len(win)
        if mask is not None:
            win = win[np.asarray(mask[r0:r1, c0:c1], bool).ravel()]
        row = {"plot_id": pid, f"{prefix}n_px": len(win), f"{prefix}valid_frac": len(win) / max(total, 1)}
        for s in stats:
            vals = funcs[s](win, axis=0) if len(win) else np.full(r.shape[-1], np.nan)
            for n, v in zip(names, np.atleast_1d(vals)):
                row[f"{n}_{s}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_spectra(cube, layout: PlotLayout, transform: GeoTransform, mask=None, buffer_m: float = 0.0):
    """Mean (masked) spectrum per plot → (plot_ids, spectra[n_plots, bands])."""
    ids, spec = [], []
    for pid, (r0, r1, c0, c1) in layout.windows(transform, buffer_m, cube.shape[:2]).items():
        w = np.asarray(cube[r0:r1, c0:c1], float).reshape(-1, cube.shape[-1])
        if mask is not None:
            w = w[np.asarray(mask[r0:r1, c0:c1], bool).ravel()]
        ids.append(pid)
        spec.append(w.mean(0) if len(w) else np.full(cube.shape[-1], np.nan))
    return ids, np.array(spec)
