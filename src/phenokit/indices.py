"""Vegetation-index library resolved by wavelength, so the same code serves handheld,
multispectral and hyperspectral sensors.

Each index declares the wavelengths it needs; :func:`compute` picks the nearest bands and
raises if a sensor cannot support an index (rather than silently using a wrong band).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .spectral import nearest_band

eps = 1e-9


@dataclass(frozen=True)
class Index:
    name: str
    bands_nm: tuple[float, ...]
    func: Callable
    reference: str
    description: str


def _valid_range(x, lo, hi):
    """NaN outside the physically meaningful range (e.g. REIP for non-vegetated pixels)."""
    x = np.asarray(x, float)
    return np.where((x >= lo) & (x <= hi), x, np.nan)


def _nd(a, b):
    return (a - b) / (a + b + eps)


INDICES: dict[str, Index] = {i.name: i for i in [
    Index("NDVI", (800, 670), lambda n, r: _nd(n, r), "Rouse et al. 1974", "greenness / canopy cover"),
    Index("GNDVI", (800, 550), lambda n, g: _nd(n, g), "Gitelson et al. 1996", "chlorophyll, less saturating"),
    Index("NDRE", (790, 720), lambda n, re: _nd(n, re), "Barnes et al. 2000", "red-edge chlorophyll/N status"),
    Index("CIre", (790, 720), lambda n, re: n / (re + eps) - 1, "Gitelson et al. 2005", "canopy chlorophyll content"),
    Index("SAVI", (800, 670), lambda n, r: 1.5 * (n - r) / (n + r + 0.5), "Huete 1988", "soil-adjusted greenness"),
    Index("EVI", (800, 670, 470),
          lambda n, r, b: 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1), "Huete et al. 2002", "enhanced VI"),
    Index("MCARI", (700, 670, 550),
          lambda r700, r670, r550: ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / (r670 + eps)),
          "Daughtry et al. 2000", "chlorophyll absorption"),
    Index("PRI", (531, 570), lambda a, b: _nd(a, b), "Gamon et al. 1992", "xanthophyll / light-use efficiency"),
    Index("WBI", (900, 970), lambda a, b: a / (b + eps), "Peñuelas et al. 1993", "canopy water"),
    Index("REIP", (670, 700, 740, 780),
          lambda r670, r700, r740, r780: _valid_range(
              700 + 40 * (((r670 + r780) / 2 - r700) / (r740 - r700 + eps)), 670, 780),
          "Guyot & Baret 1988", "red-edge inflection point (nm)"),
    Index("MTCI", (754, 709, 681), lambda a, b, c: (a - b) / (b - c + eps), "Dash & Curran 2004",
          "terrestrial chlorophyll index"),
]}

# RGB indices operate on chromatic coordinates (r, g, b normalised to sum to one)
RGB_INDICES = {
    "ExG": lambda r, g, b: 2 * g - r - b,                      # Woebbecke et al. 1995
    "ExR": lambda r, g, b: 1.4 * r - g,                         # Meyer et al. 1998
    "ExGR": lambda r, g, b: (2 * g - r - b) - (1.4 * r - g),    # Meyer & Neto 2008
    "VARI": lambda r, g, b: (g - r) / (g + r - b + eps),        # Gitelson et al. 2002
    "GLI": lambda r, g, b: (2 * g - r - b) / (2 * g + r + b + eps),
    "NGRDI": lambda r, g, b: (g - r) / (g + r + eps),
}


def compute(name: str, data: np.ndarray, wavelengths, tol_nm: float = 15.0, fwhm=None) -> np.ndarray:
    """Compute a spectral index on ``data[..., bands]`` given band-centre ``wavelengths`` (and optional FWHM)."""
    idx = INDICES[name]
    sel = [nearest_band(wavelengths, w, tol_nm, fwhm) for w in idx.bands_nm]
    if len(set(sel)) < len(sel):
        raise ValueError(f"{name}: sensor maps distinct wavelengths {idx.bands_nm} onto the same band")
    bands = [np.asarray(data[..., b], float) for b in sel]
    return idx.func(*bands)


def supported(wavelengths, tol_nm: float = 15.0, fwhm=None) -> list[str]:
    out = []
    for n, idx in INDICES.items():
        sel = []
        try:
            sel = [nearest_band(wavelengths, w, tol_nm, fwhm) for w in idx.bands_nm]
        except ValueError:
            continue
        if len(set(sel)) == len(sel):
            out.append(n)
    return out


def compute_all(data, wavelengths, names=None, tol_nm: float = 15.0, fwhm=None) -> dict[str, np.ndarray]:
    names = names or supported(wavelengths, tol_nm, fwhm)
    return {n: compute(n, data, wavelengths, tol_nm, fwhm) for n in names}


def chromatic(rgb: np.ndarray):
    rgb = np.asarray(rgb, float)
    s = rgb[..., :3].sum(-1) + eps
    return rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s


def rgb_index(name: str, rgb: np.ndarray) -> np.ndarray:
    return RGB_INDICES[name](*chromatic(rgb))


def red_edge_position(wavelengths, spectra, lo: float = 680, hi: float = 760) -> np.ndarray:
    """Red-edge position as wavelength of maximum first derivative (hyperspectral only)."""
    wl = np.asarray(wavelengths, float)
    sel = (wl >= lo) & (wl <= hi)
    d = np.gradient(np.asarray(spectra, float)[..., sel], wl[sel], axis=-1)
    return wl[sel][np.argmax(d, axis=-1)]
