"""Spectral preprocessing for imaging and point spectroscopy."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _2d(x):
    x = np.asarray(x, float)
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def nearest_band(wavelengths, target_nm: float, tol_nm: float = 15.0, fwhm=None) -> int:
    """Index of the band closest to ``target_nm``.

    A band qualifies if its centre is within ``tol_nm`` of the target or, for broadband
    sensors, within the band's own FWHM (so a 842 nm / 57 nm-wide NIR band can serve 800 nm).
    """
    wl = np.asarray(wavelengths, float)
    i = int(np.argmin(np.abs(wl - target_nm)))
    tol = max(tol_nm, float(np.asarray(fwhm, float)[i])) if fwhm is not None else tol_nm
    if abs(wl[i] - target_nm) > tol:
        raise ValueError(f"No band within {tol_nm} nm of {target_nm} nm (nearest {wl[i]:.1f})")
    return i


def savgol(spectra, window: int = 11, poly: int = 2, deriv: int = 0, delta: float = 1.0):
    s, shape = _2d(spectra)
    return savgol_filter(s, window, poly, deriv=deriv, delta=delta, axis=1).reshape(shape)


def snv(spectra):
    """Standard normal variate (Barnes et al., 1989): per-spectrum centring and scaling."""
    s, shape = _2d(spectra)
    return ((s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)).reshape(shape)


def msc(spectra, reference=None):
    """Multiplicative scatter correction (Geladi et al., 1985). Returns (corrected, reference)."""
    s, shape = _2d(spectra)
    ref = s.mean(0) if reference is None else np.asarray(reference, float)
    A = np.vstack([np.ones_like(ref), ref]).T
    coef, *_ = np.linalg.lstsq(A, s.T, rcond=None)  # (2, n)
    out = (s - coef[0][:, None]) / (coef[1][:, None] + 1e-12)
    return out.reshape(shape), ref


def continuum_removal(wavelengths, spectra):
    """Divide each spectrum by its upper convex hull (absorption-feature analysis)."""
    wl = np.asarray(wavelengths, float)
    s, shape = _2d(spectra)
    out = np.empty_like(s)
    for i, y in enumerate(s):
        hull = [0]
        for k in range(1, len(wl)):
            hull.append(k)
            while len(hull) >= 3:
                a, b, c = hull[-3], hull[-2], hull[-1]
                # remove b if it lies below segment a–c (upper hull)
                if (y[b] - y[a]) * (wl[c] - wl[a]) <= (y[c] - y[a]) * (wl[b] - wl[a]):
                    hull.pop(-2)
                else:
                    break
        cont = np.interp(wl, wl[hull], y[hull])
        out[i] = y / np.maximum(cont, 1e-12)
    return out.reshape(shape)


def gaussian_srf_resample(wavelengths, spectra, centres, fwhms):
    """Simulate a broadband sensor (e.g. multispectral camera) from narrow-band spectra.

    Each output band is the spectral-response-weighted mean with a Gaussian SRF.
    """
    wl = np.asarray(wavelengths, float)
    s, shape = _2d(spectra)
    W = []
    for c, f in zip(centres, fwhms):
        sig = f / 2.3548
        w = np.exp(-0.5 * ((wl - c) / sig) ** 2)
        W.append(w / w.sum())
    W = np.array(W).T  # (bands_in, bands_out)
    return (s @ W).reshape(shape[:-1] + (len(centres),))


def remove_ranges(wavelengths, spectra, ranges=((1340, 1460), (1790, 1960), (2450, 2600))):
    wl = np.asarray(wavelengths, float)
    keep = np.ones(len(wl), bool)
    for lo, hi in ranges:
        keep &= ~((wl >= lo) & (wl <= hi))
    return wl[keep], np.asarray(spectra)[..., keep], keep


def splice_correction(wavelengths, spectra, joins=(1000.0, 1800.0), reference: str = "vnir"):
    """Remove detector-join steps in 3-detector spectroradiometers by additive offsets.

    The step at each join is estimated by linear extrapolation from the preceding detector
    and the subsequent detector is shifted to remove it (VNIR is taken as the reference).
    """
    wl = np.asarray(wavelengths, float)
    s, shape = _2d(spectra)
    s = s.copy()
    for j in joins:
        i = int(np.searchsorted(wl, j))
        if 3 <= i < len(wl) - 1:
            pred = s[:, i - 1] + (s[:, i - 1] - s[:, i - 3]) / 2
            s[:, i:] -= (s[:, i] - pred)[:, None]
    return s.reshape(shape)


def pca_reduce(cube_or_spectra, n_components: int = 10, mask=None):
    """PCA on pixels (fit on masked pixels if given). Returns (scores, pca)."""
    from sklearn.decomposition import PCA

    s, shape = _2d(cube_or_spectra)
    fit = s if mask is None else s[np.asarray(mask).ravel()]
    pca = PCA(n_components).fit(fit)
    return pca.transform(s).reshape(shape[:-1] + (n_components,)), pca
