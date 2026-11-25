"""Radiometric calibration for spectral and thermal sensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmpiricalLine:
    """Per-band linear DN→reflectance model fitted on reference panels (Smith & Milton, 1999)."""

    gain: np.ndarray
    offset: np.ndarray
    r2: np.ndarray
    n_panels: int

    def apply(self, dn: np.ndarray) -> np.ndarray:
        return np.asarray(dn, float) * self.gain + self.offset


def fit_empirical_line(dn_cube: np.ndarray, panel_masks: list[np.ndarray], panel_reflectance) -> EmpiricalLine:
    """Fit gain/offset per band.

    ``panel_reflectance`` is either a list of scalars (flat grey panels) or a list of
    per-band certified reflectance spectra (one per panel). At least two panels spanning
    the scene's brightness range are needed; three or more allow a residual check.
    """
    if len(panel_masks) < 2:
        raise ValueError("Empirical line needs ≥2 panels (dark and bright)")
    B = dn_cube.shape[-1]
    dn = np.array([np.asarray(dn_cube[m], float).mean(0) for m in panel_masks])  # (P, B)
    refl = np.array([np.full(B, r) if np.isscalar(r) else np.asarray(r, float) for r in panel_reflectance])
    gain, offset, r2 = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        A = np.vstack([dn[:, b], np.ones(len(dn))]).T
        (g, o), *_ = np.linalg.lstsq(A, refl[:, b], rcond=None)
        pred = g * dn[:, b] + o
        ss = ((refl[:, b] - refl[:, b].mean()) ** 2).sum()
        gain[b], offset[b] = g, o
        r2[b] = 1 - ((refl[:, b] - pred) ** 2).sum() / ss if ss > 0 else np.nan
    return EmpiricalLine(gain, offset, r2, len(panel_masks))


def dark_white_correction(raw, dark, white, white_reflectance=0.99):
    """Reflectance = (raw − dark) / (white − dark) × ρ_white (line-scanner / lab imaging)."""
    raw, dark, white = (np.asarray(a, float) for a in (raw, dark, white))
    return (raw - dark) / np.maximum(white - dark, 1e-9) * white_reflectance


def flat_field(cube, reference_mask, reference_reflectance=1.0):
    """Relative reflectance by dividing by the mean spectrum of a bright, spectrally flat area.

    Fallback when no calibration panels were imaged (e.g. archived airborne radiance). The result
    is only as good as the flatness assumption of the reference (bare dry soil, a bright roof);
    record it in the metadata as ``processing_level = relative_reflectance``.
    """
    c = np.asarray(cube, float)
    ref = c[np.asarray(reference_mask, bool)].mean(0)
    return c / np.maximum(ref, 1e-9) * np.asarray(reference_reflectance, float)


def iarr(cube, mask=None):
    """Internal Average Relative Reflectance (Kruse, 1988): divide by the scene-average spectrum."""
    c = np.asarray(cube, float)
    ref = c.reshape(-1, c.shape[-1]).mean(0) if mask is None else c[np.asarray(mask, bool)].mean(0)
    return c / np.maximum(ref, 1e-9)


def irradiance_normalise(dn, irradiance, reference_irradiance):
    """Correct for changing illumination with a downwelling light sensor (DLS) per band."""
    return np.asarray(dn, float) * (np.asarray(reference_irradiance) / np.asarray(irradiance))


# --------------------------------------------------------------------------- thermal

SIGMA = 5.670374419e-8


def surface_temperature(brightness_temp_c, emissivity=0.98, reflected_temp_c=None, atm_transmittance=1.0,
                        atm_temp_c=None):
    """Broadband radiometric correction from at-sensor brightness temperature.

    T_s = [(T_b⁴ − (1−τ)T_atm⁴)/τ − (1−ε)T_refl⁴] / ε, all in kelvin (Stefan–Boltzmann approximation).
    Leaf emissivity is typically 0.95–0.99. ``reflected_temp_c`` defaults to the brightness
    temperature (i.e. no reflection correction) if unknown — measure it with a crumpled-foil target.
    """
    tb = np.asarray(brightness_temp_c, float) + 273.15
    tr = tb if reflected_temp_c is None else np.asarray(reflected_temp_c, float) + 273.15
    ta = tb if atm_temp_c is None else np.asarray(atm_temp_c, float) + 273.15
    w = (tb ** 4 - (1 - atm_transmittance) * ta ** 4) / atm_transmittance
    ts4 = (w - (1 - emissivity) * tr ** 4) / emissivity
    return np.power(np.maximum(ts4, 0), 0.25) - 273.15
