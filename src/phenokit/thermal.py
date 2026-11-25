"""Thermal phenotyping: canopy temperature extraction and crop water stress index."""
from __future__ import annotations

import numpy as np


def canopy_temperature(temp_c, veg_mask, percentile_band=(20, 80)):
    """Canopy temperature from vegetation pixels only, trimmed to suppress mixed soil pixels."""
    t = np.asarray(temp_c, float)[np.asarray(veg_mask, bool)]
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.nan
    lo, hi = np.percentile(t, percentile_band)
    return float(t[(t >= lo) & (t <= hi)].mean())


def cwsi(tc, t_wet, t_dry):
    """Empirical CWSI = (Tc − Twet)/(Tdry − Twet) (Idso et al., 1981; Jones, 1999). 0 = unstressed, 1 = no transpiration."""
    tc = np.asarray(tc, float)
    return np.clip((tc - t_wet) / np.maximum(np.asarray(t_dry) - np.asarray(t_wet), 1e-6), -0.2, 1.2)


def wet_dry_from_scene(temp_c, veg_mask, low_pct=2, high_pct=98):
    """Statistical wet/dry references from the canopy temperature distribution when artificial
    references are unavailable (a pragmatic fallback; physical references are preferred)."""
    t = np.asarray(temp_c, float)[np.asarray(veg_mask, bool)]
    return float(np.nanpercentile(t, low_pct)), float(np.nanpercentile(t, high_pct))


def canopy_air_difference(tc, t_air):
    return np.asarray(tc, float) - float(t_air)
