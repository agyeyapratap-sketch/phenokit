"""Validation statistics for sensor-derived traits against reference (ground-truth) measurements,
and quantitative-genetics summaries used to judge whether a phenotype is useful for breeding."""
from __future__ import annotations

import numpy as np
import pandas as pd


def agreement(y_true, y_pred) -> dict:
    """R², RMSE, nRMSE (% of mean), MAE, bias, Lin's CCC, RPD, OLS slope/intercept, Pearson r, n."""
    t, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    n = len(t)
    if n < 3:
        return {"n": n}
    err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res, ss_tot = float((err ** 2).sum()), float(((t - t.mean()) ** 2).sum())
    r = float(np.corrcoef(t, p)[0, 1])
    sxy = np.mean((t - t.mean()) * (p - p.mean()))
    ccc = float(2 * sxy / (t.var() + p.var() + (t.mean() - p.mean()) ** 2))
    slope, intercept = np.polyfit(t, p, 1)
    return {"n": n, "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "r": r, "rmse": rmse,
            "nrmse_pct": 100 * rmse / abs(t.mean()) if t.mean() != 0 else np.nan,
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()), "ccc": ccc,
            "rpd": float(t.std(ddof=1) / rmse) if rmse > 0 else np.inf, "slope": float(slope),
            "intercept": float(intercept)}


def bland_altman(a, b) -> dict:
    """Bland & Altman (1986) mean difference and 95 % limits of agreement between two methods."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    md, sd = float(np.nanmean(d)), float(np.nanstd(d, ddof=1))
    # proportional bias: slope of difference vs mean
    m = (a + b) / 2
    ok = np.isfinite(d) & np.isfinite(m)
    slope = float(np.polyfit(m[ok], d[ok], 1)[0]) if ok.sum() > 2 else np.nan
    return {"mean_diff": md, "loa_low": md - 1.96 * sd, "loa_high": md + 1.96 * sd, "sd_diff": sd,
            "proportional_bias_slope": slope, "mean": m, "diff": d}


def validate_by_group(df: pd.DataFrame, true_col: str, pred_col: str, group_col: str) -> pd.DataFrame:
    """Agreement statistics per group (e.g. crop species, platform, date) plus pooled."""
    rows = [{group_col: g, **agreement(d[true_col], d[pred_col])} for g, d in df.groupby(group_col)]
    rows.append({group_col: "ALL", **agreement(df[true_col], df[pred_col])})
    return pd.DataFrame(rows)


def heritability(df: pd.DataFrame, trait: str, genotype_col: str = "genotype", rep_col: str | None = "rep") -> dict:
    """Broad-sense heritability on an entry-mean basis from a one-way random-effects ANOVA.

    H² = σ²_G / (σ²_G + σ²_e / r). With a replicate (block) column a two-way RCBD ANOVA is used
    so block effects are removed from the error term.
    """
    d = df[[genotype_col, trait] + ([rep_col] if rep_col else [])].dropna()
    g = d.groupby(genotype_col)[trait]
    k = g.ngroups
    r = d.groupby(genotype_col).size().mean()
    grand = d[trait].mean()
    ss_g = (g.mean().sub(grand) ** 2 * g.size()).sum()
    ss_tot = ((d[trait] - grand) ** 2).sum()
    if rep_col:
        b = d.groupby(rep_col)[trait]
        ss_b = (b.mean().sub(grand) ** 2 * b.size()).sum()
        df_e = (k - 1) * (b.ngroups - 1)
        ss_e = ss_tot - ss_g - ss_b
    else:
        df_e = len(d) - k
        ss_e = ss_tot - ss_g
    ms_g, ms_e = ss_g / (k - 1), ss_e / max(df_e, 1)
    var_g = max((ms_g - ms_e) / r, 0.0)
    h2 = var_g / (var_g + ms_e / r) if var_g + ms_e > 0 else np.nan
    return {"trait": trait, "H2": float(h2), "var_g": float(var_g), "var_e": float(ms_e), "n_genotypes": int(k),
            "reps": float(r), "cv_pct": float(100 * np.sqrt(ms_e) / grand) if grand else np.nan}


def repeatability(a, b) -> float:
    """Test–retest correlation of two repeated measurements of the same plots (e.g. two flights)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def cross_platform_table(df: pd.DataFrame, value_cols: dict[str, str], reference: str) -> pd.DataFrame:
    """Compare the same trait measured on different platforms (e.g. UAV vs ground vs handheld).

    ``value_cols`` maps platform name → column. Each platform is compared against ``reference``.
    """
    rows = []
    for name, col in value_cols.items():
        if name == reference:
            continue
        a = agreement(df[value_cols[reference]], df[col])
        ba = bland_altman(df[col], df[value_cols[reference]])
        rows.append({"platform": name, "vs": reference, **{k: a.get(k) for k in ("n", "r", "ccc", "rmse", "bias")},
                     "loa_low": ba["loa_low"], "loa_high": ba["loa_high"]})
    return pd.DataFrame(rows)
