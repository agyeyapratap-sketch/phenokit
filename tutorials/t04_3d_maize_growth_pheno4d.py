"""Tutorial 04 — 3-D phenotyping from laser point clouds: maize growth through time (Pheno4D).

Pheno4D provides high-resolution laser scans of the same plants on consecutive days with
organ-level annotations on a subset. We use six annotated scans of plant Maize01 to show:

1. Remote, verified access to single files inside a 12.7 GB archive (HTTP range requests).
2. Point-cloud QC: point count, duplicates, isolated points, z range.
3. Trait extraction: plant height above the soil plane, organ (stem + leaf) instance count,
   convex-hull volume, projected canopy area.
4. Validating an *automatic* pipeline (RANSAC soil removal, no labels) against the
   annotation-based reference — the same logic you use to validate UAV LiDAR heights against
   manual measurements in the field.
5. Growth modelling: logistic fit of height through time.

Label interpretation used here (verify against the dataset documentation): the 5th column is
an instance ID with 0 = soil, 1 = the basal stem region (compact, lowest), ≥2 = leaves.

Run:  python tutorials/t04_3d_maize_growth_pheno4d.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import DATA, OKABE, banner, plt, save_results, savefig
from scipy.optimize import curve_fit

from phenokit import datasets, io as pio, pointcloud, qc, validation

SCANS = ["M01_0313_a.txt", "M01_0315_a.txt", "M01_0317_a.txt", "M01_0319_a.txt", "M01_0321_a.txt",
         "M01_0325_a.txt"]


def load(path, max_points=400_000, seed=0):
    a = pio.read_xyz(path)
    xyz = a[:, :3] / 1000.0  # mm → m
    lab = a[:, 4].astype(int) if a.shape[1] > 4 else a[:, 3].astype(int)
    if len(xyz) > max_points:  # uniform subsample keeps statistics, speeds up hulls/RANSAC
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, lab = xyz[idx], lab[idx]
    return xyz, lab, len(a)


def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def main():
    banner("1. Fetch single scans from the remote archive (range requests, SHA-256 pinned)")
    root = datasets.fetch("pheno4d", DATA, members=SCANS)

    banner("2–4. QC, traits from labels, automatic traits without labels")
    rows, clouds = [], {}
    for f in SCANS:
        xyz, lab, n_full = load(root / f)
        day = int(f.split("_")[1][2:])  # MMDD → day of month (all March)
        q = {r.check: r for r in qc.qc_pointcloud(xyz, target=f, min_density=1e5)}
        ref = pointcloud.plant_traits_3d(xyz, lab, ground_label=0, stem_label=1)
        auto = pointcloud.plant_traits_3d(xyz, None)
        rows.append({"scan": f, "day": day, "n_points": n_full, "duplicates": q["duplicate_fraction"].value,
                     "isolated_points": q["isolated_points"].value, "organ_instances": len(np.unique(lab[lab > 0])),
                     "leaf_count": ref["leaf_count"], "height_ref_m": ref["height"],
                     "height_auto_m": auto["height"], "hull_volume_l": ref["convex_hull_volume"] * 1000,
                     "projected_area_cm2": ref["projected_area_hull"] * 1e4,
                     "width_major_cm": ref["width_major"] * 100})
        clouds[day] = (xyz, lab)
        print(f"{f}: {n_full:,} pts, height ref {ref['height'] * 100:.1f} cm / auto {auto['height'] * 100:.1f} cm, "
              f"leaves {ref['leaf_count']}, hull {ref['convex_hull_volume'] * 1000:.2f} L")
    df = pd.DataFrame(rows)
    agree = validation.agreement(df.height_ref_m, df.height_auto_m)
    print(f"automatic vs annotation-based height: RMSE {agree['rmse'] * 100:.2f} cm, bias {agree['bias'] * 100:+.2f} cm, "
          f"r {agree['r']:.3f}")

    banner("5. Growth model")
    t = df.day.to_numpy(float)
    p0 = [df.height_ref_m.max() * 1.5, 0.3, t.mean()]
    try:
        (K, r, t0), cov = curve_fit(logistic, t, df.height_ref_m, p0=p0, maxfev=20000)
        growth = {"K_m": K, "r_per_day": r, "t0_day": t0, "max_growth_cm_per_day": 100 * K * r / 4}
    except RuntimeError:
        K = r = t0 = np.nan
        growth = {}
    rel = np.diff(np.log(df.height_ref_m)) / np.diff(t)
    growth["mean_relative_growth_rate_per_day"] = float(rel.mean())
    print({k: round(v, 3) for k, v in growth.items()})

    fig = plt.figure(figsize=(12, 3.6))
    for k, day in enumerate([min(clouds), max(clouds)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        xyz, lab = clouds[day]
        s = np.random.default_rng(1).choice(len(xyz), min(25000, len(xyz)), replace=False)
        cols = np.array(["#bfae95"] + OKABE * 3)[np.clip(lab[s], 0, 24)]
        ax.scatter(xyz[s, 0], xyz[s, 1], xyz[s, 2], c=cols, s=0.3, linewidths=0)
        ax.set_title(f"({'ab'[k]}) Maize01, scan date code 03{day:02d}")
        ax.set_axis_off()
        ax.view_init(elev=15, azim=-60)
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, df.height_ref_m * 100, "o", color=OKABE[4], label="height (annotation)")
    ax.plot(t, df.height_auto_m * 100, "x", color=OKABE[5], label="height (automatic)")
    if np.isfinite(K):
        tt = np.linspace(t.min() - 1, t.max() + 1, 100)
        ax.plot(tt, logistic(tt, K, r, t0) * 100, color=OKABE[4], lw=1)
    ax2 = ax.twinx()
    ax2.bar(t, df.leaf_count, color=OKABE[2], alpha=0.3, width=0.8, label="leaf instances")
    ax2.set_ylabel("Leaf instances")
    ax.set_xlabel("Day of month (file date code 03DD)")
    ax.set_ylabel("Height (cm)")
    ax.set_title("(c) Growth trajectory")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    savefig(fig, "t04_pheno4d_growth.png")
    print(df.round(3).to_string(index=False))
    save_results("t04_pheno4d", {"per_scan": df, "auto_vs_reference_height": agree, "growth": growth})


if __name__ == "__main__":
    main()
