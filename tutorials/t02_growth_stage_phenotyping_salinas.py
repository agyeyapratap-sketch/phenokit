"""Tutorial 02 — Phenotypic insight from hyperspectral data: lettuce growth stage (AVIRIS Salinas).

The Salinas scene contains romaine lettuce fields at 4, 5, 6 and 7 weeks after planting.
We use them to show how to move from *classification* to *phenotyping*:

1. Index trajectories (NDVI, NDRE, REIP, MCARI, WBI) across growth stages — which indices
   saturate and which keep responding?
2. A PLSR model of crop age from spectra with VIP-based wavelength importance.
3. An honest validation: each age is one field, so age is confounded with field. Random
   pixel CV is optimistic; leave-one-field-out on the *interior* stages (5 and 6 weeks) is a
   genuine interpolation test.

Run:  python tutorials/t02_growth_stage_phenotyping_salinas.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import DATA, OKABE, banner, plt, save_results, savefig
from sklearn.cross_decomposition import PLSRegression

from phenokit import calibration, datasets, indices, io as pio, ml, qc, spectral

LETTUCE = {11: 4, 12: 5, 13: 6, 14: 7}  # Salinas class id → weeks after planting


def approx_wavelengths():
    """Approximate AVIRIS centres for the 204-band corrected Salinas cube (bands 108–112, 154–167, 224 removed)."""
    wl = np.linspace(400, 2500, 224)
    return np.delete(wl, list(range(107, 112)) + list(range(153, 167)) + [223])


def main():
    banner("1. Fetch, load, QC")
    root = datasets.fetch("salinas", DATA)
    cube = pio.read_mat_cube(root / "Salinas_corrected.mat").astype(np.float32)
    gt = pio.read_mat_cube(root / "Salinas_gt.mat").astype(int)
    wl = approx_wavelengths()
    bad = qc.detect_bad_bands(cube, wl, snr_min=15)
    keep = np.setdiff1d(np.arange(len(wl)), bad["bad_bands"])
    print(f"cube {cube.shape}; flagged {len(bad['bad_bands'])} bands; keeping {len(keep)}")
    # Salinas is at-sensor radiance (no panels were imaged). Indices need reflectance, so convert to
    # *relative* reflectance by flat-fielding on bare smooth fallow soil (class 5), assumed spectrally flat
    # at 0.25. Without this step NDVI is negative for lettuce because solar irradiance falls in the NIR.
    raw_ndvi = indices.compute("NDVI", cube[gt == 14], wl, tol_nm=12).mean()
    cube = calibration.flat_field(cube[..., keep], gt == 5, 0.25).astype(np.float32)
    wl = wl[keep]
    keep = np.arange(len(wl))
    print(f"7-wk lettuce mean NDVI: radiance {raw_ndvi:.3f} → flat-field relative reflectance "
          f"{indices.compute('NDVI', cube[gt == 14], wl, tol_nm=12).mean():.3f}")
    m = np.isin(gt, list(LETTUCE))
    X = cube[m][:, keep]
    rows, cols = np.nonzero(m)
    age = np.vectorize(LETTUCE.get)(gt[m]).astype(float)
    wlk = wl[keep]
    print(f"lettuce pixels: {dict(zip(*np.unique(age, return_counts=True)))}")

    banner("2. Index trajectories across growth stages")
    Xr = X
    idx = {n: indices.compute(n, Xr, wlk, tol_nm=12) for n in ["NDVI", "NDRE", "CIre", "REIP", "MCARI", "WBI"]}
    traj = pd.DataFrame({"age_wk": age, **idx}).groupby("age_wk").agg(["mean", "std"])  # NaN-aware
    print(traj.round(3).to_string())
    # sensitivity = between-stage range relative to within-stage SD (higher = more discriminating)
    sens = {n: float((traj[n]["mean"].max() - traj[n]["mean"].min()) / traj[n]["std"].mean()) for n in idx}
    print("stage sensitivity (range of means / mean within-stage SD):", {k: round(v, 2) for k, v in sens.items()})

    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    for a, n in zip(axs, ["NDVI", "NDRE", "REIP", "WBI"]):
        a.errorbar(traj.index, traj[n]["mean"], yerr=traj[n]["std"], marker="o", color=OKABE[4], capsize=3)
        a.set_title(f"{n} (sensitivity {sens[n]:.1f})")
        a.set_xlabel("Weeks after planting")
        a.set_xticks([4, 5, 6, 7])
    savefig(fig, "t02_index_trajectories.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for k, a_ in enumerate(sorted(LETTUCE.values())):
        ax.plot(wlk, Xr[age == a_].mean(0), color=OKABE[k], label=f"{a_} wk", lw=1.2)
    ax.set_xlabel("Wavelength (nm, approx.)")
    ax.set_ylabel("Relative reflectance (flat field)")
    ax.legend(frameon=False)
    savefig(fig, "t02_lettuce_spectra.png")

    banner("3. PLSR model of crop age")
    Xd = spectral.savgol(spectral.snv(X), 7, 2, deriv=1)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xd), min(6000, len(Xd)), replace=False)
    sel = ml.select_pls_components(Xd[sub], age[sub], 15)
    k = sel["n_components"]
    print(f"PLS components (1-SE rule): {k}")
    model = ml.default_regressor("plsr", n_components=k)
    st_rand, _ = ml.cross_validate_regression(model, Xd[sub], age[sub], scheme="kfold")
    groups = ml.spatial_blocks(rows[sub], cols[sub], 20)
    st_block, _ = ml.cross_validate_regression(model, Xd[sub], age[sub], groups, scheme="group")
    print("random pixel CV   ", {k_: round(v, 3) for k_, v in st_rand.items() if k_ in ("r2", "rmse")})
    print("spatial-block CV  ", {k_: round(v, 3) for k_, v in st_block.items() if k_ in ("r2", "rmse")})
    # leave-one-field-out on interior stages: train on the other three ages, predict this one
    lofo = []
    for a_ in (5.0, 6.0):
        tr, te = age[sub] != a_, age[sub] == a_
        mdl = ml.default_regressor("plsr", n_components=k).fit(Xd[sub][tr], age[sub][tr])
        p = np.ravel(mdl.predict(Xd[sub][te]))
        lofo.append({"held_out_age": a_, "pred_mean": float(p.mean()), "pred_sd": float(p.std()),
                     "bias": float(p.mean() - a_)})
    lofo = pd.DataFrame(lofo)
    print("leave-one-field-out (interior stages):\n", lofo.round(2).to_string(index=False))

    pls = PLSRegression(k).fit((Xd[sub] - Xd[sub].mean(0)) / Xd[sub].std(0), age[sub])
    vip = ml.vip_scores(pls)
    top = wlk[np.argsort(vip)[::-1][:10]]
    print("top-10 VIP wavelengths (nm, approx.):", np.round(np.sort(top)).astype(int))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(wlk, vip, color=OKABE[5], lw=1)
    ax[0].axhline(1, ls="--", color="grey", lw=0.8)
    ax[0].set_xlabel("Wavelength (nm, approx.)")
    ax[0].set_ylabel("VIP")
    ax[0].set_title("(a) PLSR variable importance")
    for a_, c_ in zip(lofo.held_out_age, [OKABE[1], OKABE[2]]):
        tr, te = age[sub] != a_, age[sub] == a_
        mdl = ml.default_regressor("plsr", n_components=k).fit(Xd[sub][tr], age[sub][tr])
        ax[1].hist(np.ravel(mdl.predict(Xd[sub][te])), bins=40, alpha=0.6, color=c_, label=f"held-out {a_:.0f} wk")
        ax[1].axvline(a_, color=c_, ls="--")
    ax[1].set_xlabel("Predicted age (weeks)")
    ax[1].set_title("(b) Leave-one-field-out predictions")
    ax[1].legend(frameon=False)
    savefig(fig, "t02_vip_lofo.png")

    save_results("t02_salinas_lettuce", {"n_pixels": int(len(X)), "bands_kept": int(len(keep)),
                                         "raw_radiance_ndvi_7wk": float(raw_ndvi), "index_sensitivity": sens, "pls_components": k,
                                         "random_cv": st_rand, "spatial_block_cv": st_block,
                                         "leave_one_field_out": lofo, "top_vip_nm": np.sort(top).tolist(),
                                         "trajectories": traj.round(4).reset_index().to_numpy().tolist()})


if __name__ == "__main__":
    main()
