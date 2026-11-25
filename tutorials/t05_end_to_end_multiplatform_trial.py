"""Tutorial 05 — End-to-end multi-platform trial: acquisition → QC → calibration → traits → validation.

Uses the synthetic five-crop trial (wheat, barley, canola, lupin, field pea; 12 genotypes × 2
reps) observed by UAV (RGB, 5-band multispectral, VNIR hyperspectral, thermal, LiDAR), a
ground vehicle (active NDVI + RTK-GNSS) and a handheld spectroradiometer. Eleven realistic
faults are injected; the pipeline must find them all. Because the truth is known we can also
measure how well each trait is recovered and how models transfer across species.

Run:  python tutorials/t05_end_to_end_multiplatform_trial.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import OKABE, OUT, ROOT, banner, plt, save_results, savefig

from phenokit import catalog, indices, io as pio, pipeline, validation
from phenokit.synthetic import TrialConfig, generate_trial

FAULT_TO_CHECK = {  # injected fault id → QC check(s) that should catch it
    "hs_bad_band": ["bad_bands"], "rgb_blur": ["sharpness_laplacian_var"], "rgb_overexposed": ["overexposed_fraction"],
    "rgb_duplicate": ["near_duplicates"], "thermal_vignetting": ["radial_nonuniformity_c"],
    "lidar_gap": ["low_density_cell_fraction"], "lidar_outliers": ["isolated_points"],
    "gnss_float": ["rtk_fixed_fraction"], "gnss_gap": ["max_time_gap_s"], "handheld_splice": ["splice_step_1000nm"],
    "groundtruth_unit_error": ["biomass_meas_t_ha_outliers"], "protocol_violation": ["wind_ms", "panel_after"],
    "file_corruption": ["manifest_modified_files"],
}


def main():
    banner("1. Generate the synthetic trial (standard layout, metadata, SHA-256 manifest)")
    data_root = ROOT / "data" / "synthetic"
    trial = generate_trial(data_root, TrialConfig(), verbose=False)
    print(f"trial at {trial['root'].relative_to(ROOT)}; {len(trial['faults'])} faults injected")

    banner("2. Run the pipeline")
    cfg = pipeline.load_config(ROOT / "configs" / "synthetic_trial.yaml")
    out = OUT / "synthetic_run"
    res = pipeline.run(cfg, out)

    banner("3. Did QC catch every injected fault?")
    qcdf = res["qc_report"].to_frame()
    flagged = qcdf[qcdf.status != "PASS"]
    rows = []
    for f in trial["faults"]:
        hits = flagged[flagged.check.isin(FAULT_TO_CHECK[f["id"]])]
        rows.append({"fault": f["id"], "where": f["where"], "detected": not hits.empty,
                     "status": "/".join(sorted(set(hits.status))) or "-", "by": ", ".join(sorted(set(hits.check)))})
    det = pd.DataFrame(rows)
    print(det[["fault", "detected", "status", "by"]].to_string(index=False))
    false_alarms = flagged[~flagged.check.isin(sum(FAULT_TO_CHECK.values(), []) + ["flat_field_applied",
                                                                                     "used_corrupted_file"])]
    print(f"detected {det.detected.sum()}/{len(det)}; other non-PASS results (potential false alarms): {len(false_alarms)}")
    if len(false_alarms):
        print(false_alarms[["target", "check", "status", "value"]].to_string(index=False))

    banner("4. Trait recovery and validation")
    D = res["features"]
    truth = pd.read_csv(trial["paths"]["truth"])[["plot_id", "stress", "fc", "cab", "lai"]]
    D = D.merge(truth, on="plot_id")
    hv = res["height_validation"]
    print("LiDAR p99 height vs ruler height:\n", hv[["species", "n", "r2", "rmse", "bias", "ccc"]].round(3).to_string(index=False))
    cw = validation.agreement(D.stress, D.th_cwsi)
    cv = validation.agreement(D.fc, D.rgb_canopy_cover_mean)
    print(f"CWSI vs simulated stress: r={cw['r']:.3f}; RGB canopy cover vs true fc: r={cv['r']:.3f}, bias={cv['bias']:+.3f}")
    reip = validation.agreement(D.cab, D.hs_REIP_mean)
    print(f"HS REIP vs true chlorophyll: r={reip['r']:.3f}")

    bm = res["models"]["biomass_rf"]
    nm = res["models"]["nitrogen_plsr"]
    model_tab = pd.DataFrame([
        {"trait": "biomass (RF, structural+spectral)", "validation": "random 5-fold", **bm["random_kfold"]},
        {"trait": "biomass (RF, structural+spectral)", "validation": "leave-one-species-out", **bm["leave_one_species_out"]},
        {"trait": f"N % (PLSR, {nm['n_components']} LV, d1 spectra)", "validation": "random 5-fold", **nm["random_kfold"]},
        {"trait": f"N % (PLSR, {nm['n_components']} LV, d1 spectra)", "validation": "leave-one-species-out",
         **nm["leave_one_species_out"]}])[["trait", "validation", "n", "r2", "rmse", "nrmse_pct", "bias", "ccc"]]
    print(model_tab.round(3).to_string(index=False))
    T = res["models"]["nitrogen_transfer_r2"]
    print("N % PLSR transfer R² (train → test):\n", T.round(2).to_string())
    xp = res["cross_platform_ndvi"]
    print("NDVI cross-platform vs handheld:\n", xp.round(3).to_string(index=False))
    H = res["heritability"].pivot(index="trait", columns="species", values="H2")
    print("Broad-sense heritability (entry-mean):\n", H.round(2).to_string())

    banner("5. Catalogue & provenance")
    with catalog.DataCatalog(out / "catalog.sqlite") as cat:
        print(cat.query("SELECT platform, sensor, level, COUNT(f.id) AS n_files FROM datasets d "
                        "JOIN files f ON f.dataset_id = d.id GROUP BY d.id").to_string(index=False))
        print(cat.query("SELECT status, COUNT(*) AS n FROM qc_results GROUP BY status").to_string(index=False))
        print(cat.query("SELECT step, run_utc FROM provenance").to_string(index=False))

    banner("6. Figures")
    d = trial["root"] / "20260820"
    rgb = pio.read_image(d / "uav_rgb/raw/rgb_ortho.png")
    refl, meta = pio.read_envi(out / "hs_reflectance.hdr")
    ndre = indices.compute("NDRE", refl, meta["wavelength"])
    th, _ = pio.read_envi(d / "uav_thermal/raw/thermal_brightness_c.hdr")
    fig, ax = plt.subplots(1, 4, figsize=(13, 5.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1.3]})
    ax[0].imshow(rgb[::4, ::4])
    ax[0].set_title("(a) RGB orthomosaic")
    im = ax[1].imshow(ndre, cmap="viridis", vmin=0, vmax=0.5)
    ax[1].set_title("(b) NDRE (calibrated VNIR)")
    plt.colorbar(im, ax=ax[1], fraction=0.05)
    im = ax[2].imshow(th[..., 0], cmap="inferno")
    ax[2].set_title("(c) Thermal (raw, vignetted)")
    plt.colorbar(im, ax=ax[2], fraction=0.05)
    sp = D.groupby("species")
    for k, (name, g) in enumerate(sp):
        ax[3].scatter(g.height_meas_m, g.lidar_height_p99, s=12, color=OKABE[k], label=name)
    lim = [0.25, 1.6]
    ax[3].plot(lim, lim, "k--", lw=0.7)
    ax[3].set_xlabel("Ruler height (m)")
    ax[3].set_ylabel("LiDAR p99 height (m)")
    ax[3].set_title("(d) Height validation")
    ax[3].legend(frameon=False, fontsize=7)
    ax[3].set_aspect("equal")
    for a in ax[:3]:
        a.set_xticks([])
        a.set_yticks([])
    savefig(fig, "t05_trial_maps.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for k, (name, g) in enumerate(D.groupby("species")):
        ax[0].scatter(g.biomass_meas_t_ha, g.biomass_pred_random_kfold, s=10, color=OKABE[k], label=name)
        ax[1].scatter(g.biomass_meas_t_ha, g.biomass_pred_leave_one_species_out, s=10, color=OKABE[k])
    for a, t in zip(ax[:2], ["(a) Biomass, random 5-fold", "(b) Biomass, leave-one-species-out"]):
        lim = [0, np.nanmax(D.biomass_meas_t_ha) * 1.05]
        a.plot(lim, lim, "k--", lw=0.7)
        a.set_xlabel("Measured biomass (t/ha)")
        a.set_ylabel("Predicted (t/ha)")
        a.set_title(t)
    ax[0].legend(frameon=False, fontsize=7)
    im = ax[2].imshow(T.to_numpy(float).clip(-1, 1), cmap="RdBu", vmin=-1, vmax=1)
    ax[2].set_xticks(range(len(T.columns)), T.columns, rotation=45, ha="right")
    ax[2].set_yticks(range(len(T.index)), T.index)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax[2].text(j, i, f"{T.iloc[i, j]:.2f}" if T.iloc[i, j] > -1 else "<-1", ha="center", va="center", fontsize=7)
    ax[2].set_xlabel("Test species")
    ax[2].set_ylabel("Training species")
    ax[2].set_title("(c) N % PLSR transfer R²")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    savefig(fig, "t05_models.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    ref = D.hh_NDVI
    for k, (lab, col) in enumerate([("ground vehicle", "gv_ndvi"), ("UAV multispectral", "ms_NDVI_mean"),
                                    ("UAV hyperspectral", "hs_NDVI_mean")]):
        ba = validation.bland_altman(D[col], ref)
        ax[0].scatter(ba["mean"], ba["diff"], s=8, color=OKABE[k + 4], label=f"{lab} (bias {ba['mean_diff']:+.3f})")
        ax[0].axhline(ba["mean_diff"], color=OKABE[k + 4], lw=0.8)
        for lo in (ba["loa_low"], ba["loa_high"]):
            ax[0].axhline(lo, color=OKABE[k + 4], lw=0.6, ls=":")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("Mean NDVI (platform, handheld)")
    ax[0].set_ylabel("Platform − handheld")
    ax[0].set_title("(a) Cross-platform NDVI (Bland–Altman)")
    ax[0].legend(frameon=False, fontsize=7)
    traits = ["biomass_meas_t_ha", "height_meas_m", "lidar_height_p99", "spad", "hs_REIP_mean", "n_meas_pct",
              "ms_NDRE_mean", "ms_NDVI_mean", "rgb_canopy_cover_mean", "th_canopy_temp_c"]
    Hs = H.reindex([t for t in traits if t in H.index])
    im = ax[1].imshow(Hs.to_numpy(float), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax[1].set_yticks(range(len(Hs)), Hs.index, fontsize=7)
    ax[1].set_xticks(range(len(Hs.columns)), Hs.columns, rotation=30, ha="right")
    for i in range(Hs.shape[0]):
        for j in range(Hs.shape[1]):
            ax[1].text(j, i, f"{Hs.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax[1].set_title("(b) Broad-sense heritability H²")
    savefig(fig, "t05_platforms_heritability.png")

    save_results("t05_multiplatform", {"fault_detection": det, "false_alarms": false_alarms[["target", "check", "status"]],
                                       "height_validation": hv, "cwsi_vs_stress": cw, "cover_vs_fc": cv,
                                       "reip_vs_cab": reip, "models": model_tab, "n_transfer_r2": T.reset_index(),
                                       "cross_platform_ndvi": xp, "heritability": H.reset_index(),
                                       "hs_validation_panel_mae": res["qc"]["hs_validation_panel_mae"],
                                       "hs_bad_bands_nm": res["qc"]["hs_bad_bands_nm"],
                                       "thermal_flat_field_c": res.get("thermal_flat_field_c"),
                                       "lidar_removed_points": res.get("lidar_removed_points"),
                                       "ground_records_dropped_non_rtk": res.get("ground_records_dropped_non_rtk")})


if __name__ == "__main__":
    main()
