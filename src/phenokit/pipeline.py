"""Config-driven, multi-platform trial pipeline: integrity → QC → calibration → traits → validation.

The pipeline works on the standard layout (see :func:`phenokit.catalog.dataset_path`) and a
YAML config that names the files for each platform/sensor. Each stage writes provenance to
the SQLite catalogue and QC results to an HTML/CSV report, and *gates* downstream use:
data that FAIL a check are excluded or repaired, never silently used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import acquisition, calibration, catalog, indices, io as pio, pointcloud, qc, spectral, thermal, validation
from .ml import cross_validate_regression, default_regressor, select_pls_components, transfer_matrix
from .plots import PlotLayout, plot_mean_spectra, zonal_stats
from .segmentation import clean_mask, segment_vegetation_rgb, segment_vegetation_spectral


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_base"] = str(Path(path).resolve().parent)
    return cfg


def _p(cfg, *parts) -> Path:
    root = Path(cfg["trial_root"])
    if not root.is_absolute():
        root = Path(cfg["_base"]) / root
    return root.joinpath(*parts)


def run(cfg: dict, out_dir: str | Path, verbose: bool = True) -> dict:
    log = (lambda *a: print("[pipeline]", *a)) if verbose else (lambda *a: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    date = cfg["date"].replace("-", "")
    rep = qc.QCReport(f"QC report — {cfg['trial_id']} {cfg['date']}")
    cat = catalog.DataCatalog(out / "catalog.sqlite")
    results: dict = {"qc": {}, "models": {}}
    layout = PlotLayout.from_csv(_p(cfg, cfg["plot_layout"]))
    sensors = cfg["sensors"]

    # ------------------------------------------------------------------ 1. integrity & metadata
    log("1/9 integrity: verifying SHA-256 manifest")
    ver = catalog.verify_manifest(_p(cfg))
    rep.extend([qc.QCResult("trial", "manifest_missing_files", qc.FAIL if ver["missing"] else qc.PASS,
                            len(ver["missing"]), "0", "; ".join(ver["missing"])),
                qc.QCResult("trial", "manifest_modified_files", qc.FAIL if ver["modified"] else qc.PASS,
                            len(ver["modified"]), "0", "; ".join(ver["modified"]))])
    corrupted = {str(_p(cfg, m)) for m in ver["modified"]}
    results["qc"]["manifest"] = ver
    for md in sorted(_p(cfg, date).rglob("metadata.json")):
        errs = catalog.AcquisitionMetadata.load(md).validate()
        rep.extend([qc.QCResult(md.parent.parent.name, "metadata_complete", qc.FAIL if errs else qc.PASS,
                                len(errs), "0", "; ".join(errs))])
    acq_log = _p(cfg, "acquisition_log.json")
    if acq_log.exists():
        for proto, rec in json.loads(acq_log.read_text()).items():
            for issue in acquisition.validate_acquisition(rec, proto):
                rep.extend([qc.QCResult(f"protocol:{proto}", issue["field"], issue["severity"], None, "",
                                        issue["message"])])
    for platform_sensor in sorted({f"{s['platform']}_{s['sensor']}" for s in sensors.values()}):
        d = _p(cfg, date, platform_sensor, "raw")
        if d.exists():
            pl, se = platform_sensor.split("_", 1)
            cat.register_dataset(d, cfg["trial_id"], cfg["date"], pl, se, "raw",
                                 json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {})
    cat.record_step("integrity_check", [_p(cfg, catalog.MANIFEST_NAME)], [], {"result": ver["ok"]})

    def sensor_dir(key):
        s = sensors[key]
        return _p(cfg, date, f"{s['platform']}_{s['sensor']}", "raw")

    feats = [layout.table[["plot_id"]].copy()]

    # ------------------------------------------------------------------ 2. hyperspectral
    hs_cfg = sensors.get("hyperspectral")
    veg_mask_hs = None
    if hs_cfg:
        log("2/9 hyperspectral: QC, empirical-line calibration, indices, plot spectra")
        d = sensor_dir("hyperspectral")
        dn, meta = pio.read_envi(d / hs_cfg["file"])
        wl, tf = np.array(meta["wavelength"]), meta["geotransform"]
        bad = qc.detect_bad_bands(dn, wl, snr_min=hs_cfg.get("snr_min", 20), sat_value=hs_cfg.get("saturation", 4095))
        rep.extend([qc.QCResult("hyperspectral", "bad_bands", qc.WARN if bad["bad_bands"] else qc.PASS,
                                len(bad["bad_bands"]), f"SNR<{hs_cfg.get('snr_min', 20)}",
                                "; ".join(f"{wl[b]:.0f}nm: {', '.join(r)}" for b, r in bad["reasons"].items()))])
        panels = pd.read_csv(d / hs_cfg["panels"])
        pl = PlotLayout(panels.rename(columns={"panel": "plot_id"}))
        win = pl.windows(tf, buffer_m=hs_cfg.get("panel_buffer_m", 0.25), shape=dn.shape[:2])
        masks = []
        for (r0, r1, c0, c1) in win.values():
            m = np.zeros(dn.shape[:2], bool)
            m[r0:r1, c0:c1] = True
            masks.append(m)
        refl_vals = panels.reflectance.tolist()
        # calibrate with outer panels, validate on the middle one (independent check)
        val_i = len(masks) // 2
        fit_i = [i for i in range(len(masks)) if i != val_i]
        el_val = calibration.fit_empirical_line(dn, [masks[i] for i in fit_i], [refl_vals[i] for i in fit_i])
        val_refl = el_val.apply(dn[masks[val_i]]).mean(0)
        good = np.setdiff1d(np.arange(len(wl)), bad["bad_bands"])
        rep.extend(qc.qc_panel(val_refl[good], refl_vals[val_i], target="hyperspectral_validation_panel"))
        el = calibration.fit_empirical_line(dn, masks, refl_vals)
        refl = el.apply(dn).astype(np.float32)
        rep.extend(qc.qc_reflectance(refl[..., good], target="hyperspectral_reflectance"))
        results["qc"]["hs_bad_bands_nm"] = [float(wl[b]) for b in bad["bad_bands"]]
        results["qc"]["hs_validation_panel_mae"] = float(np.mean(np.abs(val_refl[good] - refl_vals[val_i])))
        wl_g, refl_g = wl[good], refl[..., good]
        pio.write_envi(out / "hs_reflectance", refl_g, wl_g, transform=tf, description="phenokit calibrated reflectance")
        cat.record_step("empirical_line", [d / hs_cfg["file"]], [out / "hs_reflectance.hdr"],
                        {"panels": refl_vals, "dropped_bands_nm": results["qc"]["hs_bad_bands_nm"]})
        ndvi = indices.compute("NDVI", refl_g, wl_g)
        veg_mask_hs = clean_mask(segment_vegetation_spectral(ndvi), min_size=0, opening=0, fill_holes=False)
        idx = indices.compute_all(refl_g, wl_g, ["NDVI", "NDRE", "CIre", "MCARI", "PRI", "WBI", "REIP", "MTCI"])
        stack = np.stack(list(idx.values()), -1)
        feats.append(zonal_stats(stack, layout, tf, veg_mask_hs, ("mean",), buffer_m=0.2, prefix="hs_",
                                 band_names=[f"hs_{k}" for k in idx]))
        ids, spec = plot_mean_spectra(refl_g, layout, tf, veg_mask_hs, buffer_m=0.2)
        results["hs_plot_spectra"] = {"plot_id": ids, "wavelength": wl_g, "spectra": spec}
        np.savez_compressed(out / "hs_plot_spectra.npz", plot_id=ids, wavelength=wl_g, spectra=spec)
        results["hs_transform"] = tf

    # ------------------------------------------------------------------ 3. multispectral
    ms_cfg = sensors.get("multispectral")
    if ms_cfg:
        log("3/9 multispectral: indices per plot")
        d = sensor_dir("multispectral")
        ms, meta = pio.read_envi(d / ms_cfg["file"])
        wl, fw, tf = meta["wavelength"], meta.get("fwhm"), meta["geotransform"]
        rep.extend(qc.qc_reflectance(ms, target="multispectral_reflectance"))
        idx = indices.compute_all(ms, wl, ["NDVI", "GNDVI", "NDRE", "CIre", "SAVI"], fwhm=fw)
        stack = np.stack(list(idx.values()), -1)
        feats.append(zonal_stats(stack, layout, tf, None, ("mean",), buffer_m=0.2, prefix="ms_",
                                 band_names=[f"ms_{k}" for k in idx]))
        feats.append(zonal_stats(ms, layout, tf, None, ("mean",), buffer_m=0.2, prefix="ms_",
                                 band_names=[f"ms_{int(w)}nm" for w in wl]).drop(columns=["ms_n_px", "ms_valid_frac"]))

    # ------------------------------------------------------------------ 4. RGB
    rgb_cfg = sensors.get("rgb")
    if rgb_cfg:
        log("4/9 RGB: frame QC, duplicate detection, vegetation segmentation, canopy cover")
        d = sensor_dir("rgb")
        hashes = {}
        for fr in sorted((d / rgb_cfg.get("frames_dir", "frames")).glob("*.png")):
            img = pio.read_image(fr)
            rep.extend([r for r in qc.qc_rgb(img, target=f"rgb:{fr.name}") if r.check != "noise_sigma"])
            hashes[fr.name] = qc.average_hash(img)
        dups = qc.find_near_duplicates(hashes, max_hamming=2)
        rep.extend([qc.QCResult("rgb_frames", "near_duplicates", qc.WARN if dups else qc.PASS, len(dups), "0",
                                "; ".join(f"{a}~{b}" for a, b, _ in dups))])
        ortho = pio.read_image(d / rgb_cfg["file"])
        rtf = pio.read_worldfile(d / rgb_cfg["file"])
        veg = segment_vegetation_rgb(ortho, "exg_otsu", min_size=4, opening=0)
        z = zonal_stats(veg.astype(float), layout, rtf, None, ("mean",), buffer_m=0.1, prefix="rgb_",
                        band_names=["rgb_canopy_cover"])
        feats.append(z[["plot_id", "rgb_canopy_cover_mean"]])
        results["rgb_veg_fraction"] = float(veg.mean())

    # ------------------------------------------------------------------ 5. thermal
    th_cfg = sensors.get("thermal")
    if th_cfg:
        log("5/9 thermal: QC, emissivity correction, canopy temperature, CWSI")
        d = sensor_dir("thermal")
        f = d / th_cfg["file"]
        is_corrupt = str(f.with_suffix(".raw")) in corrupted
        tb, meta = pio.read_envi(f)
        tb = tb[..., 0]
        cov = ndvi if hs_cfg and ndvi.shape == tb.shape else None
        tq = qc.qc_thermal(tb, target="thermal", covariate=cov)
        rep.extend(tq)
        vign = next(r for r in tq if r.check == "radial_nonuniformity_c")
        if vign.status != qc.PASS and th_cfg.get("flat_field_correction", True):
            c, r2map = qc.radial_trend(tb, cov)
            tb = tb - c * r2map
            rep.extend([qc.QCResult("thermal", "flat_field_applied", qc.WARN, c, "",
                                    f"subtracted fitted radial trend ({c:.2f} °C at edge); residual "
                                    f"{qc.radial_trend(tb, cov)[0]:.2f} °C")])
            results["thermal_flat_field_c"] = c
        refs = json.loads((d / th_cfg["references"]).read_text())
        ts = calibration.surface_temperature(tb, emissivity=th_cfg.get("emissivity", 0.98),
                                             reflected_temp_c=refs.get("air_temp_c"))
        tf = meta["geotransform"]
        if veg_mask_hs is not None and veg_mask_hs.shape == ts.shape:
            vmask = veg_mask_hs
        else:
            vmask = ts < np.nanpercentile(ts, 40)
        (a0, a1, b0, b1), (c0, c1, e0, e1) = refs["wet_ref_px"], refs["dry_ref_px"]
        t_wet, t_dry = float(np.median(ts[a0:a1, b0:b1])), float(np.median(ts[c0:c1, e0:e1]))
        rows = []
        for pid, (r0, r1, cc0, cc1) in layout.windows(tf, 0.2, ts.shape).items():
            tc = thermal.canopy_temperature(ts[r0:r1, cc0:cc1], vmask[r0:r1, cc0:cc1],
                                            tuple(th_cfg.get("canopy_percentiles", (5, 40))))
            rows.append({"plot_id": pid, "th_canopy_temp_c": tc,
                         "th_cwsi": float(thermal.cwsi(tc, t_wet, t_dry)),
                         "th_tc_minus_air": tc - refs["air_temp_c"]})
        tdf = pd.DataFrame(rows)
        if is_corrupt:
            rep.extend([qc.QCResult("thermal", "used_corrupted_file", qc.FAIL, None, "",
                                    "checksum mismatch — thermal traits flagged, excluded from models")])
            tdf["th_flag_corrupt"] = True
        feats.append(tdf)
        results["thermal_refs"] = {"wet": t_wet, "dry": t_dry}

    # ------------------------------------------------------------------ 6. LiDAR
    li_cfg = sensors.get("lidar")
    if li_cfg:
        log("6/9 LiDAR: QC, outlier removal, ground normalisation, plot height metrics")
        d = sensor_dir("lidar")
        las = pio.read_las(d / li_cfg["file"])
        xyz = las["xyz"]
        rep.extend(qc.qc_pointcloud(xyz, target="lidar", min_density=li_cfg.get("min_density", 100)))
        keep = pointcloud.statistical_outlier_removal(xyz, k=8, std_ratio=3.0)
        keep &= xyz[:, 2] < np.percentile(xyz[keep, 2], 99.9) + 2.0
        xyz = xyz[keep]
        h = pointcloud.normalise_heights(xyz, res=li_cfg.get("dtm_res", 1.0), percentile=1.0)
        lm = pointcloud.plot_height_metrics(xyz, h, layout, buffer_m=0.15)
        lm.columns = ["plot_id"] + [f"lidar_{c}" for c in lm.columns[1:]]
        feats.append(lm)
        results["lidar_removed_points"] = int((~keep).sum())
        cat.record_step("lidar_height", [d / li_cfg["file"]], [], {"dtm_res": li_cfg.get("dtm_res", 1.0)})

    # ------------------------------------------------------------------ 7. ground vehicle
    gv_cfg = sensors.get("ground")
    if gv_cfg:
        log("7/9 ground vehicle: GNSS QC, plot assignment, NDVI")
        d = sensor_dir("ground")
        gv = pd.read_csv(d / gv_cfg["file"])
        rep.extend(qc.qc_gnss_track(gv.t_s, gv[["easting", "northing"]].to_numpy(), gv.fix.to_numpy(),
                                    target="ground_gnss", max_gap_s=0.5, max_speed_ms=1.5))
        ok = gv.fix == "RTK_FIXED"
        k = layout.assign_points(gv.loc[ok, ["easting", "northing"]].to_numpy(), buffer_m=0.3)
        g = gv.loc[ok].assign(k=k).query("k >= 0")
        gmean = g.groupby("k").ndvi.agg(["mean", "count"])
        gmean.index = layout.table.plot_id.iloc[gmean.index].values
        feats.append(gmean.rename(columns={"mean": "gv_ndvi", "count": "gv_n"}).rename_axis("plot_id").reset_index())
        results["ground_records_dropped_non_rtk"] = int((~ok).sum())

    # ------------------------------------------------------------------ 8. handheld
    hh_cfg = sensors.get("handheld")
    if hh_cfg:
        log("8/9 handheld: splice QC/correction, plot spectra and indices")
        d = sensor_dir("handheld")
        wl, spectra, names = pio.read_spectrum_csv(d / hh_cfg["file"])
        pids = [n.rsplit("_s", 1)[0] for n in names]
        for pid in sorted(set(pids)):
            sel = np.array([p == pid for p in pids])
            res = qc.qc_spectrum_splice(wl, spectra[sel], target=f"handheld:{pid}")
            rep.extend([r for r in res if r.status != qc.PASS])
        corr = spectral.splice_correction(wl, spectra)
        wl_v, corr_v, _ = spectral.remove_ranges(wl, corr)
        dfh = pd.DataFrame(corr_v, columns=wl_v)
        dfh["plot_id"] = pids
        mean = dfh.groupby("plot_id").mean()
        hh = pd.DataFrame({"plot_id": mean.index})
        for n in ("NDVI", "NDRE", "PRI", "WBI", "REIP"):
            hh[f"hh_{n}"] = indices.compute(n, mean.to_numpy(), wl_v)
        feats.append(hh)
        results["handheld_spectra"] = {"plot_id": list(mean.index), "wavelength": wl_v, "spectra": mean.to_numpy()}

    # ------------------------------------------------------------------ 9. merge, validate, model
    log("9/9 merge features, ground-truth QC, models and validation")
    F = feats[0]
    for f_ in feats[1:]:
        F = F.merge(f_, on="plot_id", how="left")
    gt = pd.read_csv(_p(cfg, date, "ground_truth", cfg["ground_truth"]))
    gt_cols = cfg.get("ground_truth_traits", ["biomass_meas_t_ha", "height_meas_m", "spad", "n_meas_pct"])
    for c in gt_cols:  # within-species robust outlier screen (species differ in scale)
        for sp, g in gt.groupby("species"):
            res = qc.qc_plot_values(g, [c], z=cfg.get("outlier_z", 6.0), target=f"ground_truth:{sp}")
            for r in res:
                if r.check.endswith("_outliers") and r.status != qc.PASS:
                    rep.extend([r])
                    bad_ids = [s.strip(" '[]") for s in r.message.replace("plots:", "").split(",")]
                    gt.loc[gt.plot_id.isin(bad_ids), c] = np.nan
    D = gt.merge(F, on="plot_id", how="left")
    D.to_csv(out / "plot_features.csv", index=False)
    results["features"] = D

    # sensor height vs ruler height
    if "lidar_height_p99" in D:
        results["height_validation"] = validation.validate_by_group(D, "height_meas_m", "lidar_height_p99", "species")

    # biomass model: structural + spectral features
    feat_cols = [c for c in D.columns if c.startswith(("lidar_height", "lidar_canopy", "ms_", "rgb_canopy", "hs_"))
                 and not c.endswith(("n_px", "valid_frac")) and D[c].notna().all()]
    y = D["biomass_meas_t_ha"].to_numpy()
    ok = np.isfinite(y)
    X = D.loc[ok, feat_cols].to_numpy()
    model = default_regressor("rf")
    bm = {}
    for scheme, groups in (("random_kfold", None), ("leave_one_species_out", D.loc[ok, "species"].to_numpy())):
        st, pred = cross_validate_regression(model, X, y[ok], groups, "kfold" if groups is None else "logo")
        bm[scheme] = st
        D.loc[ok, f"biomass_pred_{scheme}"] = pred
    results["models"]["biomass_rf"] = {"features": feat_cols, **bm}
    results["models"]["biomass_by_species_kfold"] = validation.validate_by_group(
        D.loc[ok], "biomass_meas_t_ha", "biomass_pred_random_kfold", "species")
    results["models"]["biomass_by_species_loso"] = validation.validate_by_group(
        D.loc[ok], "biomass_meas_t_ha", "biomass_pred_leave_one_species_out", "species")

    # nitrogen from HS plot spectra (PLSR, first-derivative spectra)
    if "hs_plot_spectra" in results:
        S = results["hs_plot_spectra"]
        spec = pd.DataFrame(S["spectra"], index=S["plot_id"]).reindex(D.plot_id).to_numpy()
        Xs = spectral.savgol(spec, 9, 2, deriv=1)
        yN = D["n_meas_pct"].to_numpy()
        okN = np.isfinite(yN) & np.isfinite(Xs).all(1)
        sel = select_pls_components(Xs[okN], yN[okN], 15)
        pls = default_regressor("plsr", n_components=sel["n_components"])
        nres = {"n_components": sel["n_components"]}
        for scheme, groups in (("random_kfold", None), ("leave_one_species_out", D.loc[okN, "species"].to_numpy())):
            st, pred = cross_validate_regression(pls, Xs[okN], yN[okN], groups, "kfold" if groups is None else "logo")
            nres[scheme] = st
            D.loc[okN, f"n_pred_{scheme}"] = pred
        results["models"]["nitrogen_plsr"] = nres
        results["models"]["nitrogen_transfer_r2"] = transfer_matrix(pls, Xs[okN], yN[okN],
                                                                    D.loc[okN, "species"].to_numpy())

    # heritability of sensor vs manual traits
    herit = []
    for trait in [c for c in ["biomass_meas_t_ha", "height_meas_m", "spad", "n_meas_pct", "lidar_height_p99",
                              "rgb_canopy_cover_mean", "ms_NDVI_mean", "ms_NDRE_mean", "hs_REIP_mean",
                              "th_canopy_temp_c", "gv_ndvi", "hh_NDVI"] if c in D]:
        for sp, g in D.groupby("species"):
            herit.append({"species": sp, **validation.heritability(g, trait, "genotype", "rep")})
    results["heritability"] = pd.DataFrame(herit)

    # cross-platform NDVI consistency
    plat = {k: v for k, v in {"handheld": "hh_NDVI", "ground_vehicle": "gv_ndvi", "uav_multispectral": "ms_NDVI_mean",
                              "uav_hyperspectral": "hs_NDVI_mean"}.items() if v in D}
    if len(plat) > 1:
        results["cross_platform_ndvi"] = validation.cross_platform_table(D.dropna(subset=list(plat.values())), plat,
                                                                         "handheld")
    results["features"] = D
    D.to_csv(out / "plot_features.csv", index=False)

    # ------------------------------------------------------------------ report
    cat.record_qc(None, rep.results)
    cat.record_step("pipeline_run", [_p(cfg)], [out], {k: v for k, v in cfg.items() if k != "_base"})
    cat.close()
    rep.to_html(out / "qc_report.html")
    results["qc_report"] = rep
    summary = {"qc": rep.summary(), "models": {k: v for k, v in results["models"].items()
                                              if isinstance(v, dict)}}
    pio.write_json(out / "results_summary.json", _jsonable(summary))
    log(f"done → {out}")
    return results


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o
