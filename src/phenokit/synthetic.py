"""Synthetic multi-crop, multi-platform field trial with known ground truth and injected faults.

Purpose: an end-to-end, fully reproducible test-bed for acquisition → management → QC →
calibration → trait extraction → validation, where the *true* answer is known. It mimics a
dryland winter-cropping trial in a Mediterranean climate (south-west Western Australia):
wheat, barley, canola, lupin and field pea, 12 genotypes × 2 replicates each.

Produced data (standard layout ``<root>/<trial>/<date>/<platform>_<sensor>/raw``):

* UAV VNIR hyperspectral (400–1000 nm, 151 bands, 12-bit DN, 0.2 m) with 3 grey panels
* UAV 5-band multispectral reflectance (0.2 m)
* UAV RGB orthomosaic (0.025 m, PNG + world file) and six raw frames
* UAV radiometric thermal (°C brightness temperature, 0.2 m) with wet/dry references
* UAV LiDAR (LAS 1.2) ~200 pts/m²
* Ground vehicle active NDVI sensor log with GNSS track
* Handheld spectroradiometer canopy spectra 350–2500 nm (3 scans per plot)
* Ground-truth sheet (destructive biomass, height, SPAD/chlorophyll, N %, …)

The reflectance model is deliberately simple (phenomenological, not PROSAIL) but reproduces
the features that matter for pipeline testing: chlorophyll-driven red-edge shifts, LAI-driven
NIR plateau and soil mixing, water absorptions, canola flowering and pea leaf waxiness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import io as pio
from .catalog import AcquisitionMetadata, build_manifest, dataset_path
from .plots import PlotLayout
from .spectral import gaussian_srf_resample

SPECIES = {
    #             k_ext  height(m)   LAI      Cab      biomass slope  spectral quirks
    "wheat":     dict(k=0.45, h=(0.65, 0.95), lai=(2.0, 4.5), cab=(35, 60), bm=1.9, n_slope=0.065, flower=0.0, wax=0.0),
    "barley":    dict(k=0.45, h=(0.60, 0.85), lai=(1.8, 4.0), cab=(32, 55), bm=1.8, n_slope=0.062, flower=0.0, wax=0.0),
    "canola":    dict(k=0.70, h=(1.00, 1.50), lai=(2.0, 4.0), cab=(40, 65), bm=1.6, n_slope=0.075, flower=0.35, wax=0.02),
    "lupin":     dict(k=0.60, h=(0.45, 0.80), lai=(1.5, 3.5), cab=(38, 58), bm=1.5, n_slope=0.080, flower=0.05, wax=0.0),
    "field_pea": dict(k=0.55, h=(0.35, 0.60), lai=(1.5, 3.8), cab=(36, 56), bm=1.4, n_slope=0.085, flower=0.0, wax=0.05),
}

PANELS = (0.05, 0.23, 0.48)  # certified grey-panel reflectance
VNIR_WL = np.arange(400.0, 1000.1, 4.0)
FULL_WL = np.arange(350.0, 2500.1, 1.0)
MS_CENTRES, MS_FWHM = [475, 560, 668, 717, 842], [32, 27, 14, 12, 57]


def _g(wl, c, s):
    return np.exp(-0.5 * ((wl - c) / s) ** 2)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def soil_reflectance(wl, brightness=1.0, moisture=0.0):
    """Red-brown sandy/loamy soil: monotonic rise, iron-oxide shoulder, water dips when moist."""
    wl = np.asarray(wl, float)
    r = 0.08 + 0.28 * _sig((wl - 600) / 120) - 0.03 * _g(wl, 900, 60)
    r = r * (1 - 0.6 * moisture * (_g(wl, 1450, 40) + _g(wl, 1940, 50))) * brightness * (1 - 0.35 * moisture)
    return np.clip(r, 0.01, 0.8)


def leaf_canopy_reflectance(wl, cab, lai, cw=0.015, stress=0.0, flower=0.0, wax=0.0):
    """Infinite-canopy reflectance for given chlorophyll (µg cm⁻²), LAI, water (g cm⁻²) and stress."""
    wl = np.asarray(wl, float)
    rep = 700.0 + 0.38 * cab                    # red-edge shifts to longer λ with chlorophyll
    green = 0.11 * (40.0 / cab) ** 0.7 * _g(wl, 552, 32)
    vis = 0.035 + green + 0.02 * _g(wl, 470, 30) * (40 / cab)
    nir = (0.36 + 0.14 * (1 - np.exp(-0.55 * lai))) * (1 - 0.15 * stress)
    swir = nir * np.exp(-(wl - 1100) / 1800) * (1 - 0.55 * _sig((wl - 1300) / 30) * 0.4)
    plateau = np.where(wl < 1100, nir, swir)
    r = vis * (1 - _sig((wl - rep) / 9)) + plateau * _sig((wl - rep) / 9)
    water = cw / 0.015 * (0.05 * _g(wl, 970, 25) + 0.10 * _g(wl, 1200, 35) + 0.55 * _g(wl, 1450, 45)
                          + 0.75 * _g(wl, 1940, 55) + 0.25 * _g(wl, 2500, 120))
    r = r * (1 - np.clip(water, 0, 0.95))
    r = r * (1 - 0.04 * stress * _g(wl, 531, 10))          # xanthophyll (PRI) feature
    r = r + flower * 0.30 * _sig((wl - 515) / 12) * (1 - _sig((wl - 700) / 15))  # yellow petals
    r = r + wax * (0.10 - 0.08 * _sig((wl - 700) / 50))     # glaucous wax brightens VIS
    return np.clip(r, 0.005, 0.9)


def illumination(wl):
    """Relative solar irradiance at the surface: 5800 K Planck + O2-A and water vapour dips."""
    wl_m = np.asarray(wl, float) * 1e-9
    planck = 1 / (wl_m ** 5 * (np.exp(1.4388e-2 / (wl_m * 5800)) - 1))
    planck /= planck.max()
    return planck * (1 - 0.6 * _g(wl, 761, 3) - 0.25 * _g(wl, 940, 20) - 0.15 * _g(wl, 820, 15))


@dataclass
class TrialConfig:
    trial_id: str = "SYN-WA-2026"
    date: str = "2026-08-20"
    site: str = "Synthetic dryland site (Northam-like, WA)"
    lat: float = -31.65
    lon: float = 116.67
    epsg: int = 32750
    x0: float = 450000.0
    y0: float = 6498000.0
    n_genotypes: int = 12
    n_reps: int = 2
    plot_w: float = 1.8
    plot_l: float = 5.0
    gap_w: float = 0.5
    gap_l: float = 1.0
    hs_gsd: float = 0.2
    rgb_factor: int = 8           # RGB GSD = hs_gsd / rgb_factor
    lidar_density: float = 200.0
    air_temp_c: float = 19.0
    heritability: float = 0.7
    seed: int = 42
    inject_faults: bool = True


def _design(cfg: TrialConfig, rng) -> pd.DataFrame:
    rows = []
    n_cols = cfg.n_genotypes
    for s_i, sp in enumerate(SPECIES):
        for rep in range(cfg.n_reps):
            order = rng.permutation(cfg.n_genotypes)
            for c in range(n_cols):
                r = s_i * cfg.n_reps + rep + 1
                rows.append({"plot_id": f"P{r:02d}{c + 1:02d}", "species": sp,
                             "genotype": f"{sp[:3].upper()}{order[c] + 1:02d}", "rep": rep + 1})
    return pd.DataFrame(rows)


def _traits(design: pd.DataFrame, cfg: TrialConfig, rng) -> pd.DataFrame:
    """Genotype + rep + residual model with target heritability; returns true traits."""
    out = []
    g_eff = {}
    for sp, d in design.groupby("species"):
        for gname in d.genotype.unique():
            g_eff[gname] = rng.normal(0, 1, 5)  # latent genotype effects for vigour, Cab, height, stress, water
    rep_eff = {r: rng.normal(0, 0.25, 5) for r in design.rep.unique()}
    e_sd = np.sqrt((1 - cfg.heritability) / cfg.heritability)
    for row in design.itertuples():
        p = SPECIES[row.species]
        z = (g_eff[row.genotype] + rep_eff[row.rep] + rng.normal(0, e_sd, 5)) / np.sqrt(1 + e_sd ** 2)
        u = _sig(1.2 * z)  # 0–1 scores
        lai = p["lai"][0] + (p["lai"][1] - p["lai"][0]) * u[0]
        cab = p["cab"][0] + (p["cab"][1] - p["cab"][0]) * u[1]
        h = p["h"][0] + (p["h"][1] - p["h"][0]) * (0.6 * u[2] + 0.4 * u[0])
        stress = np.clip(0.15 + 0.6 * u[3] - 0.2 * u[0], 0.02, 0.95)
        cw = 0.010 + 0.010 * u[4] * (1 - 0.5 * stress)
        fc = 1 - np.exp(-p["k"] * lai)
        biomass = p["bm"] * lai * (0.6 + 0.8 * h) * (1 - 0.25 * stress) + rng.normal(0, 0.15)
        n_pct = 1.2 + p["n_slope"] * cab - 0.3 * stress + rng.normal(0, 0.12)
        out.append({"plot_id": row.plot_id, "lai": lai, "cab": cab, "height_m": h, "stress": stress, "cw": cw,
                    "fc": fc, "biomass_t_ha": max(biomass, 0.2), "n_pct": n_pct,
                    "flower_frac": p["flower"] * (0.5 + u[2]), "wax": p["wax"]})
    return design.merge(pd.DataFrame(out), on="plot_id")


def _cover_map(layout: PlotLayout, traits: pd.DataFrame, tf: pio.GeoTransform, shape, rng):
    """High-resolution vegetation cover (0/1), species-specific row structure inside plots."""
    H, W = shape
    cover = np.zeros(shape, bool)
    flower = np.zeros(shape, bool)
    plot_id_img = np.full(shape, -1, int)
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (pid, (r0, r1, c0, c1)) in enumerate(layout.windows(tf, 0.0, shape).items()):
        t = traits.loc[traits.plot_id == pid].iloc[0]
        plot_id_img[r0:r1, c0:c1] = k
        sub_x = (xx[r0:r1, c0:c1] - c0) * tf.px
        row_sp = 0.25 if t.species in ("wheat", "barley", "field_pea") else 0.45
        # distance to nearest sowing row (rows run along plot length = image rows)
        d = np.abs(((sub_x + row_sp / 2) % row_sp) - row_sp / 2)
        half_w = np.clip(t.fc * row_sp / 2 * 1.15, 0.01, row_sp / 2)
        noise = ndimage.gaussian_filter(rng.normal(0, 1, d.shape), 2.0) * 0.6 * half_w
        cover[r0:r1, c0:c1] = d < half_w + noise
        if t.flower_frac > 0:
            sm = ndimage.gaussian_filter(rng.random(d.shape), 1.0)
            fl = sm > np.quantile(sm, 1 - min(0.9, 0.6 * t.flower_frac))
            flower[r0:r1, c0:c1] = fl & cover[r0:r1, c0:c1]
    return cover, flower, plot_id_img


def _block_mean(a, f):
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    a = a[:H, :W]
    return a.reshape(H // f, f, W // f, f, *a.shape[2:]).mean((1, 3))


def generate_trial(root: str | Path, cfg: TrialConfig | None = None, verbose: bool = True) -> dict:
    """Generate the full synthetic trial. Returns a dict of paths and the injected-fault list."""
    cfg = cfg or TrialConfig()
    rng = np.random.default_rng(cfg.seed)
    root = Path(root)
    log = (lambda *a: print("[synthetic]", *a)) if verbose else (lambda *a: None)
    n_rows, n_cols = len(SPECIES) * cfg.n_reps, cfg.n_genotypes
    design = _design(cfg, rng)
    layout = PlotLayout.grid(n_rows, n_cols, cfg.plot_w, cfg.plot_l, cfg.gap_w, cfg.gap_l,
                             x0=cfg.x0 + 3.0, y0=cfg.y0 + 3.0, design=design)
    traits = _traits(design, cfg, rng)
    field_w = n_cols * (cfg.plot_w + cfg.gap_w) + 6.0
    field_l = n_rows * (cfg.plot_l + cfg.gap_l) + 6.0
    hs_tf = pio.GeoTransform(cfg.x0, cfg.y0 + field_l, cfg.hs_gsd, cfg.hs_gsd, cfg.epsg)
    Hh, Wh = int(round(field_l / cfg.hs_gsd)), int(round(field_w / cfg.hs_gsd))
    f = cfg.rgb_factor
    rgb_tf = hs_tf.scaled(f)
    Hr, Wr = Hh * f, Wh * f
    faults = []
    paths = {}

    def dpath(platform, sensor):
        p = dataset_path(root, cfg.trial_id, cfg.date, platform, sensor, "raw")
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------------------------------------------------------- scene at RGB resolution
    log("rendering cover maps")
    cover, flower, plot_img = _cover_map(layout, traits, rgb_tf, (Hr, Wr), rng)
    soil_b = 1.0 + 0.08 * ndimage.gaussian_filter(rng.normal(0, 1, (Hr, Wr)), 30) * 3
    soil_m = np.clip(0.15 + 0.15 * ndimage.gaussian_filter(rng.normal(0, 1, (Hr, Wr)), 60) * 5, 0, 0.6)
    # panels (3 grey + wet/dry thermal refs) in the south-west margin
    panel_px = int(1.0 / rgb_tf.px)
    panel_boxes = []
    for i in range(len(PANELS)):
        r0, c0 = Hr - int(2.4 / rgb_tf.px), int((0.3 + i * 1.3) / rgb_tf.px)
        panel_boxes.append((r0, r0 + panel_px, c0, c0 + panel_px))
    panel_img = np.full((Hr, Wr), -1, int)
    for i, (r0, r1, c0, c1) in enumerate(panel_boxes):
        panel_img[r0:r1, c0:c1] = i
        cover[r0:r1, c0:c1] = False

    # ---------------------------------------------------------------- per-pixel reflectance (HS grid)
    log("simulating hyperspectral reflectance")
    fc_hs = _block_mean(cover.astype(float), f)
    fl_hs = _block_mean(flower.astype(float), f)
    sb_hs, sm_hs = _block_mean(soil_b, f), _block_mean(soil_m, f)
    plot_hs = plot_img[f // 2::f, f // 2::f][:Hh, :Wh]
    panel_hs_img = panel_img[f // 2::f, f // 2::f][:Hh, :Wh]
    # spectral library per plot (avoid per-pixel model evaluation)
    tbl = traits.set_index("plot_id")
    plot_ids = list(layout.table.plot_id)
    veg_lib = np.array([leaf_canopy_reflectance(VNIR_WL, tbl.loc[p, "cab"], tbl.loc[p, "lai"], tbl.loc[p, "cw"],
                                                tbl.loc[p, "stress"], 0.0, tbl.loc[p, "wax"]) for p in plot_ids])
    flower_spec = leaf_canopy_reflectance(VNIR_WL, 40, 3, flower=1.0) - leaf_canopy_reflectance(VNIR_WL, 40, 3)
    soil_dry = soil_reflectance(VNIR_WL, 1.0, 0.0)
    soil_wet = soil_reflectance(VNIR_WL, 1.0, 0.6)
    soil = sb_hs[..., None] * (soil_dry + (soil_wet - soil_dry) * (sm_hs[..., None] / 0.6))
    veg = np.where(plot_hs[..., None] >= 0, veg_lib[np.clip(plot_hs, 0, None)], leaf_canopy_reflectance(
        VNIR_WL, 30, 1.0)[None, None, :])  # weeds/volunteers in margins
    # sparse weeds in alleys
    fc_eff = np.where(plot_hs >= 0, fc_hs, fc_hs + 0.03 * (rng.random((Hh, Wh)) < 0.05))
    refl = fc_eff[..., None] * (veg + fl_hs[..., None] / np.maximum(fc_hs[..., None], 1e-3) * flower_spec) \
        + (1 - fc_eff[..., None]) * soil
    for i, rp in enumerate(PANELS):
        refl[panel_hs_img == i] = rp
    refl = np.clip(refl, 0.003, 0.95).astype(np.float32)

    # ---------------------------------------------------------------- hyperspectral DN
    hs_dir = dpath("uav", "vnir")
    E = illumination(VNIR_WL)
    gain = 3800 * E / E.max() * (0.85 + 0.15 * _sig((VNIR_WL - 450) / 30))  # QE roll-off in blue
    dark = 60 + 5 * rng.random(len(VNIR_WL))
    dn = refl * gain + dark
    dn = dn + rng.normal(0, 1, dn.shape) * np.sqrt(np.maximum(dn, 1)) * 0.6 + rng.normal(0, 4, dn.shape)
    bad_band = int(np.argmin(np.abs(VNIR_WL - 960)))
    if cfg.inject_faults:
        dn[..., bad_band] += rng.normal(0, 150, dn.shape[:2])  # failing detector column/band
        dn[::7, :, bad_band] += 300  # striping
        faults.append({"id": "hs_bad_band", "where": f"VNIR band {bad_band} ({VNIR_WL[bad_band]:.0f} nm)",
                       "detect_with": "qc.detect_bad_bands"})
    dn = np.clip(np.round(dn), 0, 4095).astype(np.uint16)
    pio.write_envi(hs_dir / "vnir_cube", dn, VNIR_WL, fwhm=np.full(len(VNIR_WL), 6.0), transform=hs_tf,
                   description="synthetic VNIR pushbroom, 12-bit DN")
    panel_tbl = []
    for i, (r0, r1, c0, c1) in enumerate(panel_boxes):
        panel_tbl.append({"panel": i, "reflectance": PANELS[i],
                          "xmin": rgb_tf.x0 + c0 * rgb_tf.px, "xmax": rgb_tf.x0 + c1 * rgb_tf.px,
                          "ymax": rgb_tf.y0 - r0 * rgb_tf.py, "ymin": rgb_tf.y0 - r1 * rgb_tf.py})
    pd.DataFrame(panel_tbl).to_csv(hs_dir / "panels_certified.csv", index=False)
    paths["hyperspectral"] = hs_dir / "vnir_cube.hdr"

    # ---------------------------------------------------------------- multispectral reflectance
    ms_dir = dpath("uav", "ms5")
    ms = gaussian_srf_resample(VNIR_WL, refl, MS_CENTRES, MS_FWHM)
    ms = (ms * (1 + rng.normal(0, 0.01, ms.shape)) + rng.normal(0, 0.003, ms.shape)).astype(np.float32)
    pio.write_envi(ms_dir / "ms_reflectance", ms, MS_CENTRES, MS_FWHM, transform=hs_tf,
                   description="synthetic 5-band multispectral reflectance (DLS+panel calibrated)")
    paths["multispectral"] = ms_dir / "ms_reflectance.hdr"

    # ---------------------------------------------------------------- RGB orthomosaic
    log("rendering RGB")
    rgb_dir = dpath("uav", "rgb")
    soil_rgb = np.array([150, 105, 75], float)
    rgb = soil_rgb * soil_b[..., None] * (1 - 0.35 * soil_m[..., None])
    rgb += ndimage.gaussian_filter(rng.normal(0, 8, (Hr, Wr, 3)), (1, 1, 0))
    cab_img = np.where(plot_img >= 0, tbl.cab.reindex(plot_ids).to_numpy()[np.clip(plot_img, 0, None)], 35)
    shade = 0.75 + 0.25 * ndimage.gaussian_filter(rng.random((Hr, Wr)), 1.5)
    green = np.stack([90 - 0.9 * cab_img, 150 - 0.9 * cab_img, 55 - 0.4 * cab_img], -1) * shade[..., None]
    rgb = np.where(cover[..., None], green, rgb)
    rgb = np.where(flower[..., None], np.array([235, 215, 40], float) * shade[..., None], rgb)
    for i, (r0, r1, c0, c1) in enumerate(panel_boxes):
        rgb[r0:r1, c0:c1] = 255 * PANELS[i] ** (1 / 2.2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    pio.write_image(rgb_dir / "rgb_ortho.png", rgb, rgb_tf)
    paths["rgb"] = rgb_dir / "rgb_ortho.png"
    # raw frames (for frame-level QC)
    frames_dir = rgb_dir / "frames"
    fh, fw = 480, 720
    for k in range(6):
        r0 = int(rng.integers(0, Hr - fh))
        c0 = int(rng.integers(0, Wr - fw))
        fr = rgb[r0:r0 + fh, c0:c0 + fw].astype(float)
        if cfg.inject_faults and k == 2:
            fr = ndimage.uniform_filter(fr, (9, 9, 1))
        if cfg.inject_faults and k == 4:
            fr = fr * 2.2
        pio.write_image(frames_dir / f"frame_{k:03d}.png", np.clip(fr, 0, 255).astype(np.uint8))
    if cfg.inject_faults:
        # byte copy (shutil.copy's same-file check misfires on some network shares)
        (frames_dir / "frame_006.png").write_bytes((frames_dir / "frame_000.png").read_bytes())
        faults += [{"id": "rgb_blur", "where": "frames/frame_002.png", "detect_with": "qc.qc_rgb sharpness"},
                   {"id": "rgb_overexposed", "where": "frames/frame_004.png", "detect_with": "qc.qc_rgb exposure"},
                   {"id": "rgb_duplicate", "where": "frames/frame_006.png == frame_000.png",
                    "detect_with": "qc.average_hash"}]

    # ---------------------------------------------------------------- thermal
    log("simulating thermal")
    th_dir = dpath("uav", "thermal")
    tc_plot = cfg.air_temp_c - 3.0 + 9.0 * traits.set_index("plot_id").stress.reindex(plot_ids).to_numpy()
    tc_img = np.where(plot_hs >= 0, tc_plot[np.clip(plot_hs, 0, None)], cfg.air_temp_c + 2)
    t_soil = cfg.air_temp_c + 14 + 3 * (0.3 - sm_hs)
    temp = fc_eff * tc_img + (1 - fc_eff) * t_soil + rng.normal(0, 0.15, (Hh, Wh))
    # wet (water-saturated cloth) and dry (vaselined leaf) references
    wr, wc = Hh - 12, Wh - 14
    temp[wr:wr + 5, wc:wc + 5] = cfg.air_temp_c - 4.5
    temp[wr:wr + 5, wc + 7:wc + 12] = cfg.air_temp_c + 7.5
    if cfg.inject_faults:
        yy, xx = np.mgrid[0:Hh, 0:Wh]
        rad = np.hypot((yy - Hh / 2) / (Hh / 2), (xx - Wh / 2) / (Wh / 2))
        temp = temp + 1.6 * rad ** 2  # vignetting / non-uniformity
        faults.append({"id": "thermal_vignetting", "where": "thermal mosaic corners +1.6 °C",
                       "detect_with": "qc.qc_thermal corner_centre_delta"})
    pio.write_envi(th_dir / "thermal_brightness_c", temp.astype(np.float32), transform=hs_tf,
                   description="synthetic radiometric thermal, brightness temperature degC",
                   extra={"emissivity assumed": 1.0})
    pio.write_json(th_dir / "references.json", {"wet_ref_px": [wr, wr + 5, wc, wc + 5],
                                                 "dry_ref_px": [wr, wr + 5, wc + 7, wc + 12],
                                                 "air_temp_c": cfg.air_temp_c, "rh_pct": 45})
    paths["thermal"] = th_dir / "thermal_brightness_c.hdr"

    # ---------------------------------------------------------------- LiDAR
    log("simulating LiDAR")
    li_dir = dpath("uav", "lidar")
    area = field_w * field_l
    n = int(cfg.lidar_density * area)
    x = rng.uniform(0, field_w, n)
    y = rng.uniform(0, field_l, n)
    if cfg.inject_faults:  # missed swath: thin out a 3 m strip
        strip = (x > field_w * 0.62) & (x < field_w * 0.62 + 3.0)
        keep = ~strip | (rng.random(n) < 0.12)
        x, y = x[keep], y[keep]
        faults.append({"id": "lidar_gap", "where": f"x∈[{field_w * 0.62:.1f},{field_w * 0.62 + 3:.1f}] m density×0.12",
                       "detect_with": "qc.qc_pointcloud density_cv"})
    ground = 0.005 * y + 0.08 * np.sin(x / 7) + 0.05 * np.cos(y / 11)
    rr, cc = rgb_tf.map_to_pixel(cfg.x0 + x, cfg.y0 + y)
    rr, cc = np.clip(rr.astype(int), 0, Hr - 1), np.clip(cc.astype(int), 0, Wr - 1)
    pk = plot_img[rr, cc]
    hplot = tbl.height_m.reindex(plot_ids).to_numpy()
    h = np.where(pk >= 0, hplot[np.clip(pk, 0, None)], 0.15)
    is_veg = cover[rr, cc] | (rng.random(len(x)) < 0.15 * (pk >= 0))
    rel = rng.beta(5, 1.6, len(x))  # returns concentrate near canopy top
    z = ground + np.where(is_veg, h * rel, 0.0) + rng.normal(0, 0.015, len(x))
    cls = np.where(is_veg, 5, 2).astype(np.uint8)  # ASPRS: 2 ground, 5 high vegetation
    if cfg.inject_faults:
        k = 60
        x = np.r_[x, rng.uniform(0, field_w, k)]
        y = np.r_[y, rng.uniform(0, field_l, k)]
        z = np.r_[z, rng.uniform(8, 25, k)]
        cls = np.r_[cls, np.full(k, 7, np.uint8)]  # 7 = noise
        faults.append({"id": "lidar_outliers", "where": f"{k} airborne noise points 8–25 m",
                       "detect_with": "qc.qc_pointcloud / pointcloud.statistical_outlier_removal"})
    xyz = np.c_[cfg.x0 + x, cfg.y0 + y, 250.0 + z]
    pio.write_las(li_dir / "lidar.las", xyz, classification=cls,
                  intensity=np.clip(rng.normal(np.where(cls == 5, 900, 400), 80), 0, 65535))
    paths["lidar"] = li_dir / "lidar.las"

    # ---------------------------------------------------------------- ground vehicle active NDVI + GNSS
    log("simulating ground vehicle")
    gv_dir = dpath("ground", "ndvi_gnss")
    recs = []
    t = 0.0
    speed, rate = 1.0, 10.0
    red_wl, nir_wl = 660, 780
    for c in range(n_cols):
        xs = cfg.x0 + 3.0 + c * (cfg.plot_w + cfg.gap_w) + cfg.plot_w / 2
        ys = np.arange(cfg.y0 + 2.0, cfg.y0 + field_l - 2.0, speed / rate)
        if c % 2:
            ys = ys[::-1]
        for yv in ys:
            r, cidx = hs_tf.map_to_pixel(xs + rng.normal(0, 0.02), yv)
            r, cidx = int(np.clip(r, 0, Hh - 1)), int(np.clip(cidx, 0, Wh - 1))
            spec = refl[r, cidx]
            red = spec[np.argmin(np.abs(VNIR_WL - red_wl))]
            nir = spec[np.argmin(np.abs(VNIR_WL - nir_wl))]
            ndvi = (nir - red) / (nir + red) * 0.97 + 0.01 + rng.normal(0, 0.01)  # sensor-specific scaling
            recs.append({"t_s": round(t, 2), "easting": xs + rng.normal(0, 0.01), "northing": yv,
                         "fix": "RTK_FIXED", "ndvi": ndvi, "sensor_height_m": 0.9 + rng.normal(0, 0.02)})
            t += 1 / rate
        # headland turn: logging continues while the vehicle moves to the next pass (outside plots)
        if c < n_cols - 1:
            x_next = xs + cfg.plot_w + cfg.gap_w
            for k in range(1, int(20 * rate)):
                frac = k / (20 * rate)
                recs.append({"t_s": round(t, 2), "easting": xs + frac * (x_next - xs),
                             "northing": yv + (1 if c % 2 == 0 else -1) * 1.0 * np.sin(np.pi * frac),
                             "fix": "RTK_FIXED", "ndvi": np.nan, "sensor_height_m": 0.9})
                t += 1 / rate
    gv = pd.DataFrame(recs)
    if cfg.inject_faults:
        i0 = len(gv) // 3
        gv.loc[i0:i0 + 80, "fix"] = "RTK_FLOAT"
        gv = gv.drop(index=range(2 * len(gv) // 3, 2 * len(gv) // 3 + 40)).reset_index(drop=True)
        faults += [{"id": "gnss_float", "where": "ground vehicle log, 81 records RTK_FLOAT",
                    "detect_with": "qc.qc_gnss_track"},
                   {"id": "gnss_gap", "where": "ground vehicle log, 4 s gap", "detect_with": "qc.qc_gnss_track"}]
    gv.to_csv(gv_dir / "ndvi_track.csv", index=False)
    paths["ground"] = gv_dir / "ndvi_track.csv"

    # ---------------------------------------------------------------- handheld spectroradiometer
    log("simulating handheld spectra")
    hh_dir = dpath("handheld", "spectroradiometer")
    cols = {"wavelength_nm": FULL_WL}
    splice_plot = plot_ids[17]
    for p in plot_ids:
        tr = tbl.loc[p]
        for s in range(3):
            fc_local = np.clip(tr.fc + rng.normal(0, 0.05), 0.05, 0.99)
            spec = fc_local * leaf_canopy_reflectance(FULL_WL, tr.cab, tr.lai, tr.cw, tr.stress,
                                                      tr.flower_frac * 0.8, tr.wax) \
                + (1 - fc_local) * soil_reflectance(FULL_WL, 1.0, 0.15)
            spec = spec + rng.normal(0, 0.002, len(FULL_WL)) * (1 + 6 * (FULL_WL > 2350))
            spec[FULL_WL >= 1000] += rng.normal(0, 0.0015)  # small natural detector-join steps
            spec[FULL_WL >= 1800] += rng.normal(0, 0.0015)
            if cfg.inject_faults and p == splice_plot:
                spec[FULL_WL >= 1000] += 0.035
            cols[f"{p}_s{s + 1}"] = np.round(spec, 5)
    pd.DataFrame(cols).to_csv(hh_dir / "canopy_spectra.csv", index=False)
    if cfg.inject_faults:
        faults.append({"id": "handheld_splice", "where": f"{splice_plot} spectra +0.035 step at 1000 nm",
                       "detect_with": "qc.qc_spectrum_splice"})
    paths["handheld"] = hh_dir / "canopy_spectra.csv"

    # ---------------------------------------------------------------- ground truth & layout
    gt_dir = root / cfg.trial_id / cfg.date.replace("-", "") / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt = traits.copy()
    gt["biomass_meas_t_ha"] = gt.biomass_t_ha * (1 + rng.normal(0, 0.07, len(gt)))
    gt["height_meas_m"] = gt.height_m + rng.normal(0, 0.03, len(gt))
    gt["spad"] = 0.8 * gt.cab + 5 + rng.normal(0, 1.5, len(gt))
    gt["n_meas_pct"] = gt.n_pct + rng.normal(0, 0.08, len(gt))
    if cfg.inject_faults:
        bad = gt.index[40]
        gt.loc[bad, "biomass_meas_t_ha"] *= 1000  # kg/ha entered instead of t/ha
        faults.append({"id": "groundtruth_unit_error", "where": f"{gt.loc[bad, 'plot_id']} biomass ×1000",
                       "detect_with": "qc.qc_plot_values"})
    gt.to_csv(gt_dir / "ground_truth_true_and_measured.csv", index=False)
    gt[["plot_id", "species", "genotype", "rep", "biomass_meas_t_ha", "height_meas_m", "spad",
        "n_meas_pct"]].to_csv(gt_dir / "field_measurements.csv", index=False)
    layout.to_csv(root / cfg.trial_id / "plot_layout.csv")
    paths["layout"] = root / cfg.trial_id / "plot_layout.csv"
    paths["ground_truth"] = gt_dir / "field_measurements.csv"
    paths["truth"] = gt_dir / "ground_truth_true_and_measured.csv"

    # ---------------------------------------------------------------- metadata + manifest
    when = f"{cfg.date}T12:20:00+08:00"
    for platform, sensor, d in [("uav", "vnir", hs_dir), ("uav", "ms5", ms_dir), ("uav", "rgb", rgb_dir),
                                ("uav", "thermal", th_dir), ("uav", "lidar", li_dir),
                                ("ground", "ndvi_gnss", gv_dir), ("handheld", "spectroradiometer", hh_dir)]:
        AcquisitionMetadata("Synthetic multi-platform phenotyping demonstration", cfg.trial_id, cfg.site, cfg.lat,
                            cfg.lon, list(SPECIES), platform, sensor, when, "phenokit-synthetic",
                            growth_stage="cereals Z61 / canola 60% flowering",
                            plot_layout_file="../../../plot_layout.csv",
                            weather={"air_temp_c": cfg.air_temp_c, "rh_pct": 45, "wind_ms": 3.2,
                                     "cloud_cover_okta": 1}).save(d)
    acq_records = {
        "uav_multispectral": {"trial_id": cfg.trial_id, "date": cfg.date, "operator": "pilot-A", "platform": "uav",
                              "sensor": "ms5", "altitude_m": 30, "gsd_m": 0.02, "panel_before": True,
                              "panel_after": not cfg.inject_faults, "irradiance_sensor": "DLS",
                              "wind_ms": 9.1 if cfg.inject_faults else 3.2, "cloud_cover_okta": 1,
                              "solar_elevation_deg": 41.0, "gnss_mode": "RTK"},
    }
    pio.write_json(root / cfg.trial_id / "acquisition_log.json", acq_records)
    if cfg.inject_faults:
        faults.append({"id": "protocol_violation", "where": "acquisition_log.json: wind 9.1 m/s, no after-flight panel",
                       "detect_with": "acquisition.validate_acquisition"})
    trial_root = root / cfg.trial_id
    build_manifest(trial_root)
    if cfg.inject_faults:  # silent corruption after ingest (bit-rot / bad copy)
        raw = th_dir / "thermal_brightness_c.raw"
        b = bytearray(raw.read_bytes())
        b[len(b) // 2] ^= 0xFF
        raw.write_bytes(bytes(b))
        faults.append({"id": "file_corruption", "where": "thermal_brightness_c.raw (1 byte flipped after manifest)",
                       "detect_with": "catalog.verify_manifest"})
    pio.write_json(trial_root / "INJECTED_FAULTS.json", {"config": dict(cfg.__dict__), "faults": faults})
    log(f"done: {trial_root}")
    return {"root": trial_root, "paths": paths, "faults": faults, "config": cfg, "layout": layout,
            "hs_transform": hs_tf, "rgb_transform": rgb_tf}
