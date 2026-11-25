"""Unit tests: each checks a property with a known answer (round-trips, analytic values, injected faults)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from phenokit import acquisition as acq
from phenokit import calibration, catalog, indices, io as pio, ml, plots, pointcloud, qc, segmentation, spectral, validation


# ------------------------------------------------------------------ io
def test_envi_roundtrip_all_interleaves(tmp_path):
    rng = np.random.default_rng(0)
    cube = rng.random((7, 5, 4)).astype(np.float32)
    tf = pio.GeoTransform(100.0, 200.0, 0.5, 0.5, 32750)
    for il in ("bsq", "bil", "bip"):
        hdr = pio.write_envi(tmp_path / f"c_{il}", cube, [500, 600, 700, 800], transform=tf, interleave=il)
        back, meta = pio.read_envi(hdr)
        np.testing.assert_array_equal(back, cube)
        assert meta["wavelength"] == [500, 600, 700, 800]
        g = meta["geotransform"]
        assert (g.x0, g.y0, g.px, g.epsg) == (100.0, 200.0, 0.5, 32750)


def test_envi_truncated_file_detected(tmp_path):
    hdr = pio.write_envi(tmp_path / "c", np.ones((4, 4, 3), np.float32))
    raw = tmp_path / "c.raw"
    raw.write_bytes(raw.read_bytes()[:-10])
    with pytest.raises(ValueError, match="truncated"):
        pio.read_envi(hdr)


def test_las_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    xyz = rng.random((1000, 3)) * [100, 50, 5] + [450000, 6498000, 250]
    cls = rng.integers(1, 6, 1000)
    p = pio.write_las(tmp_path / "a.las", xyz, classification=cls)
    d = pio.read_las(p)
    np.testing.assert_allclose(d["xyz"], xyz, atol=0.0006)
    np.testing.assert_array_equal(d["classification"], cls)


def test_worldfile_roundtrip(tmp_path):
    tf = pio.GeoTransform(450000, 6498066, 0.025, 0.025, 32750)
    p = pio.write_image(tmp_path / "x.png", np.zeros((10, 10, 3), np.uint8), tf)
    t2 = pio.read_worldfile(p)
    assert np.allclose([t2.x0, t2.y0, t2.px, t2.py], [tf.x0, tf.y0, tf.px, tf.py]) and t2.epsg == 32750


def test_geotransform_inverse():
    tf = pio.GeoTransform(10, 20, 0.2, 0.2)
    x, y = tf.pixel_to_map(3, 7)
    r, c = tf.map_to_pixel(x, y)
    assert np.isclose(r, 3.5) and np.isclose(c, 7.5)  # pixel centre


# ------------------------------------------------------------------ acquisition
def test_gsd_and_altitude_inverse():
    s = acq.SENSOR_PRESETS["rgb_20mp"]
    g = acq.gsd_m(s, 30)
    assert np.isclose(acq.altitude_for_gsd(s, g), 30)
    assert 0.007 < g < 0.0085  # 2.41 µm pitch × 30 m / 8.8 mm


def test_solar_noon_elevation_equinox_equator():
    e, _ = acq.solar_position(0.0, 0.0, datetime(2026, 3, 20, 12, 7, tzinfo=timezone.utc))
    assert e > 88


def test_plan_flight_overlap_spacing():
    s = acq.SENSOR_PRESETS["ms_5band"]
    fp = acq.plan_flight(s, 30, 60, 40, side_overlap=0.75)
    across, _ = acq.footprint_m(s, 40)
    assert np.isclose(fp.line_spacing_m, across * 0.25)
    assert fp.flight_time_min > 0 and len(fp.waypoints) == 2 * fp.n_lines


def test_protocol_validation():
    issues = acq.validate_acquisition({"trial_id": "T", "sensor_warmup_min": 5}, "uav_thermal")
    fields = {i["field"] for i in issues}
    assert "operator" in fields and "sensor_warmup_min" in fields


# ------------------------------------------------------------------ catalogue
def test_manifest_detects_modification_and_deletion(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01")
    catalog.build_manifest(tmp_path)
    assert catalog.verify_manifest(tmp_path)["ok"]
    (tmp_path / "a.txt").write_text("hellO")
    (tmp_path / "sub" / "b.bin").unlink()
    r = catalog.verify_manifest(tmp_path)
    assert r["modified"] == ["a.txt"] and r["missing"] == ["sub/b.bin"] and not r["ok"]


def test_catalog_register_and_verify(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "f.txt").write_text("x")
    with catalog.DataCatalog(tmp_path / "c.sqlite") as cat:
        i = cat.register_dataset(d, "T", "2026-08-20", "uav", "rgb")
        assert cat.verify_dataset(i)["ok"]
        (d / "f.txt").write_text("y")
        assert cat.verify_dataset(i)["modified"] == ["f.txt"]
        cat.record_step("x", [d], [], {"a": 1})
        assert len(cat.query("SELECT * FROM provenance")) == 1


def test_metadata_validation():
    m = catalog.AcquisitionMetadata("I", "S", "site", -31.6, 116.7, ["wheat"], "uav", "rgb", "2026-08-20T12:00:00",
                                    "op")
    assert any("UTC offset" in e for e in m.validate())


# ------------------------------------------------------------------ QC
def test_blur_and_exposure_checks():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 200, 3)) * 200).astype(np.uint8)
    sharp = {r.check: r for r in qc.qc_rgb(img)}
    from scipy import ndimage

    blurred = ndimage.uniform_filter(img.astype(float), (9, 9, 1)).astype(np.uint8)
    blur = {r.check: r for r in qc.qc_rgb(blurred)}
    assert blur["sharpness_laplacian_var"].value < sharp["sharpness_laplacian_var"].value / 10
    over = {r.check: r for r in qc.qc_rgb(np.full((50, 50, 3), 255, np.uint8))}
    assert over["overexposed_fraction"].status == qc.FAIL


def test_band_snr_finds_noisy_band():
    rng = np.random.default_rng(0)
    base = np.repeat(np.linspace(0.1, 0.5, 30)[None, None], 1, 0)
    scene = rng.random((60, 60, 1)) * base + 0.2
    cube = scene + rng.normal(0, 0.002, (60, 60, 30))
    cube[..., 17] += rng.normal(0, 0.05, (60, 60))
    bad = qc.detect_bad_bands(cube, snr_min=20)
    assert bad["bad_bands"] == [17]


def test_split_leakage_and_duplicates():
    r = qc.qc_split_leakage({"train": [1, 2, 28], "test": [3, 28]})
    assert r[0].status == qc.FAIL and r[0].value == 1
    h = {"a": 0b1111, "b": 0b1110, "c": 0xFFFF0000}
    assert qc.find_near_duplicates(h, 1) == [("a", "b", 1)]


def test_thermal_vignetting_cover_adjusted():
    rng = np.random.default_rng(0)
    cover = (rng.random((80, 80)) > 0.5).astype(float)
    t = 30 - 8 * cover + rng.normal(0, 0.1, cover.shape)
    flat = {r.check: r for r in qc.qc_thermal(t, covariate=cover)}
    assert flat["radial_nonuniformity_c"].status == qc.PASS
    c, r2 = qc.radial_trend(t, cover)
    vig = {r.check: r for r in qc.qc_thermal(t + 2 * r2, covariate=cover)}
    assert vig["radial_nonuniformity_c"].status == qc.FAIL and abs(vig["radial_nonuniformity_c"].value - 2) < 0.1


def test_pointcloud_gap_and_isolated_points():
    rng = np.random.default_rng(0)
    xy = rng.random((40000, 2)) * 20
    keep = ~((xy[:, 0] > 8) & (xy[:, 0] < 11)) | (rng.random(40000) < 0.1)
    xyz = np.c_[xy[keep], rng.normal(0, 0.01, keep.sum())]
    xyz = np.r_[xyz, [[5, 5, 20.0], [15, 15, 30.0]]]
    r = {x.check: x for x in qc.qc_pointcloud(xyz, area_m2=400, min_density=50)}
    assert r["low_density_cell_fraction"].status == qc.FAIL
    assert r["isolated_points"].value >= 2


def test_splice_detection_and_correction():
    wl = np.arange(350, 2501, 1.0)
    s = 0.3 + 0.0001 * (wl - 350)
    s2 = s.copy()
    s2[wl >= 1000] += 0.03
    assert qc.qc_spectrum_splice(wl, s2[None])[0].status == qc.FAIL
    fixed = spectral.splice_correction(wl, s2[None])[0]
    assert np.abs(fixed - s).max() < 1e-3


# ------------------------------------------------------------------ calibration, spectral, indices
def test_empirical_line_recovers_reflectance():
    rng = np.random.default_rng(0)
    refl = rng.random((40, 40, 5)) * 0.6
    gain, off = np.array([1000, 1200, 900, 1500, 800.0]), np.array([50, 60, 40, 30, 20.0])
    dn = refl * gain + off
    masks, vals = [], [0.05, 0.25, 0.5]
    for i, v in enumerate(vals):
        m = np.zeros((40, 40), bool)
        m[i * 5:i * 5 + 4, :4] = True
        dn[m] = v * gain + off
        refl[m] = v
        masks.append(m)
    el = calibration.fit_empirical_line(dn, masks, vals)
    np.testing.assert_allclose(el.apply(dn), refl, atol=1e-9)
    assert np.all(el.r2 > 0.999999)


def test_surface_temperature_identity_and_direction():
    assert np.isclose(calibration.surface_temperature(25.0, emissivity=1.0), 25.0)
    # with ε<1 and a cold sky reflection, true surface is warmer than brightness temperature
    assert calibration.surface_temperature(25.0, 0.95, reflected_temp_c=-20) > 25.0


def test_indices_known_values_and_band_guard():
    wl = [550, 670, 720, 800]
    px = np.array([[0.1, 0.05, 0.2, 0.45]])
    assert np.isclose(indices.compute("NDVI", px, wl)[0], (0.45 - 0.05) / 0.5)
    with pytest.raises(ValueError):
        indices.compute("WBI", px, wl)
    # broadband NIR (842 nm, FWHM 57) qualifies for 800 nm only when FWHM is supplied
    with pytest.raises(ValueError):
        indices.compute("NDVI", px, [550, 670, 720, 842])
    indices.compute("NDVI", px, [550, 670, 720, 842], fwhm=[27, 14, 12, 57])


def test_snv_msc_properties():
    rng = np.random.default_rng(0)
    base = np.sin(np.linspace(0, 3, 50))
    X = np.array([a * base + b for a, b in rng.random((10, 2)) + 0.5])
    s = spectral.snv(X)
    assert np.allclose(s.mean(1), 0) and np.allclose(s.std(1), 1)
    m, _ = spectral.msc(X)
    assert np.allclose(m, m[0], atol=1e-8)  # pure additive/multiplicative effects removed


def test_srf_resample_flat_spectrum():
    wl = np.arange(400, 1001, 2.0)
    out = spectral.gaussian_srf_resample(wl, np.full((3, len(wl)), 0.3), [560, 668], [27, 14])
    assert np.allclose(out, 0.3)


# ------------------------------------------------------------------ segmentation, plots, point clouds
def test_otsu_bimodal():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0, 0.05, 5000), rng.normal(1, 0.05, 5000)]
    assert 0.3 < segmentation.otsu_threshold(v) < 0.7


def test_zonal_stats_on_known_raster():
    layout = plots.PlotLayout.grid(1, 2, 1.0, 2.0, 1.0, 0.0, x0=0, y0=0)
    tf = pio.GeoTransform(0, 2.0, 0.1, 0.1)
    r = np.zeros((20, 30))
    r[:, 20:30] = 5.0  # second plot spans x 2–3 m
    z = plots.zonal_stats(r, layout, tf, prefix="v_", band_names=["v"])
    assert np.allclose(z.v_mean, [0, 5]) and (z.v_n_px == 200).all()


def test_ransac_plane_and_height():
    rng = np.random.default_rng(0)
    ground = np.c_[rng.random((3000, 2)), rng.normal(0, 0.001, 3000)]
    plant = np.c_[0.5 + rng.normal(0, 0.02, (500, 2)), rng.random(500) * 0.4]
    t = pointcloud.plant_traits_3d(np.r_[ground, plant])
    assert 0.38 < t["height"] < 0.41


# ------------------------------------------------------------------ ML & validation
def test_agreement_perfect_and_biased():
    y = np.arange(10.0)
    a = validation.agreement(y, y)
    assert a["r2"] == 1 and a["ccc"] == 1 and a["rmse"] == 0
    b = validation.agreement(y, y + 1)
    assert np.isclose(b["bias"], 1) and b["ccc"] < 1 and np.isclose(b["r"], 1)


def test_heritability_extremes():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 3)
    rep = np.tile([1, 2, 3], 20)
    df = pd.DataFrame({"genotype": g, "rep": rep, "y": rng.normal(0, 1, 20)[g] * 5 + rng.normal(0, 0.1, 60)})
    assert validation.heritability(df, "y")["H2"] > 0.98
    df["y"] = rng.normal(0, 1, 60)
    assert validation.heritability(df, "y")["H2"] < 0.5


def test_spatial_blocking_reduces_leakage():
    """Autocorrelated field with spatially structured classes: random CV is optimistic."""
    from scipy import ndimage

    rng = np.random.default_rng(0)
    n = 64
    field = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 6)
    y = (field > 0).astype(int).ravel()
    X = np.stack([ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 4).ravel() for _ in range(5)], 1)
    rr, cc = np.divmod(np.arange(n * n), n)
    X = np.c_[X, rr / n, cc / n]  # location-like features enable memorisation
    df = ml.compare_validation_schemes_classification(ml.default_classifier("rf"), X, y, rr, cc, block=16)
    acc = df.set_index("scheme").overall_accuracy
    assert acc["random"] > acc["spatial_block"] + 0.1
