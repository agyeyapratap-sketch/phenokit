"""Acquisition planning for handheld, ground-based and aerial (UAV) phenotyping.

* Sensor and platform specifications (generic presets — replace with your own).
* Ground sampling distance (GSD), footprint, overlap-driven flight-line spacing and
  trigger intervals, motion-blur budget, lawnmower waypoints.
* Ground-vehicle transect planning (speed vs. along-track sampling).
* Solar geometry (NOAA algorithm) to choose acquisition windows around solar noon.
* Protocol checklists that turn an SOP into machine-checkable metadata validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

# --------------------------------------------------------------------------- sensors


@dataclass
class SensorSpec:
    name: str
    kind: str  # rgb | multispectral | hyperspectral | thermal | lidar | spectroradiometer
    focal_length_mm: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    image_width_px: int | None = None
    image_height_px: int | None = None
    pushbroom: bool = False
    band_centres_nm: list[float] = field(default_factory=list)
    band_fwhm_nm: list[float] = field(default_factory=list)
    max_frame_rate_hz: float | None = None
    fov_deg: float | None = None            # LiDAR / spectroradiometer field of view
    pulse_rate_hz: float | None = None      # LiDAR
    notes: str = ""

    def pixel_pitch_mm(self) -> float:
        return self.sensor_width_mm / self.image_width_px


# Generic presets inspired by common phenotyping payload classes; numbers are typical,
# not vendor specifications.
SENSOR_PRESETS: dict[str, SensorSpec] = {
    "rgb_20mp": SensorSpec("rgb_20mp", "rgb", 8.8, 13.2, 8.8, 5472, 3648, max_frame_rate_hz=1.0,
                           notes="1-inch 20 MP mechanical-shutter RGB"),
    "ms_5band": SensorSpec("ms_5band", "multispectral", 5.5, 4.8, 3.6, 1280, 960,
                           band_centres_nm=[475, 560, 668, 717, 842], band_fwhm_nm=[32, 27, 14, 12, 57],
                           max_frame_rate_hz=1.0, notes="5-band global-shutter multispectral + DLS"),
    "vnir_pushbroom": SensorSpec("vnir_pushbroom", "hyperspectral", 12.0, 7.68, None, 640, None, pushbroom=True,
                                 band_centres_nm=list(np.arange(400, 1001, 4.0)),
                                 band_fwhm_nm=[6.0] * 151, max_frame_rate_hz=300.0,
                                 notes="VNIR line scanner 400–1000 nm"),
    "thermal_640": SensorSpec("thermal_640", "thermal", 13.0, 10.88, 8.7, 640, 512, max_frame_rate_hz=9.0,
                              notes="Radiometric LWIR microbolometer 7.5–13.5 µm"),
    "lidar_uav": SensorSpec("lidar_uav", "lidar", fov_deg=70.4, pulse_rate_hz=240_000,
                            notes="UAV LiDAR, multi-return"),
    "spectroradiometer": SensorSpec("spectroradiometer", "spectroradiometer", fov_deg=25.0,
                                    band_centres_nm=list(np.arange(350, 2501, 1.0)),
                                    notes="Handheld 350–2500 nm, detector joins at 1000 & 1800 nm"),
}


@dataclass
class Platform:
    name: str
    tier: str  # handheld | ground | aerial
    sensors: list[str]
    typical_height_m: float
    typical_speed_ms: float
    positioning: str = "RTK-GNSS"
    notes: str = ""


PLATFORM_PRESETS = {
    "handheld": Platform("handheld", "handheld", ["spectroradiometer", "rgb_20mp"], 1.0, 0.0, "plot ID / barcode",
                         "Point spectra, SPAD/leaf-level, RGB close-range; ground-truth tier"),
    "ground": Platform("ground", "ground", ["rgb_20mp", "lidar_uav", "ms_5band", "thermal_640"], 1.5, 1.0,
                       "RTK-GNSS", "Proximal phenomobile/buggy; sub-cm GSD, canopy-structure tier"),
    "uav": Platform("uav", "aerial", ["rgb_20mp", "ms_5band", "vnir_pushbroom", "thermal_640", "lidar_uav"], 30.0,
                    5.0, "RTK/PPK-GNSS", "Whole-trial coverage in minutes; scale tier"),
}

# --------------------------------------------------------------------------- frame geometry


def gsd_m(sensor: SensorSpec, altitude_m: float) -> float:
    """Ground sampling distance (m/pixel) for a nadir frame or line camera."""
    return sensor.pixel_pitch_mm() * altitude_m / sensor.focal_length_mm


def altitude_for_gsd(sensor: SensorSpec, gsd_target_m: float) -> float:
    return gsd_target_m * sensor.focal_length_mm / sensor.pixel_pitch_mm()


def footprint_m(sensor: SensorSpec, altitude_m: float) -> tuple[float, float]:
    """(across-track, along-track) footprint in metres. Pushbroom along-track = 1 line."""
    g = gsd_m(sensor, altitude_m)
    along = g if sensor.pushbroom else g * sensor.image_height_px
    return g * sensor.image_width_px, along


def lidar_swath_m(sensor: SensorSpec, altitude_m: float) -> float:
    return 2 * altitude_m * math.tan(math.radians(sensor.fov_deg / 2))


@dataclass
class FlightPlan:
    sensor: str
    altitude_m: float
    gsd_m: float
    speed_ms: float
    line_spacing_m: float
    trigger_distance_m: float | None
    trigger_interval_s: float | None
    n_lines: int
    line_length_m: float
    flight_time_min: float
    motion_blur_px: float | None
    waypoints: np.ndarray  # (n, 2) x, y in field coordinates
    warnings: list[str]

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "waypoints"}
        d["n_waypoints"] = len(self.waypoints)
        return d


def plan_flight(sensor: SensorSpec, field_width_m: float, field_length_m: float, altitude_m: float,
                speed_ms: float = 5.0, front_overlap: float = 0.8, side_overlap: float = 0.7,
                exposure_s: float = 1 / 1000, turn_time_s: float = 8.0, margin_m: float = 10.0,
                lidar_points_per_m2: float | None = None) -> FlightPlan:
    """Plan a lawnmower survey over a rectangular trial.

    Flight lines run along ``field_length_m``. For pushbroom sensors the along-track sampling
    is set by the frame rate; for LiDAR the swath and pulse rate set density.
    """
    warnings: list[str] = []
    if sensor.kind == "lidar":
        swath = lidar_swath_m(sensor, altitude_m)
        spacing = swath * (1 - side_overlap)
        g = float("nan")
        trig_d = trig_t = blur = None
        density = sensor.pulse_rate_hz / (swath * speed_ms) / (1 - side_overlap)
        if lidar_points_per_m2 and density < lidar_points_per_m2:
            warnings.append(f"Estimated density {density:.0f} pts/m² < target {lidar_points_per_m2}")
    else:
        g = gsd_m(sensor, altitude_m)
        across, along = footprint_m(sensor, altitude_m)
        spacing = across * (1 - side_overlap)
        if sensor.pushbroom:
            trig_d = g  # square pixels need one line per GSD of travel
            trig_t = trig_d / speed_ms
            if sensor.max_frame_rate_hz and 1 / trig_t > sensor.max_frame_rate_hz:
                warnings.append(f"Needs {1 / trig_t:.0f} lines/s > sensor max {sensor.max_frame_rate_hz}; "
                                f"slow to {g * sensor.max_frame_rate_hz:.2f} m/s or fly higher")
        else:
            trig_d = along * (1 - front_overlap)
            trig_t = trig_d / speed_ms
            if sensor.max_frame_rate_hz and trig_t < 1 / sensor.max_frame_rate_hz:
                warnings.append(f"Trigger interval {trig_t:.2f}s faster than sensor max rate; reduce speed to "
                                f"{trig_d * sensor.max_frame_rate_hz:.2f} m/s")
        blur = speed_ms * exposure_s / g
        if blur > 0.5:
            warnings.append(f"Motion blur {blur:.2f} px > 0.5 px; shorten exposure or reduce speed")
    width = field_width_m + 2 * margin_m
    n_lines = max(1, int(math.ceil(width / spacing)) + 1)
    length = field_length_m + 2 * margin_m
    xs = -margin_m + np.arange(n_lines) * spacing
    wps = []
    for i, x in enumerate(xs):
        ys = (-margin_m, field_length_m + margin_m)
        wps += [(x, ys[i % 2]), (x, ys[1 - i % 2])]
    t_s = n_lines * length / speed_ms + (n_lines - 1) * turn_time_s
    return FlightPlan(sensor.name, altitude_m, g, speed_ms, spacing, trig_d, trig_t, n_lines, length,
                      t_s / 60.0, blur, np.array(wps), warnings)


def plan_ground_transect(sensor_rate_hz: float, speed_ms: float, n_rows: int, row_length_m: float,
                         target_spacing_m: float = 0.05, turn_time_s: float = 20.0) -> dict:
    """Ground-vehicle (phenomobile) planning: along-row sample spacing and duration."""
    spacing = speed_ms / sensor_rate_hz
    ok = spacing <= target_spacing_m
    return {"along_row_spacing_m": spacing, "meets_target": ok,
            "max_speed_for_target_ms": target_spacing_m * sensor_rate_hz,
            "duration_min": (n_rows * row_length_m / speed_ms + (n_rows - 1) * turn_time_s) / 60.0}


# --------------------------------------------------------------------------- sun


def solar_position(lat_deg: float, lon_deg: float, when_utc: datetime) -> tuple[float, float]:
    """Solar elevation and azimuth (degrees) — NOAA general solar position equations.

    Accuracy ~0.5° which is ample for planning acquisition windows.
    """
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    when_utc = when_utc.astimezone(timezone.utc)
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon_deg  # true solar time (min)
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(lat_deg)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180
    return 90 - math.degrees(zen), az % 360


def acquisition_window(lat_deg: float, lon_deg: float, day: date, utc_offset_h: float,
                       min_elevation_deg: float = 45.0, step_min: int = 5) -> dict:
    """Local-time window where solar elevation ≥ threshold (spectral/thermal best practice)."""
    tz = timezone(timedelta(hours=utc_offset_h))
    start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=tz)
    times, elev = [], []
    for k in range(0, 24 * 60, step_min):
        t = start + timedelta(minutes=k)
        e, _ = solar_position(lat_deg, lon_deg, t)
        times.append(t)
        elev.append(e)
    elev = np.array(elev)
    ok = np.where(elev >= min_elevation_deg)[0]
    noon = times[int(np.argmax(elev))]
    if len(ok) == 0:
        return {"solar_noon": noon, "max_elevation": float(elev.max()), "start": None, "end": None,
                "note": f"Sun never reaches {min_elevation_deg}°; use max-elevation ±1 h and log irradiance"}
    return {"solar_noon": noon, "max_elevation": float(elev.max()), "start": times[ok[0]], "end": times[ok[-1]]}


# --------------------------------------------------------------------------- protocols

DEFAULT_PROTOCOLS = {
    "uav_multispectral": {
        "required": ["trial_id", "date", "operator", "platform", "sensor", "altitude_m", "gsd_m",
                     "panel_before", "panel_after", "irradiance_sensor", "wind_ms", "cloud_cover_okta",
                     "solar_elevation_deg", "gnss_mode"],
        "rules": {"wind_ms": ("<=", 8.0), "cloud_cover_okta": ("<=", 2), "solar_elevation_deg": (">=", 35.0),
                  "panel_before": ("==", True), "panel_after": ("==", True)},
    },
    "uav_thermal": {
        "required": ["trial_id", "date", "operator", "sensor", "air_temp_c", "rh_pct", "wind_ms",
                     "wet_reference", "dry_reference", "sensor_warmup_min", "solar_elevation_deg"],
        "rules": {"sensor_warmup_min": (">=", 15), "wind_ms": ("<=", 5.0), "solar_elevation_deg": (">=", 40.0)},
    },
    "ground_vehicle": {
        "required": ["trial_id", "date", "operator", "platform", "speed_ms", "gnss_fix", "sensor_height_m"],
        "rules": {"speed_ms": ("<=", 1.5), "gnss_fix": ("in", ("RTK_FIXED",))},
    },
    "handheld_spectra": {
        "required": ["trial_id", "date", "operator", "instrument", "white_ref_interval_min", "fore_optic",
                     "height_above_canopy_m", "n_scans_per_plot"],
        "rules": {"white_ref_interval_min": ("<=", 15), "n_scans_per_plot": (">=", 3)},
    },
}


def validate_acquisition(meta: dict, protocol: str | dict) -> list[dict]:
    """Check an acquisition record against a protocol; returns a list of issues (empty = OK)."""
    proto = DEFAULT_PROTOCOLS[protocol] if isinstance(protocol, str) else protocol
    issues = []
    for k in proto["required"]:
        if k not in meta or meta[k] in (None, ""):
            issues.append({"field": k, "severity": "FAIL", "message": "missing required field"})
    ops = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b,
           "in": lambda a, b: a in b}
    for k, (op, ref) in proto.get("rules", {}).items():
        if k in meta and meta[k] is not None and not ops[op](meta[k], ref):
            issues.append({"field": k, "severity": "WARN" if op in ("<=", ">=") else "FAIL",
                           "message": f"{k}={meta[k]!r} violates {op} {ref!r}"})
    return issues
