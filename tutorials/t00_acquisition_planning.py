"""Tutorial 00 — Planning acquisitions for handheld, ground and UAV platforms.

For a 34 × 66 m multi-crop trial (synthetic layout used in Tutorial 05) at a Northam-like
latitude in Western Australia, mid-August (cereal anthesis / canola flowering):

1. Solar window: when is the sun ≥45° (spectral) or near its maximum (thermal)?
2. UAV flight plans per sensor: GSD, line spacing, trigger interval, motion blur, duration.
3. Ground-vehicle transect: maximum speed to meet an along-row sampling target.
4. Protocol validation: turning an SOP into machine-checkable acquisition metadata.

Run:  python tutorials/t00_acquisition_planning.py
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from _common import OKABE, banner, plt, save_results, savefig

from phenokit import acquisition as acq

LAT, LON, UTC_OFFSET = -31.65, 116.67, 8.0


def main():
    banner("1. Solar window")
    rows = []
    for d in [date(2026, 6, 21), date(2026, 8, 20), date(2026, 9, 22), date(2026, 10, 20)]:
        w = acq.acquisition_window(LAT, LON, d, UTC_OFFSET, min_elevation_deg=45)
        rows.append({"date": d.isoformat(), "solar_noon": w["solar_noon"].strftime("%H:%M"),
                     "max_elev": round(w["max_elevation"], 1),
                     "≥45° from": w["start"].strftime("%H:%M") if w["start"] else "never",
                     "to": w["end"].strftime("%H:%M") if w["end"] else "-"})
    sun = pd.DataFrame(rows)
    print(sun.to_string(index=False))
    print("→ In mid-winter the sun never reaches 45° at this latitude: fly within ±1 h of solar noon, "
          "log irradiance (DLS) and image panels before *and* after each flight.")

    banner("2. UAV flight plans for the 34 × 66 m trial")
    plans = {}
    for sensor, alt, speed in [("rgb_20mp", 25, 4.0), ("ms_5band", 40, 5.0), ("vnir_pushbroom", 40, 3.0),
                               ("thermal_640", 40, 4.0), ("lidar_uav", 40, 5.0)]:
        s = acq.SENSOR_PRESETS[sensor]
        fp = acq.plan_flight(s, 34, 66, alt, speed_ms=speed, front_overlap=0.8, side_overlap=0.75 if s.kind != "lidar" else 0.5,
                             exposure_s=1 / 1000 if s.kind != "thermal" else 1 / 120, lidar_points_per_m2=200)
        plans[sensor] = fp
    tab = pd.DataFrame({k: {"alt_m": v.altitude_m, "gsd_cm": round(v.gsd_m * 100, 2) if np.isfinite(v.gsd_m) else None,
                            "lines": v.n_lines, "spacing_m": round(v.line_spacing_m, 1),
                            "trigger_s": round(v.trigger_interval_s, 3) if v.trigger_interval_s else None,
                            "blur_px": round(v.motion_blur_px, 2) if v.motion_blur_px is not None else None,
                            "minutes": round(v.flight_time_min, 1), "warnings": "; ".join(v.warnings)}
                        for k, v in plans.items()}).T
    print(tab.to_string())
    alt = acq.altitude_for_gsd(acq.SENSOR_PRESETS["rgb_20mp"], 0.005)
    print(f"RGB altitude for 5 mm GSD (plant counting): {alt:.1f} m")

    fig, ax = plt.subplots(figsize=(4.2, 5))
    wp = plans["ms_5band"].waypoints
    ax.plot(wp[:, 0], wp[:, 1], color=OKABE[4], lw=0.8)
    ax.add_patch(plt.Rectangle((0, 0), 34, 66, fill=False, ec=OKABE[5], lw=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Multispectral lawnmower plan, 40 m AGL")
    savefig(fig, "t00_flight_plan.png")

    banner("3. Ground vehicle transect")
    gv = acq.plan_ground_transect(sensor_rate_hz=10, speed_ms=1.0, n_rows=12, row_length_m=66, target_spacing_m=0.1)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in gv.items()})

    banner("4. Protocol validation")
    log = {"trial_id": "SYN-WA-2026", "date": "2026-08-20", "operator": "pilot-A", "platform": "uav", "sensor": "ms5",
           "altitude_m": 40, "gsd_m": 0.027, "panel_before": True, "panel_after": False, "irradiance_sensor": "DLS",
           "wind_ms": 9.1, "cloud_cover_okta": 1, "solar_elevation_deg": 41.0, "gnss_mode": "RTK"}
    issues = acq.validate_acquisition(log, "uav_multispectral")
    for i in issues:
        print(f"  {i['severity']}: {i['message']}")
    save_results("t00_planning", {"sun": sun, "flight_plans": tab.reset_index(), "ground": gv, "protocol_issues": issues})


if __name__ == "__main__":
    main()
