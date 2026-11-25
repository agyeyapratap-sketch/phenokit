"""Command-line interface: ``phenokit <command> ...`` (see ``phenokit -h``)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _print_json(obj):
    print(json.dumps(obj, indent=2, default=str))


def cmd_datasets(a):
    from . import datasets

    if a.action == "list":
        print(datasets.describe())
    else:
        for k in a.keys:
            print(datasets.fetch(k, a.root))


def cmd_synth(a):
    from .synthetic import TrialConfig, generate_trial

    r = generate_trial(a.root, TrialConfig(seed=a.seed, inject_faults=not a.clean))
    print(f"trial written to {r['root']} ({len(r['faults'])} injected faults)")


def cmd_plan(a):
    from . import acquisition as acq

    s = acq.SENSOR_PRESETS[a.sensor]
    fp = acq.plan_flight(s, a.width, a.length, a.altitude, a.speed, a.front, a.side)
    _print_json(fp.summary())


def cmd_sun(a):
    from . import acquisition as acq

    w = acq.acquisition_window(a.lat, a.lon, date.fromisoformat(a.date), a.utc_offset, a.min_elevation)
    _print_json({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in w.items()})


def cmd_manifest(a):
    from . import catalog

    if a.action == "build":
        print(catalog.build_manifest(a.dir))
    else:
        r = catalog.verify_manifest(a.dir)
        _print_json(r)
        sys.exit(0 if r["ok"] else 2)


def cmd_qc(a):
    import numpy as np

    from . import io as pio, qc

    rep = qc.QCReport(f"phenokit QC ({a.kind})")
    if a.kind == "rgb":
        hashes = {}
        for f in a.paths:
            img = pio.read_image(f)
            rep.extend(qc.qc_rgb(img, target=Path(f).name))
            hashes[Path(f).name] = qc.average_hash(img)
        d = qc.find_near_duplicates(hashes)
        rep.extend([qc.QCResult("frames", "near_duplicates", qc.WARN if d else qc.PASS, len(d), "0", str(d))])
    elif a.kind == "envi":
        for f in a.paths:
            cube, meta = pio.read_envi(f)
            wl = meta.get("wavelength")
            bad = qc.detect_bad_bands(cube, wl)
            rep.extend([qc.QCResult(Path(f).name, "bad_bands", qc.WARN if bad["bad_bands"] else qc.PASS,
                                    len(bad["bad_bands"]), "", str(bad["reasons"]))])
            if meta.get("bands") == 1:
                rep.extend(qc.qc_thermal(cube[..., 0], target=Path(f).name))
    elif a.kind == "las":
        for f in a.paths:
            rep.extend(qc.qc_pointcloud(pio.read_las(f)["xyz"], target=Path(f).name))
    elif a.kind == "spectra":
        for f in a.paths:
            wl, s, _ = pio.read_spectrum_csv(f)
            rep.extend(qc.qc_spectrum_splice(wl, s, target=Path(f).name))
            rep.extend(qc.qc_reflectance(s, target=Path(f).name))
    out = rep.to_html(a.out)
    print(f"{rep.summary()} → {out}")
    sys.exit(2 if rep.summary()["overall"] == qc.FAIL and a.strict else 0)


def cmd_run(a):
    from . import pipeline

    cfg = pipeline.load_config(a.config)
    r = pipeline.run(cfg, a.out)
    print(r["qc_report"].summary())


def main(argv=None):
    p = argparse.ArgumentParser(prog="phenokit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("datasets", help="list or fetch public tutorial datasets")
    s.add_argument("action", choices=["list", "fetch"])
    s.add_argument("keys", nargs="*", default=[])
    s.add_argument("--root", default="data/public")
    s.set_defaults(func=cmd_datasets)

    s = sub.add_parser("synth", help="generate the synthetic multi-platform trial")
    s.add_argument("root")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--clean", action="store_true", help="do not inject faults")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("plan", help="UAV flight plan for a rectangular trial")
    s.add_argument("--sensor", default="ms_5band")
    s.add_argument("--width", type=float, required=True)
    s.add_argument("--length", type=float, required=True)
    s.add_argument("--altitude", type=float, default=40)
    s.add_argument("--speed", type=float, default=5)
    s.add_argument("--front", type=float, default=0.8)
    s.add_argument("--side", type=float, default=0.75)
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("sun", help="solar acquisition window")
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--date", required=True)
    s.add_argument("--utc-offset", type=float, required=True)
    s.add_argument("--min-elevation", type=float, default=45)
    s.set_defaults(func=cmd_sun)

    s = sub.add_parser("manifest", help="build/verify SHA-256 manifests")
    s.add_argument("action", choices=["build", "verify"])
    s.add_argument("dir")
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("qc", help="quick QC of files → HTML report")
    s.add_argument("kind", choices=["rgb", "envi", "las", "spectra"])
    s.add_argument("paths", nargs="+")
    s.add_argument("--out", default="qc_report.html")
    s.add_argument("--strict", action="store_true", help="exit code 2 if any FAIL")
    s.set_defaults(func=cmd_qc)

    s = sub.add_parser("run", help="run the trial pipeline from a YAML config")
    s.add_argument("config")
    s.add_argument("--out", default="outputs/run")
    s.set_defaults(func=cmd_run)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
