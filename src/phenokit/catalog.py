"""Data management: standard layout, MIAPPE-style metadata, checksums and a SQLite catalogue.

Design principles (FAIR):
* every file has a SHA-256 in a manifest written *at ingest* (before any processing);
* every dataset has a machine-readable metadata sidecar (``metadata.json``);
* every processing step is recorded as provenance (inputs → outputs, parameters, software version);
* raw data are immutable — derived products go to sibling ``calibrated/`` and ``products/`` folders.
"""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import platform as _platform
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

LEVELS = ("raw", "calibrated", "products", "qc")


def dataset_path(root: str | Path, trial_id: str, date: str, platform: str, sensor: str,
                 level: str = "raw") -> Path:
    """``<root>/<trial>/<YYYYMMDD>/<platform>_<sensor>/<level>`` — one folder per acquisition."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    return Path(root) / trial_id / date.replace("-", "") / f"{platform}_{sensor}" / level


# --------------------------------------------------------------------------- metadata


@dataclass
class AcquisitionMetadata:
    """Minimal MIAPPE-aligned record for one sensor acquisition."""

    investigation: str
    study: str                      # e.g. trial ID
    site_name: str
    latitude: float
    longitude: float
    crop_species: list[str]
    platform: str                   # handheld | ground | uav
    sensor: str
    acquisition_datetime: str       # ISO-8601 with offset
    operator: str
    processing_level: str = "raw"
    growth_stage: str | None = None  # e.g. Zadoks Z31
    plot_layout_file: str | None = None
    weather: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    licence: str = "CC-BY-4.0"
    notes: str = ""

    REQUIRED = ("investigation", "study", "site_name", "latitude", "longitude", "crop_species", "platform",
                "sensor", "acquisition_datetime", "operator")

    def validate(self) -> list[str]:
        errs = [f"missing {k}" for k in self.REQUIRED if getattr(self, k) in (None, "", [])]
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            errs.append("latitude/longitude out of range")
        try:
            dt = datetime.fromisoformat(self.acquisition_datetime)
            if dt.tzinfo is None:
                errs.append("acquisition_datetime lacks a UTC offset (ambiguous time)")
        except ValueError:
            errs.append("acquisition_datetime is not ISO-8601")
        if self.platform not in ("handheld", "ground", "uav"):
            errs.append("platform must be handheld|ground|uav")
        return errs

    def save(self, folder: str | Path) -> Path:
        p = Path(folder) / "metadata.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AcquisitionMetadata":
        return cls(**json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- checksums


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


MANIFEST_NAME = "MANIFEST.sha256.csv"


def build_manifest(root: str | Path, exclude: tuple[str, ...] = (MANIFEST_NAME,)) -> Path:
    """Hash every file under ``root`` and write ``MANIFEST.sha256.csv`` (path,size,sha256)."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in exclude:
            rows.append((p.relative_to(root).as_posix(), p.stat().st_size, sha256_file(p)))
    out = root / MANIFEST_NAME
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "sha256"])
        w.writerows(rows)
    return out


def verify_manifest(root: str | Path, manifest: str | Path | None = None) -> dict:
    """Compare files on disk with the manifest. Returns ``{ok, missing, modified, extra}``."""
    root = Path(root)
    manifest = Path(manifest) if manifest else root / MANIFEST_NAME
    with open(manifest, newline="") as f:
        expected = {r["path"]: r for r in csv.DictReader(f)}
    present = {p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and p.name != manifest.name}
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    modified = []
    for rel in sorted(set(expected) & present):
        p = root / rel
        if p.stat().st_size != int(expected[rel]["size"]) or sha256_file(p) != expected[rel]["sha256"]:
            modified.append(rel)
    return {"ok": not (missing or modified), "n_expected": len(expected), "missing": missing,
            "modified": modified, "extra": extra}


# --------------------------------------------------------------------------- catalogue

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY, trial_id TEXT, date TEXT, platform TEXT, sensor TEXT, level TEXT,
  path TEXT UNIQUE, metadata_json TEXT, registered_utc TEXT);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY, dataset_id INTEGER REFERENCES datasets(id), relpath TEXT, size INTEGER,
  sha256 TEXT, UNIQUE(dataset_id, relpath));
CREATE TABLE IF NOT EXISTS qc_results (
  id INTEGER PRIMARY KEY, dataset_id INTEGER REFERENCES datasets(id), target TEXT, check_name TEXT,
  status TEXT, value REAL, threshold TEXT, message TEXT, run_utc TEXT);
CREATE TABLE IF NOT EXISTS provenance (
  id INTEGER PRIMARY KEY, step TEXT, inputs TEXT, outputs TEXT, parameters TEXT, software TEXT,
  user TEXT, host TEXT, run_utc TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DataCatalog:
    """Lightweight SQLite catalogue: what data exist, are they intact, did they pass QC, how were they made."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(_SCHEMA)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def register_dataset(self, path: str | Path, trial_id: str, date: str, platform: str, sensor: str,
                         level: str = "raw", metadata: dict | None = None, hash_files: bool = True) -> int:
        path = Path(path)
        cur = self.con.execute(
            "INSERT INTO datasets(trial_id,date,platform,sensor,level,path,metadata_json,registered_utc) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET metadata_json=excluded.metadata_json "
            "RETURNING id",
            (trial_id, date, platform, sensor, level, str(path.resolve()), json.dumps(metadata or {}), _now()))
        ds_id = cur.fetchone()[0]
        if hash_files:
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    self.con.execute("INSERT OR REPLACE INTO files(dataset_id,relpath,size,sha256) VALUES(?,?,?,?)",
                                     (ds_id, p.relative_to(path).as_posix(), p.stat().st_size, sha256_file(p)))
        self.con.commit()
        return ds_id

    def record_qc(self, dataset_id: int | None, results) -> None:
        """Store QC results (iterable of :class:`phenokit.qc.QCResult`)."""
        run = _now()
        self.con.executemany(
            "INSERT INTO qc_results(dataset_id,target,check_name,status,value,threshold,message,run_utc) "
            "VALUES(?,?,?,?,?,?,?,?)",
            [(dataset_id, r.target, r.check, r.status, None if r.value is None else float(r.value),
              str(r.threshold), r.message, run) for r in results])
        self.con.commit()

    def record_step(self, step: str, inputs, outputs, parameters: dict | None = None) -> None:
        self.con.execute(
            "INSERT INTO provenance(step,inputs,outputs,parameters,software,user,host,run_utc) VALUES(?,?,?,?,?,?,?,?)",
            (step, json.dumps([str(i) for i in inputs]), json.dumps([str(o) for o in outputs]),
             json.dumps(parameters or {}, default=str), f"phenokit {__version__}; python {_platform.python_version()}",
             _safe_user(), _platform.node(), _now()))
        self.con.commit()

    def query(self, sql: str, params=()):
        import pandas as pd

        return pd.read_sql_query(sql, self.con, params=params)

    def verify_dataset(self, dataset_id: int) -> dict:
        """Re-hash files of a registered dataset against the catalogue."""
        path = Path(self.con.execute("SELECT path FROM datasets WHERE id=?", (dataset_id,)).fetchone()[0])
        bad, missing = [], []
        for rel, size, sha in self.con.execute("SELECT relpath,size,sha256 FROM files WHERE dataset_id=?",
                                               (dataset_id,)):
            p = path / rel
            if not p.exists():
                missing.append(rel)
            elif p.stat().st_size != size or sha256_file(p) != sha:
                bad.append(rel)
        return {"ok": not (bad or missing), "modified": bad, "missing": missing}


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover
        return os.environ.get("USERNAME", "unknown")
