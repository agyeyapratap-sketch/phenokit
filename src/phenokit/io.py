"""Readers/writers for the sensor formats met on phenotyping platforms.

Dependency-light by design: ENVI (hyperspectral/thermal), MATLAB ``.mat`` cubes,
PNG/JPEG/TIFF images with ESRI world files, ASCII/NPZ point clouds and a minimal
pure-NumPy LAS 1.2 reader/writer. ``rasterio``/``laspy`` are used when available
for GeoTIFF/LAZ but are never required.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- geo


@dataclass
class GeoTransform:
    """North-up affine transform: map = origin + pixel * size.

    ``x0, y0`` are map coordinates of the *upper-left corner* of pixel (0, 0).
    ``px`` and ``py`` are positive pixel sizes in map units (metres).
    """

    x0: float = 0.0
    y0: float = 0.0
    px: float = 1.0
    py: float = 1.0
    epsg: int | None = None

    def pixel_to_map(self, row, col):
        row, col = np.asarray(row, float), np.asarray(col, float)
        return self.x0 + (col + 0.5) * self.px, self.y0 - (row + 0.5) * self.py

    def map_to_pixel(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (self.y0 - y) / self.py, (x - self.x0) / self.px  # fractional row, col

    def scaled(self, factor: float) -> "GeoTransform":
        """Transform for the same extent resampled by ``factor`` (e.g. 4 = 4x finer)."""
        return GeoTransform(self.x0, self.y0, self.px / factor, self.py / factor, self.epsg)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("x0", "y0", "px", "py", "epsg") if k in d})


# --------------------------------------------------------------------------- ENVI

_ENVI_DTYPES = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
                12: np.uint16, 13: np.uint32, 14: np.int64, 15: np.uint64}
_ENVI_CODES = {np.dtype(v): k for k, v in _ENVI_DTYPES.items()}


def parse_envi_header(path: str | Path) -> dict:
    """Parse an ENVI ``.hdr`` file into a dict (lists for ``{...}`` values)."""
    text = Path(path).read_text(errors="replace")
    if not text.lstrip().upper().startswith("ENVI"):
        raise ValueError(f"{path} is not an ENVI header (missing 'ENVI' magic)")
    meta: dict = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, re.M):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            items = [v.strip() for v in val[1:-1].replace("\n", " ").split(",")]
            meta[key] = [v for v in items if v != ""]
        else:
            meta[key] = val
    for k in ("samples", "lines", "bands", "data type", "byte order", "header offset"):
        if k in meta:
            meta[k] = int(meta[k])
    for k in ("wavelength", "fwhm", "data gain values", "data offset values"):
        if k in meta:
            meta[k] = [float(v) for v in meta[k]]
    return meta


def _envi_data_file(hdr: Path) -> Path:
    for ext in ("", ".raw", ".img", ".dat", ".bsq", ".bil", ".bip"):
        cand = hdr.with_suffix(ext) if ext else hdr.with_suffix("")
        if cand.exists() and cand.suffix.lower() != ".hdr":
            return cand
    raise FileNotFoundError(f"No binary data file found next to {hdr}")


def read_envi(hdr_path: str | Path, mmap: bool = False):
    """Read an ENVI image. Returns ``(cube[H, W, B], meta)``.

    ``meta['wavelength']`` (nm) and ``meta['geotransform']`` are populated when present.
    """
    hdr = Path(hdr_path)
    meta = parse_envi_header(hdr)
    H, W, B = meta["lines"], meta["samples"], meta["bands"]
    dtype = np.dtype(_ENVI_DTYPES[meta["data type"]])
    dtype = dtype.newbyteorder(">" if meta.get("byte order", 0) == 1 else "<")
    data_file = _envi_data_file(hdr)
    expected = H * W * B * dtype.itemsize + meta.get("header offset", 0)
    actual = data_file.stat().st_size
    if actual < expected:
        raise ValueError(f"{data_file}: truncated ({actual} bytes < expected {expected})")
    raw = np.memmap(data_file, dtype=dtype, mode="r", offset=meta.get("header offset", 0),
                    shape=(H * W * B,)) if mmap else np.fromfile(
        data_file, dtype=dtype, count=H * W * B, offset=meta.get("header offset", 0))
    il = meta.get("interleave", "bsq").lower()
    if il == "bsq":
        cube = raw.reshape(B, H, W).transpose(1, 2, 0)
    elif il == "bil":
        cube = raw.reshape(H, B, W).transpose(0, 2, 1)
    elif il == "bip":
        cube = raw.reshape(H, W, B)
    else:
        raise ValueError(f"Unknown interleave {il}")
    if "map info" in meta:
        mi = meta["map info"]
        # map info = {proj, refpx_x, refpx_y, easting, northing, px, py, ...}
        try:
            rx, ry = float(mi[1]) - 1.0, float(mi[2]) - 1.0  # ENVI reference pixel is 1-based
            px, py = float(mi[5]), float(mi[6])
            x0, y0 = float(mi[3]) - rx * px, float(mi[4]) + ry * py
            epsg = None
            m = re.search(r"epsg\s*=\s*(\d+)", " ".join(meta.get("coordinate system string", [])) if isinstance(
                meta.get("coordinate system string"), list) else str(meta.get("coordinate system string", "")), re.I)
            if m:
                epsg = int(m.group(1))
            if "phenokit epsg" in meta:
                epsg = int(meta["phenokit epsg"])
            meta["geotransform"] = GeoTransform(x0, y0, px, py, epsg)
        except (IndexError, ValueError):
            pass
    return (cube if mmap else np.ascontiguousarray(cube)), meta


def write_envi(base_path: str | Path, cube: np.ndarray, wavelengths=None, fwhm=None,
               transform: GeoTransform | None = None, interleave: str = "bsq",
               description: str = "phenokit export", extra: dict | None = None) -> Path:
    """Write ``cube[H, W, B]`` as ENVI ``<base>.raw`` + ``<base>.hdr``. Returns header path."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    if cube.ndim == 2:
        cube = cube[..., None]
    H, W, B = cube.shape
    if cube.dtype not in _ENVI_CODES:
        cube = cube.astype(np.float32)
    data = {"bsq": cube.transpose(2, 0, 1), "bil": cube.transpose(0, 2, 1), "bip": cube}[interleave]
    np.ascontiguousarray(data).astype(cube.dtype.newbyteorder("<")).tofile(base.with_suffix(".raw"))
    lines = ["ENVI", f"description = {{{description}}}", f"samples = {W}", f"lines = {H}",
             f"bands = {B}", "header offset = 0", "file type = ENVI Standard",
             f"data type = {_ENVI_CODES[cube.dtype]}", f"interleave = {interleave}", "byte order = 0"]
    if wavelengths is not None:
        lines += ["wavelength units = Nanometers",
                  "wavelength = {" + ", ".join(f"{w:.3f}" for w in wavelengths) + "}"]
    if fwhm is not None:
        lines.append("fwhm = {" + ", ".join(f"{w:.3f}" for w in fwhm) + "}")
    if transform is not None:
        lines.append(f"map info = {{Arbitrary, 1.0, 1.0, {transform.x0:.4f}, {transform.y0:.4f}, "
                     f"{transform.px:.6f}, {transform.py:.6f}, units=Meters}}")
        if transform.epsg:
            lines.append(f"phenokit epsg = {transform.epsg}")
    for k, v in (extra or {}).items():
        lines.append(f"{k} = {{{', '.join(map(str, v))}}}" if isinstance(v, (list, tuple)) else f"{k} = {v}")
    hdr = base.with_suffix(".hdr")
    hdr.write_text("\n".join(lines) + "\n")
    return hdr


# --------------------------------------------------------------------------- MATLAB cubes


def read_mat_cube(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load the (largest) array from a MATLAB ``.mat`` file, e.g. Indian Pines / Salinas."""
    from scipy.io import loadmat

    d = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    if key is not None:
        return np.asarray(d[key])
    return np.asarray(max(d.values(), key=lambda a: np.asarray(a).size))


# --------------------------------------------------------------------------- images


def read_image(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im)


def write_worldfile(image_path: str | Path, transform: GeoTransform) -> Path:
    """Write an ESRI world file (.pgw/.jgw/.tfw) so GIS software can place the image."""
    p = Path(image_path)
    ext = {".png": ".pgw", ".jpg": ".jgw", ".jpeg": ".jgw", ".tif": ".tfw", ".tiff": ".tfw"}[p.suffix.lower()]
    wf = p.with_suffix(ext)
    # world file references the *centre* of the upper-left pixel
    wf.write_text("\n".join(f"{v:.6f}" for v in (
        transform.px, 0.0, 0.0, -transform.py, transform.x0 + transform.px / 2, transform.y0 - transform.py / 2)) + "\n")
    if transform.epsg:
        p.with_suffix(p.suffix + ".epsg").write_text(str(transform.epsg))
    return wf


def read_worldfile(image_path: str | Path) -> GeoTransform:
    p = Path(image_path)
    ext = {".png": ".pgw", ".jpg": ".jgw", ".jpeg": ".jgw", ".tif": ".tfw", ".tiff": ".tfw"}[p.suffix.lower()]
    a, _, _, e, c, f = [float(v) for v in p.with_suffix(ext).read_text().split()]
    epsg_file = p.with_suffix(p.suffix + ".epsg")
    epsg = int(epsg_file.read_text()) if epsg_file.exists() else None
    return GeoTransform(c - a / 2, f - e / 2, a, -e, epsg)


def write_image(path: str | Path, img: np.ndarray, transform: GeoTransform | None = None) -> Path:
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(p)
    if transform is not None:
        write_worldfile(p, transform)
    return p


# --------------------------------------------------------------------------- point clouds


def read_xyz(path: str | Path, usecols=None) -> np.ndarray:
    """Read whitespace/comma-delimited point cloud text (x y z [extra...])."""
    try:
        import pandas as pd

        df = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python" if usecols is None else "c",
                         comment="#", usecols=usecols)
        return df.to_numpy(dtype=np.float64)
    except Exception:  # pragma: no cover - fallback
        return np.loadtxt(path, usecols=usecols)


_LAS_HEADER = struct.Struct("<4sHHIHH8sBB32s32sHHHIIBHIIIIII3d3d6d")  # LAS 1.2 public header (227 bytes)


def write_las(path: str | Path, xyz: np.ndarray, intensity=None, classification=None,
              return_number=None, scale=0.001) -> Path:
    """Write a minimal LAS 1.2 file (point format 0). Pure NumPy, no laspy required."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    mins, maxs = xyz.min(0), xyz.max(0)
    offs = np.floor(mins)
    rec = np.zeros(n, dtype=[("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"),
                             ("flags", "u1"), ("cls", "u1"), ("scan_angle", "i1"), ("user", "u1"), ("src", "<u2")])
    for i, k in enumerate("XYZ"):
        rec[k] = np.round((xyz[:, i] - offs[i]) / scale).astype(np.int32)
    if intensity is not None:
        rec["intensity"] = np.asarray(intensity, np.uint16)
    rn = np.ones(n, np.uint8) if return_number is None else np.asarray(return_number, np.uint8)
    rec["flags"] = (rn & 0b111) | (np.maximum(rn, 1) & 0b111) << 3
    if classification is not None:
        rec["cls"] = np.asarray(classification, np.uint8)
    by_return = [int((rn == r).sum()) for r in range(1, 6)]
    header = _LAS_HEADER.pack(
        b"LASF", 0, 0, 0, 0, 0, b"\0" * 8, 1, 2, b"phenokit".ljust(32, b"\0"), b"phenokit".ljust(32, b"\0"),
        1, 2026, 227, 227, 0, 0, 20, n, *by_return, scale, scale, scale, *offs,
        maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())
    return p


def read_las(path: str | Path) -> dict:
    """Read LAS 1.0–1.4 point formats 0–3 (uncompressed). Uses laspy if installed (for LAZ)."""
    p = Path(path)
    if p.suffix.lower() == ".laz":
        import laspy  # noqa: F401 - informative ImportError for LAZ

        las = laspy.read(p)
        return {"xyz": np.vstack([las.x, las.y, las.z]).T, "intensity": np.asarray(las.intensity),
                "classification": np.asarray(las.classification), "return_number": np.asarray(las.return_number)}
    with open(p, "rb") as f:
        buf = f.read()
    if buf[:4] != b"LASF":
        raise ValueError(f"{p} is not a LAS file")
    offset_to_points = struct.unpack_from("<I", buf, 96)[0]
    fmt = buf[104] & 0x3F
    rec_len = struct.unpack_from("<H", buf, 105)[0]
    n = struct.unpack_from("<I", buf, 107)[0]
    scale = np.array(struct.unpack_from("<3d", buf, 131))
    offs = np.array(struct.unpack_from("<3d", buf, 155))
    if fmt > 3:
        raise NotImplementedError("Point formats >3 need laspy")
    raw = np.frombuffer(buf, dtype=np.uint8, count=n * rec_len, offset=offset_to_points).reshape(n, rec_len)
    ints = raw[:, :12].copy().view("<i4").reshape(n, 3)
    xyz = ints * scale + offs
    flags = raw[:, 14]
    return {"xyz": xyz, "intensity": raw[:, 12:14].copy().view("<u2").ravel(),
            "classification": raw[:, 15] & 0x1F, "return_number": flags & 0b111}


# --------------------------------------------------------------------------- spectra (handheld)


def read_spectrum_csv(path: str | Path):
    """Read a handheld spectroradiometer export with columns ``wavelength, <sample...>``.

    Returns ``(wavelengths[nm], spectra[n_samples, n_bands], sample_names)``.
    """
    import pandas as pd

    df = pd.read_csv(path)
    wl = df.iloc[:, 0].to_numpy(float)
    return wl, df.iloc[:, 1:].to_numpy(float).T, list(df.columns[1:])


def write_json(path: str | Path, obj) -> Path:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, GeoTransform):
            return o.to_dict()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=default))
    return p
