"""Verified access to public phenotyping / crop remote-sensing datasets used in the tutorials.

Every file is pinned by SHA-256: a download that does not match is rejected, which
protects the tutorials (and your own pipelines) against silent upstream changes, partial
downloads and tampered mirrors. Licences differ per dataset — read ``LICENCE_NOTE`` and the
original source before redistributing data or results.
"""
from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import sha256_file

UA = {"User-Agent": "phenokit-dataset-fetcher/0.1"}


@dataclass
class RemoteFile:
    name: str
    urls: list[str]
    sha256: str | None
    size: int | None = None
    zip_member: str | None = None  # extract this member from a remote zip using HTTP range requests


@dataclass
class Dataset:
    key: str
    title: str
    modality: str
    crops: str
    citation: str
    source_page: str
    licence_note: str
    files: list[RemoteFile] = field(default_factory=list)
    archive: RemoteFile | None = None  # whole-archive download (zip)


_HYB = "https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/"
_EHU = "https://www.ehu.eus/ccwintco/uploads/"
_P4D = "https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip"

REGISTRY: dict[str, Dataset] = {
    "indian_pines": Dataset(
        "indian_pines", "AVIRIS Indian Pines (corrected, 200 bands)", "airborne hyperspectral 400–2500 nm, 20 m",
        "corn, soybean, wheat, alfalfa, oats, grass/pasture, woods (16 classes)",
        "Baumgardner, M. F., Biehl, L. L., & Landgrebe, D. A. (2015). 220 Band AVIRIS Hyperspectral Image Data Set: "
        "June 12, 1992 Indian Pine Test Site 3. Purdue University Research Repository. doi:10.4231/R7RX991C",
        "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        "Distributed for research use; cite the Purdue record. Mirror: HybridSN repository.",
        [RemoteFile("Indian_pines_corrected.mat", [_HYB + "Indian_pines_corrected.mat",
                                                   _EHU + "6/67/Indian_pines_corrected.mat"],
                    "PIN"),
         RemoteFile("Indian_pines_gt.mat", [_HYB + "Indian_pines_gt.mat", _EHU + "c/c4/Indian_pines_gt.mat"], "PIN")]),
    "salinas": Dataset(
        "salinas", "AVIRIS Salinas Valley (corrected, 204 bands)", "airborne hyperspectral 400–2500 nm, 3.7 m",
        "broccoli, lettuce (4 growth stages), celery, vineyard, fallow, stubble, corn (16 classes)",
        "AVIRIS Salinas Valley scene (NASA JPL, 1998) as distributed by the GIC, University of the Basque Country.",
        "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        "Distributed for research use via the GIC (UPV/EHU) page. Mirror: HybridSN repository.",
        [RemoteFile("Salinas_corrected.mat", [_HYB + "Salinas_corrected.mat", _EHU + "a/a3/Salinas_corrected.mat"],
                    "PIN"),
         RemoteFile("Salinas_gt.mat", [_HYB + "Salinas_gt.mat", _EHU + "f/fa/Salinas_gt.mat"], "PIN")]),
    "cwfid": Dataset(
        "cwfid", "Crop/Weed Field Image Dataset (CWFID)", "ground-vehicle close-range field images + masks",
        "organic carrot (crop) with weeds; 60 images, vegetation masks, crop/weed annotations",
        "Haug, S., & Ostermann, J. (2015). A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision "
        "Based Precision Agriculture Tasks. Computer Vision – ECCV 2014 Workshops, 105–116. "
        "doi:10.1007/978-3-319-16220-1_8",
        "https://github.com/cwfid/dataset",
        "Copyrighted; non-commercial research use only; cite the paper.",
        archive=RemoteFile("cwfid-v1.0.zip", ["https://codeload.github.com/cwfid/dataset/zip/refs/tags/v1.0"], "PIN")),
    "pheno4d": Dataset(
        "pheno4d", "Pheno4D maize/tomato spatio-temporal point clouds", "laser-scanned 3-D point clouds (lab)",
        "maize (7 plants) and tomato (7 plants), ~daily scans, organ-level labels on a subset",
        "Schunck, D., Magistri, F., Rosu, R. A., Cornelißen, A., Chebrolu, N., Paulus, S., Léon, J., Behnke, S., "
        "Stachniss, C., Kuhlmann, H., & Klingbeil, L. (2021). Pheno4D: A spatio-temporal dataset of maize and tomato "
        "plant point clouds for phenotyping and advanced plant analysis. PLoS ONE 16(8): e0256340. "
        "doi:10.1371/journal.pone.0256340",
        "https://www.ipb.uni-bonn.de/data/pheno4d/",
        "See the dataset page for licence terms. The full archive is 12.7 GB; phenokit extracts single scans "
        "by HTTP range requests.",
        [RemoteFile(m.split("/")[-1], [_P4D], "PIN", zip_member=m) for m in (
            "Maize01/M01_0313_a.txt", "Maize01/M01_0315_a.txt", "Maize01/M01_0317_a.txt",
            "Maize01/M01_0319_a.txt", "Maize01/M01_0321_a.txt", "Maize01/M01_0325_a.txt")]),
}

# Pinned SHA-256 values (filled from verified first downloads; see tutorials/README.md)
PINNED_SHA256: dict[str, str] = {
    "indian_pines/Indian_pines_corrected.mat": "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939",
    "indian_pines/Indian_pines_gt.mat": "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c",
    "salinas/Salinas_corrected.mat": "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
    "salinas/Salinas_gt.mat": "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
    "cwfid/cwfid-v1.0.zip": "3dfbdf638beb9f1f6b8ba16aee31c1b95dfa1f2524f5b09b89f37fc1dc3db1ab",
    "pheno4d/M01_0313_a.txt": "16597460469aa54b2082d27499bd157abe36c02901a50a5f518cb189df0a448d",
    "pheno4d/M01_0315_a.txt": "eb45055d2ef325e9a10b422c06becf96cad048f53f48e80cedb347aae6eac538",
    "pheno4d/M01_0317_a.txt": "abbc2d0dac5e55d122d4bff5bbcbc637a2c3e53b5c0218bb90f2d32427a501ea",
    "pheno4d/M01_0319_a.txt": "b504c1fa58ab5c68c18b4a7653e8ef1953bd862e44bdc9d409738af6501bd096",
    "pheno4d/M01_0321_a.txt": "a977f2f3a202a0f18b4cb2065a4076e8cf18b0cf411497c636cf7e7d39209b97",
    "pheno4d/M01_0325_a.txt": "486efc1c96dd56d1697e1e554497672774f759c4846154f2613d4f98ea99c874",
}


class ChecksumError(RuntimeError):
    pass


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only file over HTTP using Range requests (lets ``zipfile`` read one member
    of a multi-GB remote archive without downloading the rest)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.pos, self.block = url, 0, block
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise IOError("server does not support range requests")
            self.size = int(r.headers["Content-Length"])
        self._cache_start, self._cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        cs = self._cache_start
        if not (cs <= self.pos and end <= cs + len(self._cache)):
            fetch_end = min(max(end, self.pos + self.block), self.size)
            req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{fetch_end - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self._cache = r.read()
            self._cache_start = cs = self.pos
        out = self._cache[self.pos - cs:end - cs]
        self.pos += len(out)
        return out


def _verify(path: Path, f: RemoteFile, key: str) -> str:
    digest = sha256_file(path)
    pinned = PINNED_SHA256.get(f"{key}/{f.name}") or (f.sha256 if f.sha256 not in (None, "PIN") else None)
    if pinned and digest != pinned:
        path.rename(path.with_suffix(path.suffix + ".rejected"))
        raise ChecksumError(f"{f.name}: sha256 {digest} != pinned {pinned}")
    return digest


def fetch(key: str, root: str | Path = "data/public", members: list[str] | None = None,
          verbose: bool = True) -> Path:
    """Download (or reuse) a registered dataset into ``<root>/<key>/`` and verify checksums."""
    ds = REGISTRY[key]
    out = Path(root) / key
    out.mkdir(parents=True, exist_ok=True)
    files = ([ds.archive] if ds.archive else []) + [f for f in ds.files if members is None or f.name in members]
    for f in files:
        dest = out / f.name
        if not dest.exists():
            last = None
            for url in f.urls:
                try:
                    if verbose:
                        print(f"[phenokit] fetching {f.name} from {url}" + (f" :: {f.zip_member}" if f.zip_member else ""))
                    if f.zip_member:
                        with zipfile.ZipFile(HTTPRangeReader(url)) as z, z.open(f.zip_member) as src, \
                                open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1 << 20)
                    else:
                        _download(url, dest)
                    break
                except (OSError, zipfile.BadZipFile) as e:  # try next mirror
                    last = e
                    if dest.exists():
                        dest.unlink()
            else:
                raise IOError(f"all mirrors failed for {f.name}: {last}")
        digest = _verify(dest, f, key)
        if verbose:
            print(f"[phenokit] {f.name}: sha256 {digest[:16]}… OK")
        if ds.archive is f and f.name.endswith(".zip"):
            marker = out / ".extracted"
            if not marker.exists():
                with zipfile.ZipFile(dest) as z:
                    for m in z.infolist():  # zip-slip guard
                        target = (out / m.filename).resolve()
                        if not str(target).startswith(str(out.resolve())):
                            raise IOError(f"unsafe path in archive: {m.filename}")
                    z.extractall(out)
                marker.write_text("ok")
    return out


def describe(key: str | None = None) -> str:
    keys = [key] if key else list(REGISTRY)
    lines = []
    for k in keys:
        d = REGISTRY[k]
        lines += [f"{k}: {d.title}", f"  modality: {d.modality}", f"  crops: {d.crops}", f"  cite: {d.citation}",
                  f"  source: {d.source_page}", f"  licence: {d.licence_note}", ""]
    return "\n".join(lines)
