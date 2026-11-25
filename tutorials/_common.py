"""Shared helpers for the phenokit tutorials (paths, figure style, result logging)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    # Warm up torch *before* any project module is imported. Torch's lazy imports (triggered on first
    # optimiser use) call os.path.realpath on every loaded module; on some Windows network shares this
    # raises WinError 59 for project files. One tiny optimisation step here performs those imports early.
    import torch

    _p = torch.nn.Parameter(torch.zeros(1))
    _o = torch.optim.AdamW([_p])
    torch.nn.functional.cross_entropy(torch.zeros(1, 2) + _p, torch.zeros(1, dtype=torch.long)).backward()
    _o.step()
    torch.optim.lr_scheduler.OneCycleLR(_o, 1e-3, total_steps=2)
    torch.optim.lr_scheduler.CosineAnnealingLR(_o, 2)
    del _p, _o
except ImportError:  # pragma: no cover - deep learning parts are optional
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public"
FIG = ROOT / "docs" / "chapter" / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

# colour-blind-safe categorical palette (Okabe–Ito)
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def savefig(fig, name: str):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  figure → {p.relative_to(ROOT)}")
    return p


def save_results(name: str, obj) -> Path:
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if hasattr(o, "to_dict"):
            return o.to_dict(orient="records") if hasattr(o, "columns") else o.to_dict()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    p = OUT / "tutorial_results" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conv(obj), indent=2, default=str), encoding="utf-8")
    print(f"  results → {p.relative_to(ROOT)}")
    return p


def banner(text: str):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)
