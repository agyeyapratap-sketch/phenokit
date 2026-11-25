"""Tutorial 01 — Hyperspectral data QC and leakage-aware crop classification (AVIRIS Indian Pines).

What you will learn
-------------------
1. Fetch a public dataset with checksum verification and inspect it.
2. Quality-check an imaging-spectroscopy cube: per-band SNR (spectral–spatial decorrelation),
   bad/absorption bands, spectral spikes.
3. Build per-pixel crop classifiers (random forest, SVM, 1-D CNN).
4. Quantify *spatial leakage*: random pixel cross-validation vs spatially blocked CV.

Run:  python tutorials/t01_hyperspectral_qc_and_leakage_indian_pines.py [--no-cnn]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from _common import DATA, OKABE, banner, plt, save_results, savefig

from phenokit import datasets, io as pio, ml, qc, spectral

CLASSES = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
           "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
           "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]


def approx_wavelengths():
    """Approximate AVIRIS-1992 band centres for the 200-band 'corrected' cube.

    The 220-band product spans ~400–2500 nm (~10 nm sampling); bands 104–108, 150–163 and 220
    (1-based) were removed for water absorption. Exact centres are in the original
    MultiSpec/Purdue metadata — use those for quantitative work.
    """
    wl = np.linspace(400, 2500, 220)
    drop = list(range(103, 108)) + list(range(149, 163)) + [219]
    return np.delete(wl, drop)


def main(run_cnn: bool = True):
    banner("1. Fetch & load (SHA-256 verified)")
    root = datasets.fetch("indian_pines", DATA)
    cube = pio.read_mat_cube(root / "Indian_pines_corrected.mat").astype(np.float32)
    gt = pio.read_mat_cube(root / "Indian_pines_gt.mat").astype(int)
    wl = approx_wavelengths()
    H, W, B = cube.shape
    print(f"cube {cube.shape}, labelled pixels {int((gt > 0).sum())}, classes {gt.max()}")

    banner("2. Quality control")
    snr = qc.band_snr(cube)
    bad = qc.detect_bad_bands(cube, wl, snr_min=15, known_absorption=True)
    spikes = qc.detect_spatial_spikes(cube, z=10)
    print(f"median SNR {np.median(snr):.0f}; bands with SNR<15: {[round(w) for w in wl[snr < 15]]}")
    print(f"bands inside residual water-absorption windows: {[round(wl[b]) for b in bad['bad_bands'] if snr[b] >= 15]}")
    print(f"impulse-noise rate {spikes.mean() * 100:.4f}% of samples "
          f"(bands most affected: {[round(w) for w in wl[np.argsort(spikes.mean((0, 1)))[::-1][:3]]]})")
    keep = np.setdiff1d(np.arange(B), bad["bad_bands"])
    print(f"keeping {len(keep)}/{B} bands")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    rgb_idx = [spectral.nearest_band(wl, w, 20) for w in (860, 650, 550)]
    fc = cube[..., rgb_idx]
    fc = np.clip((fc - np.percentile(fc, 2, (0, 1))) / (np.percentile(fc, 98, (0, 1)) - np.percentile(fc, 2, (0, 1))), 0, 1)
    ax[0].imshow(fc)
    ax[0].set_title("(a) False colour (NIR/red/green)")
    cmap = plt.get_cmap("tab20", 17)
    ax[1].imshow(np.ma.masked_where(gt == 0, gt), cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    ax[1].set_title("(b) Reference labels (16 classes)")
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    ax[2].plot(wl, snr, color=OKABE[4], lw=1)
    ax[2].scatter(wl[bad["bad_bands"]], snr[bad["bad_bands"]], color=OKABE[5], s=12, zorder=3, label="flagged")
    ax[2].axhline(15, ls="--", color="grey", lw=0.8)
    ax[2].set_xlabel("Wavelength (nm, approx.)")
    ax[2].set_ylabel("SNR (SSDC estimate)")
    ax[2].set_title("(c) Per-band SNR")
    ax[2].legend(frameon=False)
    savefig(fig, "t01_indian_pines_qc.png")

    banner("3. Spectra by crop class")
    lab = gt > 0
    X = cube[lab][:, keep]
    y = gt[lab] - 1
    rows, cols = np.nonzero(lab)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, (ci, name) in enumerate([(1, "Corn-notill"), (9, "Soybean-notill"), (12, "Wheat"), (13, "Woods"),
                                    (7, "Hay-windrowed")]):
        s = X[y == ci]
        m, sd = s.mean(0), s.std(0)
        ax.plot(wl[keep], m, color=OKABE[k], lw=1.2, label=name)
        ax.fill_between(wl[keep], m - sd, m + sd, color=OKABE[k], alpha=0.15, lw=0)
    ax.set_xlabel("Wavelength (nm, approx.)")
    ax.set_ylabel("Radiance-derived value (DN)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    savefig(fig, "t01_class_spectra.png")

    banner("4. Random vs spatially-blocked cross-validation")
    Xs = spectral.snv(X)
    res = []
    for name, model in [("RF", ml.default_classifier("rf")), ("SVM", ml.default_classifier("svm"))]:
        t0 = time.time()
        df = ml.compare_validation_schemes_classification(model, Xs, y, rows, cols, block=16, n_splits=5)
        df.insert(0, "model", name)
        res.append(df)
        print(df.round(3).to_string(index=False), f"  ({time.time() - t0:.0f}s)")
    # buffered blocks: also drop training pixels within 3 px of the test block (edge autocorrelation)
    df = ml.compare_validation_schemes_classification(ml.default_classifier("rf"), Xs, y, rows, cols, block=16,
                                                      n_splits=5, buffer=3)
    df = df[df.scheme == "spatial_block"].assign(scheme="spatial_block+3px_buffer")
    df.insert(0, "model", "RF")
    res.append(df)
    print(df.round(3).to_string(index=False))

    cnn_rows = []
    if run_cnn:
        from sklearn.model_selection import GroupKFold, StratifiedKFold

        from phenokit import deep

        for scheme, splitter, groups in [("random", StratifiedKFold(5, shuffle=True, random_state=0), None),
                                         ("spatial_block", GroupKFold(5), ml.spatial_blocks(rows, cols, 16))]:
            pred = np.empty_like(y)
            t0 = time.time()
            for tr, te in splitter.split(Xs, y, groups):
                pred[te] = deep.train_spectral_classifier(Xs[tr], y[tr], Xs[te], 16, epochs=40)
            rep = ml.classification_report(y, pred)
            cnn_rows.append({"model": "1D-CNN", "scheme": scheme, **{k: rep[k] for k in (
                "overall_accuracy", "kappa", "macro_f1", "average_accuracy")}})
            print(f"1D-CNN {scheme}: OA {rep['overall_accuracy']:.3f} kappa {rep['kappa']:.3f} ({time.time() - t0:.0f}s)")
    table = pd.concat(res + ([pd.DataFrame(cnn_rows)] if cnn_rows else []), ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    piv = table.pivot_table(index="model", columns="scheme", values="overall_accuracy")
    order = [c for c in ["random", "spatial_block", "spatial_block+3px_buffer"] if c in piv]
    piv[order].plot.bar(ax=ax, color=[OKABE[1], OKABE[5], OKABE[6]][:len(order)], rot=0, width=0.75)
    ax.set_ylabel("Overall accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, title=None)
    ax.set_title("Same data, same models — different validation design")
    savefig(fig, "t01_leakage_bar.png")

    # per-class recall under spatial blocking for RF (which crops fail to generalise?)
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(Xs, y, ml.spatial_blocks(rows, cols, 16)):
        pred[te] = clone(ml.default_classifier("rf")).fit(Xs[tr], y[tr]).predict(Xs[te])
    rep_sp = ml.classification_report(y, pred, labels=np.arange(16))
    per_class = pd.DataFrame({"class": CLASSES, "n": np.bincount(y, minlength=16),
                              "recall_spatial": [rep_sp["per_class_recall"][i] for i in range(16)]})
    print(per_class.round(2).to_string(index=False))
    save_results("t01_indian_pines", {"n_bands_kept": len(keep), "median_snr": float(np.median(snr)),
                                      "impulse_noise_pct": float(spikes.mean() * 100),
                                      "flagged_bands_nm": [float(wl[b]) for b in bad["bad_bands"]],
                                      "validation_table": table, "per_class_spatial_rf": per_class})


if __name__ == "__main__":
    main(run_cnn="--no-cnn" not in sys.argv)
