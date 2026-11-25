"""Tutorial 03 — Ground-vehicle imaging: dataset QC, vegetation segmentation and crop/weed AI (CWFID).

CWFID was acquired from an autonomous field robot in an organic carrot field. Each image is a
red/NIR composite (stored as PNG with R = B = red, G = NIR), with a vegetation mask and
crop/weed annotations. Steps:

1. Dataset integrity QC *before* any modelling: completeness, train/test leakage, near
   duplicates, image quality, and agreement between the two label sources.
2. Vegetation segmentation: NDVI+Otsu vs excess-index+Otsu vs k-means, scored by IoU.
3. Crop/weed discrimination: (a) classical object features + random forest;
   (b) a compact U-Net for per-pixel soil/crop/weed segmentation.

Run:  python tutorials/t03_crop_weed_vision_cwfid.py [--no-unet]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import yaml
from _common import DATA, OKABE, banner, plt, save_results, savefig
from scipy import ndimage

from phenokit import datasets, io as pio, ml, qc, segmentation

SOIL, CROP, WEED = 0, 1, 2


def load(base, i):
    img = pio.read_image(base / "images" / f"{i:03d}_image.png").astype(np.float32)
    red, nir = img[..., 0], img[..., 1]
    veg = ~pio.read_image(base / "masks" / f"{i:03d}_mask.png").astype(bool)   # mask True = soil
    ann = pio.read_image(base / "annotations" / f"{i:03d}_annotation.png")
    lab = np.zeros(veg.shape, np.int64)
    lab[ann[..., 1] > 127] = CROP
    lab[ann[..., 0] > 127] = WEED
    return img, red, nir, veg, lab


def ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)


def main(run_unet: bool = True):
    banner("1. Fetch & integrity QC")
    base = datasets.fetch("cwfid", DATA) / "dataset-1.0"
    split = yaml.safe_load((base / "train_test_split.yaml").read_text())
    ids = sorted(int(p.name[:3]) for p in (base / "images").glob("*_image.png"))
    print(f"{len(ids)} images; split train={len(split['train'])} test={len(split['test'])}")
    report = qc.QCReport("CWFID dataset QC")
    missing = [i for i in ids if not (base / "masks" / f"{i:03d}_mask.png").exists()
               or not (base / "annotations" / f"{i:03d}_annotation.png").exists()]
    report.extend([qc.QCResult("cwfid", "missing_masks_or_annotations", qc.FAIL if missing else qc.PASS,
                               len(missing), "0", str(missing))])
    unassigned = sorted(set(ids) - set(split["train"]) - set(split["test"]))
    report.extend([qc.QCResult("cwfid", "images_not_in_any_split", qc.WARN if unassigned else qc.PASS,
                               len(unassigned), "0", str(unassigned))])
    leak = qc.qc_split_leakage({"train": split["train"], "test": split["test"]}, target="cwfid_split")
    report.extend(leak)
    for r in leak:
        print(f"split check: {r.check} → {r.status} {r.message}")

    hashes, iou_rows, iq = {}, [], []
    data = {}
    for i in ids:
        img, red, nir, veg, lab = load(base, i)
        data[i] = (img, red, nir, veg, lab)
        hashes[f"{i:03d}"] = qc.average_hash(img)
        q = {r.check: r for r in qc.qc_rgb(img, target=f"img{i:03d}")}
        iq.append({"id": i, "sharpness": q["sharpness_laplacian_var"].value,
                   "over": q["overexposed_fraction"].value, "brightness": q["mean_brightness"].value})
        iou_rows.append({"id": i, "label_iou": qc.qc_mask_consistency(veg, lab > 0)[0].value})
    dups = qc.find_near_duplicates(hashes, max_hamming=3)
    report.extend([qc.QCResult("cwfid", "near_duplicate_images", qc.WARN if dups else qc.PASS, len(dups), "≤3 bits",
                               str(dups[:10]))])
    iq = pd.DataFrame(iq).merge(pd.DataFrame(iou_rows), on="id")
    sharp_thr = iq.sharpness.median() - 3 * 1.4826 * (iq.sharpness - iq.sharpness.median()).abs().median()
    blurry = iq.loc[iq.sharpness < sharp_thr, "id"].tolist()
    low_iou = iq.loc[iq.label_iou < 0.85, "id"].tolist()
    report.extend([qc.QCResult("cwfid", "relatively_blurry_images", qc.WARN if blurry else qc.PASS, len(blurry),
                               "robust z<-3 within dataset", str(blurry)),
                   qc.QCResult("cwfid", "mask_vs_annotation_low_iou", qc.WARN if low_iou else qc.PASS, len(low_iou),
                               "IoU<0.85", str(low_iou))])
    print(f"near-duplicates: {dups}; relatively blurry: {blurry}; mask/annotation IoU median "
          f"{iq.label_iou.median():.3f} (min {iq.label_iou.min():.3f} on image {int(iq.loc[iq.label_iou.idxmin(), 'id'])})")
    report.to_html(DATA.parent.parent / "outputs" / "cwfid_qc_report.html")

    # remediate the leakage: an image in both splits is kept in train only
    test_ids = [i for i in split["test"] if i not in split["train"]]
    train_ids = list(split["train"])
    print(f"after remediation: train {len(train_ids)}, test {len(test_ids)}")

    banner("2. Vegetation segmentation vs reference mask (test images)")
    seg_rows = []
    for i in test_ids:
        img, red, nir, veg, lab = data[i]
        nd = ndvi(red, nir)
        preds = {
            "NDVI+Otsu": segmentation.clean_mask(nd > segmentation.otsu_threshold(nd), min_size=30),
            "ExG(false-colour)+Otsu": segmentation.segment_vegetation_rgb(img.astype(np.uint8), "exg_otsu", min_size=30),
            "k-means (Lab)": segmentation.segment_vegetation_rgb(img.astype(np.uint8), "kmeans_lab", min_size=30),
        }
        for name, p in preds.items():
            seg_rows.append({"id": i, "method": name, **segmentation.segmentation_metrics(p, veg)})
    seg = pd.DataFrame(seg_rows)
    seg_summary = seg.groupby("method")[["iou", "dice", "precision", "recall"]].mean()
    print(seg_summary.round(3).to_string())

    banner("3a. Crop/weed: object features + random forest")

    def objects(i):
        img, red, nir, veg, lab = data[i]
        nd = ndvi(red, nir)
        L, feats = segmentation.object_features(img, veg, min_size=50, nir=nir)
        if feats.empty:
            return feats
        feats["mean_ndvi"] = ndimage.mean(nd, L, feats.label.to_numpy())
        feats["std_ndvi"] = ndimage.standard_deviation(nd, L, feats.label.to_numpy())
        crop = ndimage.sum(lab == CROP, L, feats.label.to_numpy())
        weed = ndimage.sum(lab == WEED, L, feats.label.to_numpy())
        feats["y"] = np.where(crop >= weed, CROP, WEED)
        feats["img"] = i
        return feats

    tr = pd.concat([objects(i) for i in train_ids], ignore_index=True)
    te = pd.concat([objects(i) for i in test_ids], ignore_index=True)
    cols = ["area_px", "perimeter_px", "compactness", "eccentricity", "extent", "bbox_aspect", "mean_g", "std_g",
            "mean_r", "mean_exg", "mean_nir", "mean_ndvi", "std_ndvi"]
    rf = ml.default_classifier("rf").fit(tr[cols], tr.y)
    p = rf.predict(te[cols])
    obj_rep = ml.classification_report(te.y.to_numpy(), p, labels=np.array([CROP, WEED]))
    print(f"objects train {len(tr)} test {len(te)}; OA {obj_rep['overall_accuracy']:.3f}, "
          f"macro-F1 {obj_rep['macro_f1']:.3f}, recall crop {obj_rep['per_class_recall'][CROP]:.2f} "
          f"weed {obj_rep['per_class_recall'][WEED]:.2f}")
    imp = pd.Series(rf.feature_importances_, cols).sort_values(ascending=False)
    print("top features:", imp.head(5).round(3).to_dict())
    # pixel-level IoU for the object approach (objects painted with their predicted class)
    obj_pix = []
    te_pred = te.assign(pred=p)
    for i in test_ids:
        img, red, nir, veg, lab = data[i]
        L, _ = ndimage.label(veg)
        out = np.zeros_like(lab)
        sub = te_pred[te_pred.img == i]
        lut = np.zeros(L.max() + 1, int)
        lut[sub.label.to_numpy()] = sub.pred.to_numpy()
        out = lut[L]
        obj_pix.append((lab, out))

    def pixel_iou(pairs):
        res = {}
        for c, n in [(SOIL, "soil"), (CROP, "crop"), (WEED, "weed")]:
            inter = sum(((a == c) & (b == c)).sum() for a, b in pairs)
            union = sum(((a == c) | (b == c)).sum() for a, b in pairs)
            res[n] = inter / union
        res["mIoU"] = float(np.mean(list(res.values())))
        return res

    obj_iou = pixel_iou(obj_pix)
    print("object-RF pixel IoU:", {k: round(v, 3) for k, v in obj_iou.items()})

    unet_iou, unet_time = None, None
    example = test_ids[0]
    unet_pred_example = None
    if run_unet:
        banner("3b. Crop/weed: compact U-Net (CPU)")
        from phenokit import deep

        def prep(i):
            img, red, nir, veg, lab = data[i]
            x = np.stack([red / 255, nir / 255, (ndvi(red, nir) + 1) / 2], -1)[::2, ::2]
            return x.astype(np.float32), lab[::2, ::2]

        Xtr, Ytr = zip(*[prep(i) for i in train_ids])
        freq = np.bincount(np.concatenate([y.ravel() for y in Ytr]), minlength=3) + 1
        w = (freq.sum() / freq) ** 0.5
        w = w / w.mean()
        t0 = time.time()
        model = deep.train_unet(list(Xtr), list(Ytr), 3, epochs=30, patch=128, patches_per_epoch=320, batch=16,
                                class_weights=w, verbose=True)
        unet_time = time.time() - t0
        pairs = []
        for i in test_ids:
            x, y = prep(i)
            pr = deep.predict_unet(model, x)
            pairs.append((y, pr))
            if i == example:
                unet_pred_example = pr
        unet_iou = pixel_iou(pairs)
        print(f"U-Net pixel IoU ({unet_time:.0f}s training):", {k: round(v, 3) for k, v in unet_iou.items()})

    # figure: example image, reference, NDVI segmentation, crop/weed prediction
    img, red, nir, veg, lab = data[example]
    nd = ndvi(red, nir)
    cmap = plt.matplotlib.colors.ListedColormap(["#d9c7a7", OKABE[2], OKABE[5]])
    n = 4 if unet_pred_example is not None else 3
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 2.8))
    ax[0].imshow(img.astype(np.uint8))
    ax[0].set_title(f"(a) Image {example:03d} (R/NIR/R)")
    ax[1].imshow(nd > segmentation.otsu_threshold(nd), cmap="Greens")
    ax[1].set_title("(b) NDVI + Otsu")
    ax[2].imshow(lab, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax[2].set_title("(c) Reference: soil/crop/weed")
    if unet_pred_example is not None:
        ax[3].imshow(unet_pred_example, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax[3].set_title("(d) U-Net prediction")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    savefig(fig, "t03_cwfid_example.png")

    save_results("t03_cwfid", {"split_leakage": [r.__dict__ for r in leak], "near_duplicates": dups,
                               "blurry": blurry, "low_label_iou": low_iou,
                               "label_iou_median": float(iq.label_iou.median()),
                               "segmentation": seg_summary.reset_index(),
                               "object_rf": {k: v for k, v in obj_rep.items() if k != "confusion"},
                               "object_rf_pixel_iou": obj_iou, "object_rf_top_features": imp.head(8).to_dict(),
                               "unet_pixel_iou": unet_iou, "unet_train_seconds": unet_time,
                               "n_train": len(train_ids), "n_test": len(test_ids)})


if __name__ == "__main__":
    main(run_unet="--no-unet" not in sys.argv)
