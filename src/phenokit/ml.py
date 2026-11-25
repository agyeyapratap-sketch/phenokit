"""Leakage-aware machine learning for phenotyping.

The central idea: the *validation design* must match the *deployment question*.
Random pixel/plot splits answer "can I interpolate within this trial?"; spatially blocked
splits answer "does it work on unseen parts of the field?"; leave-one-species-out (or
leave-one-site/season-out) answers "will it transfer to a new crop/environment?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .validation import agreement


# --------------------------------------------------------------------------- splitting


def spatial_blocks(rows: np.ndarray, cols: np.ndarray, block: int) -> np.ndarray:
    """Group ID of square spatial blocks (``block`` pixels or plots on a side)."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    ncb = int(cols.max() // block + 1)
    return (rows // block) * ncb + cols // block


def buffered_block_split(rows, cols, block: int, test_blocks, buffer: int = 0):
    """Train/test indices where training pixels within ``buffer`` px of a test block are dropped."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    g = spatial_blocks(rows, cols, block)
    test = np.isin(g, test_blocks)
    if buffer > 0:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.c_[rows[test], cols[test]])
        d, _ = tree.query(np.c_[rows[~test], cols[~test]], k=1)
        train_idx = np.where(~test)[0][d > buffer]
    else:
        train_idx = np.where(~test)[0]
    return train_idx, np.where(test)[0]


# --------------------------------------------------------------------------- models


def default_classifier(kind: str = "rf", seed: int = 0):
    if kind == "rf":
        return RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=1)
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=100, gamma="scale"))
    raise ValueError(kind)


def default_regressor(kind: str = "rf", seed: int = 0, n_components: int = 8):
    if kind == "rf":
        return RandomForestRegressor(400, n_jobs=-1, random_state=seed, min_samples_leaf=2)
    if kind == "plsr":
        return make_pipeline(StandardScaler(), PLSRegression(n_components))
    raise ValueError(kind)


def select_pls_components(X, y, max_components: int = 20, cv: int = 5, groups=None, seed: int = 0) -> dict:
    """Choose PLSR latent variables by (grouped) CV RMSE with the one-standard-error rule."""
    splitter = GroupKFold(cv) if groups is not None else KFold(cv, shuffle=True, random_state=seed)
    max_components = min(max_components, X.shape[1], int(len(X) * (1 - 1 / cv)) - 1)
    rmse = np.zeros((max_components, cv))
    for f, (tr, te) in enumerate(splitter.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        for k in range(1, max_components + 1):
            m = PLSRegression(k).fit(sc.transform(X[tr]), y[tr])
            rmse[k - 1, f] = np.sqrt(np.mean((m.predict(sc.transform(X[te])).ravel() - y[te]) ** 2))
    mean, se = rmse.mean(1), rmse.std(1, ddof=1) / np.sqrt(cv)
    best = int(np.argmin(mean))
    one_se = int(np.where(mean <= mean[best] + se[best])[0][0]) + 1
    return {"n_components": one_se, "best_min": best + 1, "cv_rmse": mean}


def vip_scores(pls: PLSRegression) -> np.ndarray:
    """Variable importance in projection (Wold et al., 2001) for a fitted PLSRegression."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = w.shape[0]
    ss = np.diag(t.T @ t @ q.T @ q).reshape(-1)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (wn ** 2 @ ss) / ss.sum())


# --------------------------------------------------------------------------- CV runners


def cross_validate_regression(model, X, y, groups=None, scheme: str = "kfold", n_splits: int = 5,
                              seed: int = 0) -> tuple[dict, np.ndarray]:
    """Out-of-fold predictions and agreement stats. ``scheme``: kfold | group | logo."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if scheme == "kfold":
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    elif scheme == "group":
        splits = GroupKFold(n_splits).split(X, y, groups)
    elif scheme == "logo":
        splits = LeaveOneGroupOut().split(X, y, groups)
    else:
        raise ValueError(scheme)
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = np.ravel(m.predict(X[te]))
    return agreement(y, pred), pred


def classification_report(y_true, y_pred, labels=None) -> dict:
    labels = np.unique(np.r_[y_true, y_pred]) if labels is None else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return {"overall_accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
            "average_accuracy": float(per_class.mean()), "per_class_recall": dict(zip(map(int, labels), per_class)),
            "confusion": cm}


def compare_validation_schemes_classification(model, X, y, rows, cols, block: int = 16, n_splits: int = 5,
                                              buffer: int = 0, seed: int = 0, train_fraction: float | None = None):
    """Random (stratified) vs spatially-blocked CV for pixel classification.

    Returns a DataFrame with OA/kappa/macro-F1 per scheme — the gap is the optimistic bias
    caused by spatial autocorrelation between training and test pixels.
    """
    X, y = np.asarray(X), np.asarray(y)
    out = []
    schemes = {"random": StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y),
               "spatial_block": GroupKFold(n_splits).split(X, y, spatial_blocks(rows, cols, block))}
    for name, splits in schemes.items():
        pred = np.empty_like(y)
        for tr, te in splits:
            if buffer and name == "spatial_block":
                from scipy.spatial import cKDTree

                d, _ = cKDTree(np.c_[rows[te], cols[te]]).query(np.c_[rows[tr], cols[tr]], k=1)
                tr = tr[d > buffer]
            if train_fraction:
                rng = np.random.default_rng(seed)
                tr = rng.choice(tr, max(1, int(len(tr) * train_fraction)), replace=False)
            pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
        rep = classification_report(y, pred)
        out.append({"scheme": name, **{k: rep[k] for k in ("overall_accuracy", "kappa", "macro_f1",
                                                            "average_accuracy")}})
    return pd.DataFrame(out)


def transfer_matrix(model, X, y, domain) -> pd.DataFrame:
    """Train on one domain (species/site/platform), test on each other → R² matrix."""
    X, y, domain = np.asarray(X, float), np.asarray(y, float), np.asarray(domain)
    doms = list(pd.unique(domain))
    M = pd.DataFrame(index=doms, columns=doms, dtype=float)
    for a in doms:
        tr = domain == a
        m = clone(model).fit(X[tr], y[tr])
        for b in doms:
            te = domain == b
            if a == b:
                st, _ = cross_validate_regression(model, X[te], y[te], scheme="kfold")
                M.loc[a, b] = st.get("r2", np.nan)
            else:
                M.loc[a, b] = agreement(y[te], np.ravel(m.predict(X[te]))).get("r2", np.nan)
    M.index.name = "train"
    M.columns.name = "test"
    return M
