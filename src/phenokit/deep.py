"""Deep-learning models (optional, requires PyTorch).

* :class:`SpectralCNN1D` — 1-D convolutional network for per-pixel/plot spectra.
* :class:`TinyUNet` — compact U-Net (Ronneberger et al., 2015) for semantic segmentation
  (e.g. soil / crop / weed) that trains on CPU in minutes on small datasets.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("phenokit.deep requires PyTorch: pip install phenokit[deep]") from e


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SpectralCNN1D(nn.Module):
    def __init__(self, n_bands: int, n_out: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(width, 2 * width, 5, padding=2), nn.BatchNorm1d(2 * width), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(2 * width, 2 * width, 3, padding=1), nn.BatchNorm1d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * width, n_out))

    def forward(self, x):  # x: (N, bands)
        return self.net(x.unsqueeze(1))


def train_spectral_classifier(X_train, y_train, X_test, n_classes: int, epochs: int = 60, lr: float = 2e-3,
                              batch: int = 128, seed: int = 0, verbose: bool = False):
    """Train a 1-D CNN on standardised spectra; returns predicted labels for ``X_test``."""
    set_seed(seed)
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    Xtr = torch.tensor((X_train - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor((X_test - mu) / sd, dtype=torch.float32)
    model = SpectralCNN1D(Xtr.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and ep % 10 == 0:
            print(f"epoch {ep} loss {tot / len(Xtr):.3f}")
    model.eval()
    with torch.no_grad():
        return torch.cat([model(Xte[i:i + 4096]).argmax(1) for i in range(0, len(Xte), 4096)]).numpy()


class _Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 3, base: int = 16):
        super().__init__()
        self.e1, self.e2, self.e3 = _Block(in_ch, base), _Block(base, 2 * base), _Block(2 * base, 4 * base)
        self.b = _Block(4 * base, 8 * base)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * base, 4 * base, 2, 2), _Block(8 * base, 4 * base)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), _Block(4 * base, 2 * base)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * base, base, 2, 2), _Block(2 * base, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def train_unet(images, labels, n_classes: int, epochs: int = 40, patch: int = 128, patches_per_epoch: int = 256,
               lr: float = 3e-3, batch: int = 16, class_weights=None, seed: int = 0, base: int = 16,
               verbose: bool = False):
    """Train :class:`TinyUNet` on random patches with flip augmentation.

    ``images``: list of float arrays (H, W, C) in 0–1; ``labels``: list of int arrays (H, W).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    C = images[0].shape[-1]
    model = TinyUNet(C, n_classes, base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (patches_per_epoch // batch))
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(patches_per_epoch // batch):
            xb, yb = [], []
            for _ in range(batch):
                k = rng.integers(len(images))
                H, W = labels[k].shape
                r, c = rng.integers(0, H - patch + 1), rng.integers(0, W - patch + 1)
                x, y = images[k][r:r + patch, c:c + patch], labels[k][r:r + patch, c:c + patch]
                if rng.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                if rng.random() < 0.5:
                    x, y = x[::-1], y[::-1]
                xb.append(np.ascontiguousarray(x.transpose(2, 0, 1)))
                yb.append(np.ascontiguousarray(y))
            xb = torch.tensor(np.array(xb), dtype=torch.float32)
            yb = torch.tensor(np.array(yb), dtype=torch.long)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=w)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if verbose and ep % 5 == 0:
            print(f"epoch {ep} loss {tot / (patches_per_epoch // batch):.3f}")
    return model


def predict_unet(model, image: np.ndarray) -> np.ndarray:
    """Predict a full image (padded to a multiple of 8)."""
    model.eval()
    H, W, _ = image.shape
    ph, pw = (-H) % 8, (-W) % 8
    x = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    with torch.no_grad():
        out = model(torch.tensor(x.transpose(2, 0, 1)[None].copy(), dtype=torch.float32))
    return out.argmax(1)[0, :H, :W].numpy()
