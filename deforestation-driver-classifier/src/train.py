"""Latih dan evaluasi CNN penyebab deforestasi.

Format data (.npz):
    X       : float32 (N, C, H, W)  -- patch Sentinel-1/2 + mask peringatan
    y       : int64   (N,)          -- indeks kelas
    classes : (K,) nama kelas       -- opsional

Contoh:
    python -m src.train --data data/synthetic.npz --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import DriverCNN


def normalize_stats(X: np.ndarray):
    """Mean/std per kanal dari data latih. Kanal terakhir (mask) tidak dinormalisasi."""
    mean = X.mean(axis=(0, 2, 3), keepdims=True)
    std = X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    mean[:, -1], std[:, -1] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def predict(model, X, device, batch=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def plot_confusion(cm, names, path):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title("Confusion matrix (ternormalisasi per baris)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic.npz")
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.data, allow_pickle=False)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    names = [str(c) for c in d["classes"]] if "classes" in d.files else [str(i) for i in range(y.max() + 1)]

    # Split stratified: 70% latih, 15% validasi, 15% uji
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=a.seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=a.seed)

    mean, std = normalize_stats(X_tr)
    X_tr, X_va, X_te = [(v - mean) / std for v in (X_tr, X_va, X_te)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DriverCNN(X.shape[1], len(names)).to(device)

    counts = np.bincount(y_tr, minlength=len(names)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(names) * counts.clip(min=1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    mk = lambda X_, y_, shuffle: DataLoader(
        TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_)),
        batch_size=a.batch_size, shuffle=shuffle)
    tr_loader, va_loader = mk(X_tr, y_tr, True), mk(X_va, y_va, False)

    best_acc, history = 0.0, []
    for ep in range(1, a.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, tr_loader, device, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, va_loader, device, criterion)
        scheduler.step()
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        print(f"epoch {ep:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "classes": names, "in_channels": X.shape[1]}, out / "best_model.pt")

    # Evaluasi pada data uji dengan model terbaik
    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    y_pred = predict(model, X_te, device)
    report = classification_report(y_te, y_pred, target_names=names, digits=3, zero_division=0)
    print("\n=== Evaluasi data uji ===\n" + report)
    (out / "classification_report.txt").write_text(report)
    plot_confusion(confusion_matrix(y_te, y_pred, labels=range(len(names))), names, out / "confusion_matrix.png")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Hasil disimpan di: {out.resolve()}")


if __name__ == "__main__":
    main()
