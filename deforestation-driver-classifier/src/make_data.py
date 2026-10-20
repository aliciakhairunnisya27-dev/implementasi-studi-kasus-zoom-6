"""Buat dataset sintetis dan simpan ke .npz.

Contoh:
    python -m src.make_data --n-per-class 300 --out data/synthetic.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .synthetic import CHANNELS, CLASSES, make_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=300)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic.npz")
    a = p.parse_args()

    X, y = make_dataset(a.n_per_class, a.size, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, classes=np.array(CLASSES), channels=np.array(CHANNELS))
    print(f"Disimpan: {out}  X={X.shape}  y={y.shape}")


if __name__ == "__main__":
    main()
