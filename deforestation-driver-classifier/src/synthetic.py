"""Generator data sintetis untuk demo klasifikasi penyebab langsung deforestasi.

Setiap sampel adalah patch (C, H, W) di sekitar satu peringatan deforestasi:
  kanal 0-5 : Sentinel-2 (B2, B3, B4, B8, B11, B12), reflektansi 0-1
  kanal 6-7 : Sentinel-1 (VV, VH) dalam dB
  kanal 8   : mask peringatan (1 = piksel yang ditandai hilang tutupan pohonnya)

DATA INI SINTETIS dan hanya untuk mendemonstrasikan pipeline. Pola spektral dan
bentuknya disederhanakan; hasil akurasi TIDAK mencerminkan performa pada data nyata.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

CLASSES = [
    "pertanian_skala_kecil",
    "pertanian_skala_besar",
    "jalan",
    "penebangan_selektif",
    "pertambangan",
    "kebakaran",
    "gangguan_alami",
]
CHANNELS = ["B2", "B3", "B4", "B8", "B11", "B12", "VV", "VH", "alert_mask"]

# Tutupan hutan sehat: [B2, B3, B4, B8, B11, B12, VV(dB), VH(dB)]
FOREST = np.array([0.03, 0.06, 0.03, 0.35, 0.15, 0.07, -7.0, -13.0])

# Tanda spektral/radar kasar setelah gangguan, per kelas
SIGNATURES = {
    "pertanian_skala_kecil": [0.05, 0.08, 0.07, 0.26, 0.20, 0.11, -8.5, -14.5],
    "pertanian_skala_besar": [0.06, 0.10, 0.12, 0.29, 0.29, 0.20, -10.5, -17.5],
    "jalan": [0.10, 0.13, 0.15, 0.22, 0.30, 0.24, -9.5, -16.0],
    "penebangan_selektif": [0.04, 0.07, 0.05, 0.27, 0.18, 0.09, -8.5, -14.5],
    "pertambangan": [0.12, 0.16, 0.20, 0.26, 0.36, 0.31, -11.5, -18.5],
    "kebakaran": [0.04, 0.05, 0.07, 0.12, 0.23, 0.20, -9.5, -16.5],
    "gangguan_alami": [0.05, 0.06, 0.05, 0.16, 0.08, 0.04, -19.0, -25.0],
}


def _smooth_noise(rng, size, sigma):
    n = gaussian_filter(rng.standard_normal((size, size)), sigma)
    return n / (n.std() + 1e-8)


def _grid(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return yy, xx


def _blob(rng, size, cy, cx, radius, rough=3.0):
    yy, xx = _grid(size)
    r = np.hypot(yy - cy, xx - cx)
    return (r + rough * _smooth_noise(rng, size, 3)) < radius


def _line(rng, size, width):
    yy, xx = _grid(size)
    theta = rng.uniform(0, np.pi)
    cy, cx = size / 2 + rng.uniform(-6, 6, 2)
    dist = np.abs((xx - cx) * np.sin(theta) - (yy - cy) * np.cos(theta))
    return dist < width


def make_mask(cls: str, size: int, rng: np.random.Generator) -> np.ndarray:
    c = size / 2
    if cls == "pertanian_skala_kecil":
        m = np.zeros((size, size), bool)
        for _ in range(rng.integers(1, 4)):
            m |= _blob(rng, size, c + rng.uniform(-8, 8), c + rng.uniform(-8, 8),
                       rng.uniform(3.5, 6.5), rough=1.5)
    elif cls == "pertanian_skala_besar":
        h, w = rng.integers(20, 34, 2)
        m = np.zeros((size, size), bool)
        y0, x0 = int(c - h / 2), int(c - w / 2)
        m[y0:y0 + h, x0:x0 + w] = True
    elif cls == "jalan":
        m = _line(rng, size, width=rng.uniform(1.2, 2.2))
    elif cls == "penebangan_selektif":
        m = np.zeros((size, size), bool)
        for _ in range(rng.integers(6, 13)):
            m |= _blob(rng, size, rng.uniform(10, size - 10), rng.uniform(10, size - 10),
                       rng.uniform(1.5, 2.6), rough=0.5)
        m |= _line(rng, size, 1.0) & _blob(rng, size, c, c, 18, rough=2)
    elif cls == "pertambangan":
        m = _blob(rng, size, c, c, rng.uniform(12, 17), rough=4.0)
    elif cls == "kebakaran":
        m = _blob(rng, size, c, c, rng.uniform(15, 21), rough=7.0)
    elif cls == "gangguan_alami":
        yy, xx = _grid(size)
        amp, freq, phase = rng.uniform(4, 9), rng.uniform(0.08, 0.16), rng.uniform(0, 6.28)
        ridge = c + amp * np.sin(freq * xx + phase)
        m = np.abs(yy - ridge) < rng.uniform(3.5, 6)
    else:
        raise ValueError(cls)
    return m


def make_sample(cls: str, size: int, rng: np.random.Generator,
                cloud_prob: float = 0.15) -> np.ndarray:
    """Buat satu patch (9, size, size) untuk kelas `cls`."""
    texture = np.stack([_smooth_noise(rng, size, 1.2) for _ in range(8)])
    base = np.broadcast_to(FOREST[:, None, None], (8, size, size)).copy()
    base[:6] *= 1 + 0.12 * texture[:6]
    base[6:] += 1.0 * texture[6:]

    mask = make_mask(cls, size, rng)
    soft = gaussian_filter(mask.astype(np.float32), 0.8)
    jitter = 1 + rng.uniform(-0.15, 0.15, 8)
    sig = np.array(SIGNATURES[cls]) * jitter
    sig[6:] = np.array(SIGNATURES[cls])[6:] + rng.uniform(-1.5, 1.5, 2)
    disturbed = sig[:, None, None] * (1 + 0.08 * texture)
    disturbed[6:] = sig[6:, None, None] + 0.8 * texture[6:]
    x = base * (1 - soft) + disturbed * soft

    # Awan: menutupi data optik di sebagian patch, SAR tetap bersih
    if rng.random() < cloud_prob:
        cloud = gaussian_filter(_blob(rng, size, rng.uniform(0, size), rng.uniform(0, size),
                                      rng.uniform(18, 30), 6.0).astype(np.float32), 2.0)
        x[:6] = x[:6] * (1 - cloud) + 0.6 * cloud

    x[:6] = np.clip(x[:6], 0.0, 1.0)
    return np.concatenate([x, mask[None].astype(np.float32)]).astype(np.float32)


def make_dataset(n_per_class: int = 300, size: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(make_sample(cls, size, rng))
            y.append(label)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]
