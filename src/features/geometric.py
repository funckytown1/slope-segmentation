"""
Per-point features: geometry (eigen-features + verticality) and color (HSV).

Scale handling is the crux for cross-slope comparability: neighborhood size
is set as a multiple of each cloud's own median point spacing, so features
mean the same thing on a dense slope and a sparse one. Color is optionally
standardized per-cloud so one slope's gray doesn't map to a different
feature value than another's.

Prefers `jakteristics` for the geometric eigen-features (fast, C-backed);
falls back to a scikit-learn + PCA implementation if it isn't installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


GEOMETRIC_FEATURES = [
    "planarity", "linearity", "sphericity",
    "surface_variation", "verticality", "anisotropy",
]


def estimate_point_spacing(xyz: np.ndarray, sample: int = 20000) -> float:
    """Median nearest-neighbor distance -- the cloud's characteristic spacing."""
    from sklearn.neighbors import NearestNeighbors

    n = len(xyz)
    idx = np.random.default_rng(0).choice(n, size=min(sample, n), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(xyz)
    d, _ = nn.kneighbors(xyz[idx])
    return float(np.median(d[:, 1]))


def _geometric_jakteristics(xyz: np.ndarray, radius: float) -> pd.DataFrame:
    import jakteristics as jak

    names = ["planarity", "linearity", "sphericity",
             "surface_variation", "verticality", "anisotropy"]
    feats = jak.compute_features(xyz, search_radius=radius, feature_names=names)
    return pd.DataFrame(feats, columns=names)


def _geometric_fallback(xyz: np.ndarray, k: int) -> pd.DataFrame:
    """PCA over kNN neighborhoods. Slower; use jakteristics when available."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(xyz)
    _, idx = nn.kneighbors(xyz)

    out = np.zeros((len(xyz), 6), dtype=np.float64)
    for i in range(len(xyz)):
        pts = xyz[idx[i]]
        cov = np.cov((pts - pts.mean(0)).T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        evals = np.clip(evals, 1e-12, None)
        l1, l2, l3 = evals
        s = l1 + l2 + l3
        planarity = (l2 - l3) / l1
        linearity = (l1 - l2) / l1
        sphericity = l3 / l1
        surface_variation = l3 / s
        anisotropy = (l1 - l3) / l1
        # verticality from the smallest-eigenvalue vector (surface normal)
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        verticality = 1.0 - abs(normal[2])
        out[i] = [planarity, linearity, sphericity,
                  surface_variation, verticality, anisotropy]
    return pd.DataFrame(out, columns=[
        "planarity", "linearity", "sphericity",
        "surface_variation", "verticality", "anisotropy"])


def geometric_features(xyz: np.ndarray, radius_mult: float = 5.0,
                       k_fallback: int = 30) -> pd.DataFrame:
    spacing = estimate_point_spacing(xyz)
    radius = radius_mult * spacing
    print(f"  point spacing ~{spacing:.3f} m; feature radius {radius:.3f} m")
    try:
        return _geometric_jakteristics(xyz, radius)
    except ImportError:
        print("  jakteristics not installed -> slower PCA fallback")
        return _geometric_fallback(xyz, k_fallback)


def color_features(rgb: np.ndarray, per_cloud_normalize: bool = True) -> pd.DataFrame:
    """RGB (0-1) -> H, S, V, optionally per-cloud standardized."""
    import matplotlib.colors as mcolors

    hsv = mcolors.rgb_to_hsv(np.clip(rgb, 0, 1))
    df = pd.DataFrame(hsv, columns=["hue", "sat", "val"])
    if per_cloud_normalize:
        # Standardize sat/val (lighting/lithology shift); leave hue as-is
        # since it's an angle and more stable across lighting.
        for c in ("sat", "val"):
            mu, sd = df[c].mean(), df[c].std() + 1e-9
            df[c + "_z"] = (df[c] - mu) / sd
    return df


def build_features(df: pd.DataFrame, per_cloud_color_norm: bool = True,
                   radius_mult: float = 5.0) -> pd.DataFrame:
    """Full per-point feature table for one cloud (canonical df in)."""
    xyz = df[["x", "y", "z"]].to_numpy()
    rgb = df[["r", "g", "b"]].to_numpy()
    geo = geometric_features(xyz, radius_mult=radius_mult)
    col = color_features(rgb, per_cloud_normalize=per_cloud_color_norm)
    feats = pd.concat([geo.reset_index(drop=True), col.reset_index(drop=True)], axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats
