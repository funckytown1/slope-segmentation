"""
Canonical point-cloud schema used everywhere downstream of the loaders.

Every dataset-specific loader (John Henry LAZ+ortho, Zenodo txt/pcd, a
future self-captured twin, etc.) must reduce its source data to a pandas
DataFrame with exactly these columns before anything else touches it:

    x, y, z   float64   -- coordinates in the source CRS, meters
    r, g, b   float32   -- color, normalized to [0, 1]
    label     Int64     -- optional; pandas nullable int, NA if unlabeled

Keeping one canonical format is what lets Stage 1/2/3 code stay dataset
agnostic: write a new loader, not new pipeline code, when swapping data in.

Feature computation (src/features/) is responsible for normalizing scale
(point spacing / neighborhood radius) across datasets -- this module only
guarantees the columns and units above, not comparable point density.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

CANONICAL_COLUMNS = ["x", "y", "z", "r", "g", "b"]
OPTIONAL_COLUMNS = ["label"]


def validate_canonical(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required canonical columns: {missing}")

    for c in ("r", "g", "b"):
        cmin, cmax = df[c].min(), df[c].max()
        if cmin < -1e-6 or cmax > 1.0 + 1e-6:
            raise ValueError(
                f"Column '{c}' out of expected [0, 1] range (got min={cmin}, "
                f"max={cmax}). Did a loader forget to normalize color?"
            )

    if df[CANONICAL_COLUMNS].isna().any().any():
        n_bad = df[CANONICAL_COLUMNS].isna().any(axis=1).sum()
        raise ValueError(
            f"{n_bad} rows have NaNs in required columns. Drop or fix them "
            f"before saving -- NaNs should be filtered out in the loader "
            f"(e.g. points outside the orthophoto extent)."
        )


def save_canonical(df: pd.DataFrame, path: str | Path) -> None:
    validate_canonical(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} points -> {path}")


def load_canonical(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_canonical(df)
    return df


def summarize(df: pd.DataFrame, name: str = "cloud") -> None:
    """Quick printed sanity check -- used by the verify script and loaders."""
    print(f"--- {name}: {len(df):,} points ---")
    print(f"  x: [{df.x.min():.2f}, {df.x.max():.2f}]")
    print(f"  y: [{df.y.min():.2f}, {df.y.max():.2f}]")
    print(f"  z: [{df.z.min():.2f}, {df.z.max():.2f}]")
    for c in ("r", "g", "b"):
        print(f"  {c}: mean={df[c].mean():.3f} std={df[c].std():.3f}")
    # A near-zero std on all three channels usually means "colorization
    # silently failed and everything is black/one color" -- worth flagging
    # loudly rather than letting it slide into Stage 1.
    if df[["r", "g", "b"]].std().max() < 1e-3:
        print("  WARNING: color has ~zero variance across the whole cloud. "
              "This usually means colorization failed (e.g. CRS mismatch, "
              "all points fell outside the raster). Inspect before proceeding.")
