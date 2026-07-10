"""
Per-source loaders. Each returns a canonical DataFrame (see io/canonical.py):
columns x, y, z, r, g, b with color in [0, 1].

Add a new dataset == add a function here. Nothing downstream changes.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .canonical import validate_canonical, CANONICAL_COLUMNS


# --------------------------------------------------------------------------
# Zenodo rock slopes
# --------------------------------------------------------------------------
def load_zenodo_txt(path: str | Path) -> pd.DataFrame:
    """
    Zenodo slope .txt: whitespace-delimited 'X Y Z R G B', RGB already 0-1.
    Guards against files that turn out to be 0-255 or geometry-only.

    Uses pandas' C parser (delim_whitespace) rather than np.loadtxt -- the
    latter is ~10-30x slower and some of these files are ~700 MB. XYZ are kept
    float64 because Slope_4 is in UTM (easting ~682000), where float32 would
    throw away ~6 cm of precision.
    """
    import warnings

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # delim_whitespace deprecation
        raw = pd.read_csv(path, delim_whitespace=True, header=None,
                          comment="#", dtype="float64")
    if raw.shape[1] < 6:
        raise ValueError(
            f"{path.name}: only {raw.shape[1]} columns -> geometry-only, no RGB. "
            f"This file can't be used for the color-dependent pipeline."
        )
    raw = raw.iloc[:, :6]
    raw.columns = CANONICAL_COLUMNS

    df = pd.DataFrame()
    df[["x", "y", "z"]] = raw[["x", "y", "z"]].astype("float64")
    rgb = raw[["r", "g", "b"]]
    # If color looks like 0-255, rescale. Zenodo .txt should already be 0-1.
    if float(rgb.to_numpy().max()) > 1.5:
        rgb = rgb / 255.0
    df[["r", "g", "b"]] = rgb.clip(0.0, 1.0).astype("float32")
    validate_canonical(df)
    return df


def load_zenodo_pcd(path: str | Path) -> pd.DataFrame:
    """Zenodo slope .pcd via Open3D. Requires the file to carry color."""
    import open3d as o3d

    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_colors():
        raise ValueError(f"{path.name}: .pcd has no color field -> unusable for color pipeline.")

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)  # Open3D returns colors in 0-1
    df = pd.DataFrame(
        np.hstack([pts, cols]).astype(np.float64), columns=CANONICAL_COLUMNS
    )
    df[["r", "g", "b"]] = df[["r", "g", "b"]].clip(0.0, 1.0).astype("float32")
    validate_canonical(df)
    return df


def load_zenodo(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return load_zenodo_txt(path)
    if path.suffix.lower() == ".pcd":
        return load_zenodo_pcd(path)
    raise ValueError(f"Unsupported Zenodo file type: {path.suffix}")


# --------------------------------------------------------------------------
# Canonical parquet (produced by ingest; read by verify / Stage 1)
# --------------------------------------------------------------------------
def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_canonical(df)
    return df


# --------------------------------------------------------------------------
# Generic dispatch + optional voxel downsampling
# --------------------------------------------------------------------------
def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".txt", ".pcd"):
        return load_zenodo(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return load_parquet(path)
    raise ValueError(f"Don't know how to load {path}")


def voxel_downsample(df: pd.DataFrame, voxel_size: float) -> pd.DataFrame:
    """
    Downsample a canonical cloud with Open3D, preserving color. voxel_size is
    in the cloud's coordinate units. Returns a canonical DataFrame.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
    pcd.colors = o3d.utility.Vector3dVector(df[["r", "g", "b"]].to_numpy())
    ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    out = pd.DataFrame(
        np.hstack([np.asarray(ds.points), np.asarray(ds.colors)]),
        columns=CANONICAL_COLUMNS,
    )
    out[["r", "g", "b"]] = out[["r", "g", "b"]].astype("float32")
    return out
