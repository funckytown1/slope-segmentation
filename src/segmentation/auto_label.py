"""
Rule-based weak labeling: per-point geohazard classes from geometry + color.

This is the "initial labeling" step, done automatically instead of by hand in
CloudCompare. It runs directly on a canonical cloud: it computes per-point
features (or reuses them if already present), applies a transparent set of
rules, and writes a labeled parquet + class-colored previews you can review
before committing to the supervised (Random Forest) stage.

Decision logic (per point):
  1. vegetation      -- color gate: green excess (g clearly above r and b).
                        The one class color nails directly.
  2. shadow/background-- brightness gate: very dark points (low HSV value):
                        deep recesses, occlusion, non-rock background.
  3. the remaining rock face is split by *geometry*, scored per point with
     features z-scored across the rock points (so thresholds adapt to each
     cloud instead of being hard-coded):
       intact_rock              smooth + planar + steep
       discontinuity_or_fracture rough / edgy / broken
       loose_block_or_debris    blobby + low on the slope (talus)

Usage (from src/):
    python -m segmentation.auto_label \
        --input ../data/processed/zenodo/slope_6.parquet --voxel 0
    python -m segmentation.auto_label \
        --input ../data/processed/zenodo/slope_4.parquet --voxel 0.01
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pcio.loaders import load_any, voxel_downsample   # noqa: E402
from pcio.canonical import summarize                   # noqa: E402
from features.geometric import build_features, GEOMETRIC_FEATURES  # noqa: E402

CLASSES = [
    "intact_rock", "discontinuity_or_fracture", "loose_block_or_debris",
    "vegetation", "shadow_or_background",
]
CLASS_COLORS = {
    "intact_rock":               (0.55, 0.55, 0.58),   # gray
    "discontinuity_or_fracture": (0.85, 0.20, 0.20),   # red
    "loose_block_or_debris":     (0.95, 0.70, 0.15),   # amber
    "vegetation":                (0.20, 0.65, 0.25),   # green
    "shadow_or_background":      (0.20, 0.25, 0.38),   # dark blue-gray
}

# Vegetation gate: brightness-invariant Excess-Green (ExG = (2g-r-b)/(r+g+b)),
# *centered on each cloud's own median*. A point is vegetation only if it is
# greener than the typical surface of THAT slope -- not greener than a fixed
# constant. This is what lets it survive a slope whose rock has a green color
# cast (e.g. slope_3), where an absolute green threshold mislabels the whole
# face. (Absolute thresholds not transferring across slopes is exactly the
# color-doesn't-transfer thesis the pipeline is built around.)
VEG_EXG_MARGIN = 0.05       # how much greener than the cloud's median surface
VEG_EXG_FLOOR = 0.08        # absolute ExG floor, so reddish rock can't sneak in
VEG_MIN_SAT = 0.12          # real vegetation is saturated; desaturated cast is not
SHADOW_MAX_VALUE = 0.22     # HSV value below this = shadow / recess / background

# Geometry scoring weights for the rock-face split (multiply z-scored feats).
# "debris_height" is a low-is-debris height term: local height (m above local
# ground) when the local_height feature is present, else global normalized
# height as a fallback.
W_INTACT = {"planarity": 1.0, "surface_variation": -0.7, "verticality": 0.5}
W_FRACTURE = {"surface_variation": 1.0, "linearity": 0.7, "planarity": -0.5}
W_DEBRIS = {"sphericity": 1.0, "debris_height": -0.8, "verticality": -0.3}

# Horizontal cell size (m) for the scale-invariant local-height feature. Debris
# (talus) sits within a physical ~1-2 m of local ground regardless of whether
# the slope is 3 m or 130 m tall, so "metres above local ground" transfers
# across scales where global normalized height does not.
LOCAL_HEIGHT_CELL = 3.0


def add_local_height(df: pd.DataFrame, cell: float = LOCAL_HEIGHT_CELL) -> pd.DataFrame:
    """
    Add a scale-invariant `local_height` column: each point's height above the
    minimum z in its horizontal (x,y) grid cell of size `cell` metres. Grid
    binning is O(n) -- fast even on ~20M points -- and features-only (no
    jakteristics recompute). Stored so the Random Forest can use it too.
    """
    z = df["z"].to_numpy()
    ix = np.floor(df["x"].to_numpy() / cell).astype(np.int64)
    iy = np.floor(df["y"].to_numpy() / cell).astype(np.int64)
    cell_min = pd.DataFrame({"ix": ix, "iy": iy, "z": z}) \
        .groupby(["ix", "iy"])["z"].transform("min").to_numpy()
    df["local_height"] = (z - cell_min).astype("float32")
    return df


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-9 else pd.Series(0.0, index=s.index)


def rule_label_points(df: pd.DataFrame) -> pd.Series:
    """
    df: per-point rows with geometric features + hue/sat/val + x,y,z,r,g,b.
    Returns a per-point label_name Series (one of CLASSES).
    """
    n = len(df)
    # Debris height term: scale-invariant local height if available, else the
    # global normalized height fallback (see W_DEBRIS / add_local_height).
    if "local_height" in df.columns:
        debris_height = df["local_height"]
    else:
        z = df["z"].to_numpy()
        zr = float(z.max() - z.min()) or 1.0
        debris_height = pd.Series((z - z.min()) / zr, index=df.index)

    value = df["val"] if "val" in df.columns else df[["r", "g", "b"]].max(axis=1)
    if "sat" in df.columns:
        sat = df["sat"]
    else:
        mx = df[["r", "g", "b"]].max(axis=1); mn = df[["r", "g", "b"]].min(axis=1)
        sat = (mx - mn) / mx.clip(lower=1e-9)

    # Brightness-invariant Excess-Green, centered on this cloud's median so the
    # gate adapts to a global color cast instead of firing on greenish rock.
    total = (df["r"] + df["g"] + df["b"]).clip(lower=1e-6)
    exg = (2 * df["g"] - df["r"] - df["b"]) / total
    exg_centered = exg - exg.median()

    label = pd.Series(index=df.index, dtype=object)

    is_veg = (exg_centered >= VEG_EXG_MARGIN) & (exg >= VEG_EXG_FLOOR) & (sat >= VEG_MIN_SAT)
    label[is_veg] = "vegetation"

    is_shadow = (~is_veg) & (value <= SHADOW_MAX_VALUE)
    label[is_shadow] = "shadow_or_background"

    rock = label.isna()
    if rock.any():
        rdf = df[rock]
        zc = {}
        for c in ["planarity", "linearity", "sphericity", "surface_variation",
                  "verticality"]:
            if c in rdf.columns:
                zc[c] = _zscore(rdf[c])
        zc["debris_height"] = _zscore(debris_height[rock])

        def score(weights):
            s = pd.Series(0.0, index=rdf.index)
            for feat, w in weights.items():
                if feat in zc:
                    s = s + w * zc[feat]
            return s

        scores = pd.DataFrame({
            "intact_rock": score(W_INTACT),
            "discontinuity_or_fracture": score(W_FRACTURE),
            "loose_block_or_debris": score(W_DEBRIS),
        })
        label[rock] = scores.idxmax(axis=1).values
    return label


def render_class_views(df: pd.DataFrame, labels: pd.Series, name: str,
                       path: Path, max_points: int = 300_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(df)
    idx = (np.random.default_rng(0).choice(n, size=max_points, replace=False)
           if n > max_points else np.arange(n))
    sub = df.iloc[idx]
    lab = labels.iloc[idx]
    cols = np.array([CLASS_COLORS.get(l, (0.1, 0.1, 0.1)) for l in lab])
    pairs = [("x", "y", "top-down (XY)"), ("x", "z", "front (XZ)"),
             ("y", "z", "side (YZ)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b, title) in zip(axes, pairs):
        ax.scatter(sub[a], sub[b], c=cols, s=0.4, marker=".", linewidths=0)
        ax.set_aspect("equal"); ax.set_xlabel(f"{a} (m)"); ax.set_ylabel(f"{b} (m)")
        ax.set_title(title)
    present = [c for c in CLASSES if (labels == c).any()]
    handles = [Patch(facecolor=CLASS_COLORS[c], label=c) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(present),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{name} ({n:,} pts)", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_class_ply(df: pd.DataFrame, labels: pd.Series, path: Path) -> None:
    try:
        import open3d as o3d
    except Exception as e:
        print(f"  (skipped class PLY: {e})"); return
    cols = np.array([CLASS_COLORS.get(l, (0.1, 0.1, 0.1)) for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  wrote {path}")


def run(input_path: str, voxel: float, radius_mult: float, out_dir: Path) -> None:
    df = load_any(Path(input_path))
    stem = Path(input_path).stem
    summarize(df, name=stem)

    if voxel > 0:
        df = voxel_downsample(df, voxel)
        print(f"  downsampled to {len(df):,} points (voxel {voxel} m)")

    have_feats = all(c in df.columns for c in GEOMETRIC_FEATURES)
    if not have_feats:
        print("Computing per-point features...")
        feats = build_features(df, per_cloud_color_norm=True, radius_mult=radius_mult)
        df = pd.concat([df.reset_index(drop=True),
                        feats.reset_index(drop=True)], axis=1)

    labels = rule_label_points(df).astype(str)
    dist = labels.value_counts()
    total = int(dist.sum())
    print(f"\nRule-based draft for {stem} ({total:,} points):")
    for cls in CLASSES:
        nc = int(dist.get(cls, 0))
        print(f"  {cls:<28} {nc:>10,} ({100*nc/total:4.1f}%)")

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["label_name"] = labels.values
    pq = out_dir / f"{stem}_rules.parquet"
    df_out.to_parquet(pq, index=False)
    print(f"  wrote {pq}")

    render_class_views(df, labels, f"{stem}: rule-based draft segmentation",
                       out_dir / f"{stem}_rules_views.png")
    print(f"  wrote {out_dir / f'{stem}_rules_views.png'}")
    save_class_ply(df, labels, out_dir / f"{stem}_rules.ply")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="canonical slope parquet")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel downsample size (m); 0 = full resolution")
    ap.add_argument("--radius-mult", type=float, default=5.0)
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "rules"))
    args = ap.parse_args()
    run(args.input, args.voxel, args.radius_mult, Path(args.out_dir))


if __name__ == "__main__":
    main()
