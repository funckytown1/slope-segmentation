"""
Verify a canonical colorized cloud (parquet, or a Zenodo .txt/.pcd directly).

Prints a summary and writes two previews to outputs/previews/:
  - a fast top-down RGB scatter PNG (no GUI needed)
  - a downsampled .ply to open in CloudCompare for a real 3D look

Usage:
    python scripts/verify_colorized_cloud.py --input data/processed/zenodo/Slope_6.parquet
    python scripts/verify_colorized_cloud.py --input data/raw/zenodo_slopes/Slope_6.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Make src importable when run as a script from repo root.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcio.canonical import summarize        
from pcio.loaders import load_any             


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-preview-points", type=int, default=400_000,
                    help="Subsample cap for the PNG scatter (speed).")
    ap.add_argument("--ply-voxel", type=float, default=0.0,
                    help="Voxel size (m) for the .ply preview; 0 = no downsample.")
    args = ap.parse_args()

    in_path = Path(args.input)
    df = load_any(in_path)
    name = in_path.stem
    summarize(df, name=name)

    valid_color = (df[["r", "g", "b"]].to_numpy().std(axis=1) > 1e-4) | \
                  (df[["r", "g", "b"]].to_numpy().sum(axis=1) > 1e-4)
    pct = 100.0 * valid_color.mean()
    print(f"  points with non-black color: {pct:.1f}%")

    out_dir = REPO / "outputs" / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- PNG top-down scatter ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    idx = np.random.default_rng(0).choice(
        n, size=min(args.max_preview_points, n), replace=False
    )
    sub = df.iloc[idx]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(sub.x, sub.y, c=sub[["r", "g", "b"]].to_numpy(), s=0.5, marker=".")
    ax.set_aspect("equal")
    ax.set_title(f"{name}: top-down RGB ({len(sub):,} of {n:,} pts)")
    png_path = out_dir / f"{name}_topdown.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")

    # --- PLY for CloudCompare ---
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
        pcd.colors = o3d.utility.Vector3dVector(df[["r", "g", "b"]].to_numpy())
        if args.ply_voxel > 0:
            pcd = pcd.voxel_down_sample(args.ply_voxel)
        ply_path = out_dir / f"{name}_preview.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  wrote {ply_path}  ({len(pcd.points):,} pts)")
    except Exception as e:
        print(f"  (skipped .ply: {e})")

    print("\nCheckpoint: open the PNG/PLY. If it reads as a real rock face "
          "(rock, cracks, vegetation legible), proceed to Stage 1 features.")


if __name__ == "__main__":
    main()
