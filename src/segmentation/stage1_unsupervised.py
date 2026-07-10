"""
Stage 1: unsupervised segmentation. No labels required.

Pipeline:
  load canonical cloud -> (optional voxel downsample) -> per-point features
  -> standardize -> HDBSCAN cluster -> save cluster-colored PLY + PNG.

Purpose is twofold: (1) a shippable label-free result, and (2) a labeling
canvas -- you assign a class to a whole cluster in CloudCompare instead of
painting points one by one, which is what makes the Stage 2 sparse labeling
fast.

Usage:
    python -m segmentation.stage1_unsupervised \
        --input data/processed/zenodo/Slope_6.parquet \
        --voxel 0.05 --min-cluster-size 200
(run from the src/ directory, or add src/ to PYTHONPATH)
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
from features.geometric import build_features        # noqa: E402


def run(input_path, voxel, min_cluster_size, color_norm, radius_mult):
    df = load_any(Path(input_path))
    summarize(df, name=Path(input_path).stem)

    if voxel > 0:
        df = voxel_downsample(df, voxel)
        print(f"  downsampled to {len(df):,} points (voxel {voxel} m)")

    print("Computing features...")
    feats = build_features(df, per_cloud_color_norm=color_norm, radius_mult=radius_mult)

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(feats.to_numpy())

    print("Clustering (HDBSCAN)...")
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X)
    except ImportError:
        from sklearn.cluster import KMeans
        print("  hdbscan not installed -> KMeans(k=6) fallback")
        labels = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  {n_clusters} clusters, {n_noise:,} noise points")

    df = df.copy()
    df["cluster"] = labels

    out_dir = REPO / "outputs" / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    # Save features + cluster ids for the labeling step downstream.
    proc = REPO / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    feats_out = feats.copy()
    feats_out[["x", "y", "z", "r", "g", "b"]] = df[["x", "y", "z", "r", "g", "b"]].values
    feats_out["cluster"] = labels
    feats_path = proc / f"{stem}_stage1.parquet"
    feats_out.to_parquet(feats_path, index=False)
    print(f"  wrote {feats_path} (features + clusters for labeling/Stage 2)")

    # Cluster-colored PLY
    try:
        import open3d as o3d
        rng = np.random.default_rng(1)
        palette = rng.random((max(n_clusters, 1) + 1, 3))
        cluster_colors = np.zeros((len(df), 3))
        for c in set(labels):
            cluster_colors[labels == c] = (
                np.array([0.1, 0.1, 0.1]) if c == -1 else palette[c % len(palette)]
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        ply_path = out_dir / f"{stem}_stage1_clusters.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  wrote {ply_path}")
    except Exception as e:
        print(f"  (skipped cluster PLY: {e})")

    print("\nNext: open the cluster PLY in CloudCompare, decide which clusters "
          "map to which classes, and sparse-label from there for Stage 2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="Voxel downsample size (m). 0 = no downsample.")
    ap.add_argument("--min-cluster-size", type=int, default=200)
    ap.add_argument("--no-color-norm", action="store_true",
                    help="Disable per-cloud color standardization.")
    ap.add_argument("--radius-mult", type=float, default=5.0,
                    help="Feature neighborhood radius as a multiple of point spacing.")
    args = ap.parse_args()
    run(args.input, args.voxel, args.min_cluster_size,
        not args.no_color_norm, args.radius_mult)


if __name__ == "__main__":
    main()
