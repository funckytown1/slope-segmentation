"""
Review all converted slopes at a glance.

For every canonical parquet in data/processed/zenodo/ this writes:
  - outputs/review/<slope>_views.png  -- 3 orthographic RGB projections
    (XY top-down, XZ, YZ) so the rock face reads no matter its orientation
  - a row in outputs/review/slopes_summary.md / .csv (points, extent,
    point spacing, color stats)
  - optionally a downsampled .ply per slope for a real 3D look in CloudCompare

Run after ingest:
    python scripts/review_slopes.py --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _point_spacing(xyz: np.ndarray, fit_cap: int = 2_000_000,
                   query: int = 20000) -> float:
    """
    Median nearest-neighbor spacing. For huge clouds we build the tree on a
    random subsample and correct: for a ~2D surface, spacing scales as
    1/sqrt(density), so true_spacing ~= measured * sqrt(n_sub / n_full).
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    n = len(xyz)
    if n > fit_cap:
        fit_idx = rng.choice(n, size=fit_cap, replace=False)
        fit_xyz = xyz[fit_idx]
        correction = np.sqrt(fit_cap / n)
    else:
        fit_xyz = xyz
        correction = 1.0
    q_idx = rng.choice(len(fit_xyz), size=min(query, len(fit_xyz)), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(fit_xyz)
    d, _ = nn.kneighbors(fit_xyz[q_idx])
    return float(np.median(d[:, 1]) * correction)


def _views_png(df: pd.DataFrame, name: str, path: Path, max_points: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    idx = (np.random.default_rng(0).choice(n, size=max_points, replace=False)
           if n > max_points else np.arange(n))
    sub = df.iloc[idx]
    rgb = sub[["r", "g", "b"]].to_numpy()
    pairs = [("x", "y", "top-down (XY)"), ("x", "z", "front (XZ)"),
             ("y", "z", "side (YZ)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b, title) in zip(axes, pairs):
        ax.scatter(sub[a], sub[b], c=rgb, s=0.4, marker=".", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{a} (m)"); ax.set_ylabel(f"{b} (m)")
        ax.set_title(title)
    fig.suptitle(f"{name}: {n:,} pts  ({len(sub):,} shown)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--dataset", default="zenodo_rock_slopes")
    ap.add_argument("--max-points", type=int, default=300_000,
                    help="Subsample cap for the preview scatter.")
    ap.add_argument("--ply-voxel", type=float, default=0.05,
                    help="Voxel size (m) for the review .ply; 0 = skip .ply.")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text())
    d = cfg["datasets"][args.dataset]
    proc_dir = REPO / d["output_parquet_dir"]
    slope_names = list(d["slopes"].keys())

    out_dir = REPO / "outputs" / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = REPO / "outputs" / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in slope_names:
        pq = proc_dir / f"{name}.parquet"
        if not pq.exists():
            print(f"  {name}: parquet not found ({pq}) -- skipping")
            continue
        df = pd.read_parquet(pq)
        xyz = df[["x", "y", "z"]].to_numpy()
        ext = xyz.max(0) - xyz.min(0)
        spacing = _point_spacing(xyz)
        nonblack = 100.0 * ((df[["r", "g", "b"]].to_numpy().sum(1) > 1e-3).mean())
        rows.append({
            "slope": name,
            "points": len(df),
            "extent_x_m": round(float(ext[0]), 2),
            "extent_y_m": round(float(ext[1]), 2),
            "extent_z_m": round(float(ext[2]), 2),
            "spacing_m": round(spacing, 4),
            "mean_r": round(float(df.r.mean()), 3),
            "mean_g": round(float(df.g.mean()), 3),
            "mean_b": round(float(df.b.mean()), 3),
            "pct_colored": round(nonblack, 1),
        })
        print(f"  {name}: {len(df):,} pts, extent {ext.round(2)}, "
              f"spacing ~{spacing:.3f} m")

        _views_png(df, name, out_dir / f"{name}_views.png", args.max_points)

        if args.ply_voxel > 0:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(df[["r", "g", "b"]].to_numpy())
                pcd = pcd.voxel_down_sample(args.ply_voxel)
                o3d.io.write_point_cloud(str(prev_dir / f"{name}_preview.ply"), pcd)
            except Exception as e:
                print(f"    (skipped {name} ply: {e})")

    if not rows:
        print("No parquets found. Run ingest first.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "slopes_summary.csv", index=False)
    cols = list(summary.columns)
    md = ["# Converted slopes — review summary", "",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in summary.iterrows():
        md.append("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], (int,))
                                    else str(r[c]) for c in cols) + " |")
    md += ["",
           "Views: `outputs/review/<slope>_views.png` (XY / XZ / YZ, RGB).",
           "3D: `outputs/previews/<slope>_preview.ply` (open in CloudCompare)."]
    (out_dir / "slopes_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir/'slopes_summary.md'} and per-slope *_views.png")


if __name__ == "__main__":
    main()
