"""
Leave-one-slope-out (LOSO) cross-validation for the rock-slope segmentation.

Every slope is the unseen site exactly once: for each held-out slope we train
a Random Forest on the *other* slopes' rule-labeled points and predict the
held-out one. This measures cross-site generalization without cherry-picking a
split, and ranks the slopes by how hard each is to predict when unseen.

Reuses the per-point feature+label parquets written by the rules draft
(outputs/rules/<slope>_rules.parquet) -- no feature recomputation.

Memory-lean by construction: the big clouds (slope_3 is ~19.5M points) are
never all held in RAM at once. Each slope is read once, downcast to float32,
and reduced to a small capped training sample + a capped eval sample; the full
held-out cloud is streamed back from disk only when writing its PLY.

Two honesty details:
  * points-per-slope cap so a big slope can't dominate "train on all but one".
  * two macro-F1s per fold: all classes, and geometry-only (intact / fracture
    / debris), so the color-driven vegetation/shadow classes -- which hinge on
    whether a rare class was even present in training -- don't mask whether the
    *geometry* transfers across sites.

Usage:
    python scripts/loso_experiment.py --rules-dir outputs/rules \
        --train-cap 80000 --eval-cap 300000 --save-ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from segmentation.auto_label import CLASSES, CLASS_COLORS, render_class_views 
from segmentation.labeling import NON_FEATURE_COLS   

GEOM_CLASSES = ["intact_rock", "discontinuity_or_fracture", "loose_block_or_debris"]
NAME_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_NAME = {i: c for c, i in NAME_TO_ID.items()}


def _feature_columns(path: Path) -> list[str]:
    """Feature column names from the parquet schema (no data loaded)."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema.names
    return [c for c in names if c not in NON_FEATURE_COLS]


def _sample_slope(path: Path, feat_cols, train_cap, eval_cap, seed):
    """
    Read one rules parquet and return small capped arrays, freeing the full
    frame immediately. Returns:
      Xtr (train_cap, F) float32, ytr int8,
      Xev (eval_cap, F) float32, yev int8, xyz_ev (eval_cap, 3) float64
    """
    cols = feat_cols + ["label_name", "x", "y", "z"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["label_name"].isin(NAME_TO_ID)]
    y = df["label_name"].map(NAME_TO_ID).to_numpy(np.int8)
    X = df[feat_cols].to_numpy(np.float32)
    xyz = df[["x", "y", "z"]].to_numpy(np.float64)
    del df
    rng = np.random.default_rng(seed)
    n = len(X)

    def pick(cap):
        return rng.choice(n, size=cap, replace=False) if n > cap else np.arange(n)

    itr, iev = pick(train_cap), pick(eval_cap)
    out = (X[itr].copy(), y[itr].copy(),
           X[iev].copy(), y[iev].copy(), xyz[iev].copy())
    del X, y, xyz
    return out


def _plot_bar(names, values, title, ylabel, path, color="#3b6fb0"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(values)
    names = np.array(names)[order]; values = np.array(values)[order]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names)), 4))
    ax.bar(range(len(values)), values, color=color)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def _plot_perclass_heatmap(mat, slopes, classes, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(classes), 1.5 + 0.5 * len(slopes)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(slopes))); ax.set_yticklabels(slopes)
    ax.set_title("Per-class F1 by held-out slope")
    for i in range(len(slopes)):
        for j in range(len(classes)):
            v = mat[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}")


def _write_ply_for_held(path, feat_cols, clf, full_ply, ply_max_points,
                        out_ply, out_png_title, out_png):
    """Stream the full held-out cloud from disk, predict, write PLY + a
    full-cloud PNG. Kept separate so the big frame is loaded and freed here."""
    cols = feat_cols + ["x", "y", "z"]
    df = pd.read_parquet(path, columns=cols)
    n = len(df)
    if full_ply or n <= ply_max_points:
        note = f"{n:,} pts (native)"
    else:
        idx = np.random.default_rng(0).choice(n, size=ply_max_points, replace=False)
        df = df.iloc[np.sort(idx)]
        note = f"{len(df):,} pts (random subsample of {n:,})"
    pred = clf.predict(df[feat_cols].to_numpy(np.float32))
    lab = pd.Series([ID_TO_NAME[int(i)] for i in pred], index=df.index)
    try:
        import open3d as o3d
        colarr = np.array([CLASS_COLORS[c] for c in lab])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
        pcd.colors = o3d.utility.Vector3dVector(colarr)
        o3d.io.write_point_cloud(str(out_ply), pcd)
        print(f"  wrote {out_ply.name}  [{note}]")
    except Exception as e:
        print(f"  (skipped ply: {e})")
    render_class_views(df.reset_index(drop=True), lab.reset_index(drop=True),
                       out_png_title, out_png)
    del df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-dir", default="outputs/rules")
    ap.add_argument("--out-dir", default="outputs/loso")
    ap.add_argument("--train-cap", type=int, default=80000,
                    help="max points contributed per training slope (balance).")
    ap.add_argument("--eval-cap", type=int, default=300000,
                    help="max held-out points used for metrics (speed/memory).")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--save-ply", action="store_true",
                    help="also save a predicted held-out segmentation PLY per fold.")
    ap.add_argument("--ply-max-points", type=int, default=2_000_000,
                    help="held-out slopes above this are randomly subsampled to this "
                         "many points for the viewing PLY instead of native resolution.")
    ap.add_argument("--full-ply", action="store_true",
                    help="force native-resolution PLYs for ALL folds (slower, large).")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, confusion_matrix, classification_report

    rules_dir = REPO / args.rules_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {pq.stem.replace("_rules", ""): pq
             for pq in sorted(rules_dir.glob("*_rules.parquet"))}
    slopes = list(paths)
    if len(slopes) < 2:
        raise SystemExit("Need >=2 rules parquets. Run the rules draft on more slopes.")
    feat_cols = _feature_columns(paths[slopes[0]])
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    print("Sampling slopes (capped, memory-lean)...")
    samples = {}
    for i, s in enumerate(slopes):
        samples[s] = _sample_slope(paths[s], feat_cols, args.train_cap,
                                   args.eval_cap, seed=i)
        Xtr, ytr, Xev, yev, _ = samples[s]
        print(f"  {s}: train {len(Xtr):,}, eval {len(Xev):,}")

    geom_ids = [NAME_TO_ID[c] for c in GEOM_CLASSES]
    all_ids = [NAME_TO_ID[c] for c in CLASSES]

    results = {}
    perclass = np.full((len(slopes), len(CLASSES)), np.nan)

    for si, held in enumerate(slopes):
        Xtr = np.concatenate([samples[s][0] for s in slopes if s != held])
        ytr = np.concatenate([samples[s][1] for s in slopes if s != held])
        clf = RandomForestClassifier(n_estimators=args.trees, n_jobs=-1,
                                     class_weight="balanced", random_state=0)
        clf.fit(Xtr, ytr)
        del Xtr, ytr

        _, _, Xev, yev, xyz_ev = samples[held]
        yp = clf.predict(Xev)

        macro_all = f1_score(yev, yp, labels=all_ids, average="macro", zero_division=0)
        macro_geom = f1_score(yev, yp, labels=geom_ids, average="macro", zero_division=0)
        rep = classification_report(yev, yp, labels=all_ids, target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        for ci, c in enumerate(CLASSES):
            perclass[si, ci] = rep[c]["f1-score"] if rep[c]["support"] > 0 else np.nan
        cm = confusion_matrix(yev, yp, labels=all_ids)
        results[held] = {
            "n_train": int(sum(len(samples[s][0]) for s in slopes if s != held)),
            "n_test_eval": int(len(Xev)),
            "macro_f1_all": float(macro_all),
            "macro_f1_geometry_only": float(macro_geom),
            "per_class_f1": {c: (None if np.isnan(perclass[si, ci]) else float(perclass[si, ci]))
                             for ci, c in enumerate(CLASSES)},
            "confusion_labels": CLASSES, "confusion": cm.tolist(),
        }
        print(f"[held-out {held}] macro-F1 all={macro_all:.3f}  "
              f"geometry-only={macro_geom:.3f}")

        # Free per-fold prediction PNG from the eval sample (has xyz).
        lab_eval = pd.Series([ID_TO_NAME[int(i)] for i in yp])
        ev_df = pd.DataFrame(xyz_ev, columns=["x", "y", "z"])
        render_class_views(ev_df, lab_eval,
                           f"{held}: RF prediction when held out (eval sample)",
                           out_dir / f"{held}_loso_pred_views.png")

        if args.save_ply:
            _write_ply_for_held(
                paths[held], feat_cols, clf, args.full_ply, args.ply_max_points,
                out_dir / f"{held}_loso_pred.ply",
                f"{held}: RF prediction when held out (full cloud)",
                out_dir / f"{held}_loso_pred_full_views.png")
        del clf

    _plot_bar(slopes, [results[s]["macro_f1_all"] for s in slopes],
              "LOSO generalization: macro-F1 by held-out slope (all classes)",
              "macro-F1", out_dir / "loso_difficulty_all.png")
    _plot_bar(slopes, [results[s]["macro_f1_geometry_only"] for s in slopes],
              "LOSO: macro-F1 by held-out slope (geometry-only classes)",
              "macro-F1", out_dir / "loso_difficulty_geom.png", color="#7a4fb0")
    _plot_perclass_heatmap(perclass, slopes, CLASSES, out_dir / "loso_perclass_f1.png")

    (out_dir / "loso_metrics.json").write_text(json.dumps(results, indent=2))
    _write_md(results, slopes, out_dir / "loso_metrics.md")
    print(f"\nWrote {out_dir/'loso_metrics.md'} + difficulty/per-class figures")


def _write_md(results, slopes, path):
    ranked = sorted(slopes, key=lambda s: results[s]["macro_f1_all"])
    lines = ["# Leave-one-slope-out (LOSO) cross-validation", "",
             "Each slope held out once; RF trained on the others' rule labels "
             "(capped per slope for balance).", "",
             "| held-out slope | macro-F1 (all) | macro-F1 (geometry-only) | train pts |",
             "|---|---|---|---|"]
    for s in slopes:
        r = results[s]
        lines.append(f"| {s} | {r['macro_f1_all']:.3f} | "
                     f"{r['macro_f1_geometry_only']:.3f} | {r['n_train']:,} |")
    lines += ["",
              f"**Hardest held-out (all classes): `{ranked[0]}` "
              f"({results[ranked[0]]['macro_f1_all']:.3f})** — most out-of-distribution.",
              f"**Easiest: `{ranked[-1]}` ({results[ranked[-1]]['macro_f1_all']:.3f})**.",
              "",
              "Per-class F1 by held-out slope: see `loso_perclass_f1.png` "
              "(blank = class absent in that slope's truth). Vegetation/shadow "
              "F1 depends on whether the class was present in the *training* "
              "slopes, which is a data-coverage effect, not a geometry-transfer "
              "one — hence the geometry-only column."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
