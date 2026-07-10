"""
Stage 2: supervised Random Forest on the labeled points.

Produces the headline deliverable artifacts:
  - in-slope metrics (held-out split on the training slope)
  - cross-slope metrics (train on slope_a, test on slope_b's labeled slice)
  - confusion matrix, per-class F1 bar, feature-importance bar (PNGs)
  - full class-colored segmentation PLY for each slope
  - metrics.json + metrics.md summary you can paste into the report/slides

Two-step usage:

  # 1. make a fill-in label template from Stage 1 clusters
  python -m segmentation.stage2_supervised make-template \
      --stage1 ../data/processed/slope_a_stage1.parquet \
      --out ../configs/labels_slope_a.yaml

  # ... fill in classes in the YAML using the cluster PLY ...

  # 2. train + evaluate
  python -m segmentation.stage2_supervised train \
      --train-stage1 ../data/processed/slope_a_stage1.parquet \
      --train-labels ../configs/labels_slope_a.yaml \
      --test-stage1  ../data/processed/slope_b_stage1.parquet \
      --test-labels  ../configs/labels_slope_b.yaml \
      --config ../configs/dataset.yaml

--test-* are optional; omit them to skip the cross-slope evaluation (you'll
still get a predicted segmentation PLY for slope_b if you pass --test-stage1
without --test-labels).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from segmentation.labeling import (      # noqa: E402
    make_template, apply_cluster_map, labels_from_cloud,
    feature_columns, class_registry,
)

OUT = REPO / "outputs"


def _load_config_classes(config_path):
    if not config_path:
        return None
    cfg = yaml.safe_load(Path(config_path).read_text())
    return cfg.get("classes")


def _plot_confusion(cm, class_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(class_names),
                                    1.4 + 1.0 * len(class_names)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _plot_bar(labels, values, title, ylabel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(values)[::-1]
    labels = np.array(labels)[order]
    values = np.array(values)[order]
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(labels)), 4))
    ax.bar(range(len(values)), values, color="#3b6fb0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _render_topdown_classes(df, pred_ids, registry, title, path,
                            max_points=500_000):
    """
    Class-colored top-down scatter (matplotlib only -- no open3d, no GUI), so
    there's always a viewable segmentation result even without CloudCompare.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    id_to_color = {registry["name_to_id"][n]: registry["name_to_color"][n]
                   for n in registry["names"]}
    n = len(df)
    idx = (np.random.default_rng(0).choice(n, size=max_points, replace=False)
           if n > max_points else np.arange(n))
    xy = df.iloc[idx][["x", "y"]].to_numpy()
    cols = np.array([id_to_color.get(int(i), (0.1, 0.1, 0.1))
                     for i in np.asarray(pred_ids)[idx]])

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=0.5, marker=".", linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    present = sorted(set(int(i) for i in np.asarray(pred_ids)[idx]))
    id_to_name = {v: k for k, v in registry["name_to_id"].items()}
    handles = [Patch(facecolor=id_to_color.get(i, (0.1, 0.1, 0.1)),
                     label=id_to_name.get(i, str(i))) for i in present]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _save_segmentation_ply(df, pred_ids, registry, path):
    try:
        import open3d as o3d
    except Exception as e:
        print(f"  (skipped segmentation PLY: {e})")
        return
    id_to_color = {registry["name_to_id"][n]: registry["name_to_color"][n]
                   for n in registry["names"]}
    colors = np.array([id_to_color.get(int(i), (0.1, 0.1, 0.1)) for i in pred_ids])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].to_numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  wrote {path}")


def cmd_make_template(args):
    config_classes = _load_config_classes(args.config)
    hint = [c["name"] for c in config_classes] if config_classes else None
    make_template(args.stage1, args.out, class_hint=hint)


def _get_labeled(stage1, labels_yaml, labeled_cloud, class_map):
    if labels_yaml:
        return apply_cluster_map(stage1, labels_yaml)
    if labeled_cloud:
        return labels_from_cloud(stage1, labeled_cloud, class_map)
    raise SystemExit("Provide --train-labels (cluster map) or --labeled-cloud.")


def cmd_train(args):
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score)

    config_classes = _load_config_classes(args.config)

    # --- training slope ---
    train_df = _get_labeled(args.train_stage1, args.train_labels,
                            args.train_labeled_cloud, None)
    lab = train_df[train_df["label_name"].notna()].copy()
    if len(lab) == 0:
        raise SystemExit("No labeled points found -- fill in the label template first.")

    present = sorted(lab["label_name"].unique())
    reg = class_registry(config_classes, present)
    feat_cols = feature_columns(train_df)
    print(f"Features ({len(feat_cols)}): {feat_cols}")

    lab["y"] = lab["label_name"].map(reg["name_to_id"])
    X = lab[feat_cols].to_numpy()
    y = lab["y"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)

    # class names in id order, restricted to those present
    present_ids = sorted(np.unique(y))
    id_to_name = {v: k for k, v in reg["name_to_id"].items()}
    present_names = [id_to_name[i] for i in present_ids]

    # --- in-slope metrics ---
    yp = clf.predict(Xte)
    report_in = classification_report(
        yte, yp, labels=present_ids, target_names=present_names,
        output_dict=True, zero_division=0)
    cm_in = confusion_matrix(yte, yp, labels=present_ids)
    macro_in = f1_score(yte, yp, labels=present_ids, average="macro", zero_division=0)
    print(f"\nIn-slope macro-F1: {macro_in:.3f}")

    _plot_confusion(cm_in, present_names, "In-slope confusion (row-normalized)",
                    OUT / "confusion_in_slope.png")
    per_class_f1 = [report_in[n]["f1-score"] for n in present_names]
    _plot_bar(present_names, per_class_f1, "In-slope per-class F1", "F1",
              OUT / "f1_in_slope.png")
    _plot_bar(feat_cols, clf.feature_importances_,
              "Random Forest feature importance", "importance",
              OUT / "feature_importance.png")

    metrics = {
        "train_slope": Path(args.train_stage1).stem,
        "n_labeled_train": int(len(lab)),
        "classes": present_names,
        "in_slope": {"macro_f1": float(macro_in),
                     "per_class_f1": dict(zip(present_names, per_class_f1)),
                     "report": report_in},
    }

    # --- full-slope predicted segmentation PLY (train slope) ---
    all_pred = clf.predict(train_df[feat_cols].to_numpy())
    train_stem = Path(args.train_stage1).stem
    _save_segmentation_ply(train_df, all_pred, reg,
                           OUT / f"{train_stem}_segmented.ply")
    _render_topdown_classes(train_df, all_pred, reg,
                            f"{train_stem}: predicted segmentation (top-down)",
                            OUT / f"{train_stem}_segmented_topdown.png")

    # --- cross-slope evaluation ---
    if args.test_stage1:
        test_df = pd.read_parquet(args.test_stage1)
        test_feat = test_df[feat_cols].to_numpy()
        test_pred = clf.predict(test_feat)
        test_stem = Path(args.test_stage1).stem
        _save_segmentation_ply(test_df, test_pred, reg,
                               OUT / f"{test_stem}_segmented.ply")
        _render_topdown_classes(test_df, test_pred, reg,
                                f"{test_stem}: predicted segmentation (top-down)",
                                OUT / f"{test_stem}_segmented_topdown.png")

        if args.test_labels or args.test_labeled_cloud:
            tl = _get_labeled(args.test_stage1, args.test_labels,
                              args.test_labeled_cloud, None)
            tl = tl[tl["label_name"].notna()].copy()
            tl = tl[tl["label_name"].isin(reg["name_to_id"])]
            if len(tl):
                yt = tl["label_name"].map(reg["name_to_id"]).to_numpy()
                yq = clf.predict(tl[feat_cols].to_numpy())
                ids_x = sorted(np.unique(np.concatenate([yt, present_ids])))
                names_x = [id_to_name[i] for i in ids_x]
                macro_x = f1_score(yt, yq, average="macro", zero_division=0)
                cm_x = confusion_matrix(yt, yq, labels=ids_x)
                _plot_confusion(cm_x, names_x,
                                "Cross-slope confusion (row-normalized)",
                                OUT / "confusion_cross_slope.png")
                rep_x = classification_report(
                    yt, yq, target_names=[id_to_name[i] for i in sorted(np.unique(yt))],
                    output_dict=True, zero_division=0)
                metrics["cross_slope"] = {
                    "test_slope": Path(args.test_stage1).stem,
                    "n_labeled_test": int(len(tl)),
                    "macro_f1": float(macro_x),
                    "report": rep_x,
                }
                print(f"Cross-slope macro-F1: {macro_x:.3f}  "
                      f"(vs {macro_in:.3f} in-slope -- expect a drop; that's the "
                      f"generalization story)")

    # --- write metrics.json + metrics.md ---
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_metrics_md(metrics, OUT / "metrics.md")
    print(f"\nWrote {OUT/'metrics.json'} and {OUT/'metrics.md'}")
    print("Artifacts in outputs/: confusion_*.png, f1_in_slope.png, "
          "feature_importance.png, *_segmented.ply")


def _write_metrics_md(m, path):
    lines = ["# Stage 2 results", ""]
    lines.append(f"- Train slope: `{m['train_slope']}`  "
                 f"({m['n_labeled_train']:,} labeled points)")
    lines.append(f"- Classes: {', '.join(m['classes'])}")
    lines.append(f"- **In-slope macro-F1: {m['in_slope']['macro_f1']:.3f}**")
    lines.append("")
    lines.append("| Class | In-slope F1 |")
    lines.append("|---|---|")
    for n, f in m["in_slope"]["per_class_f1"].items():
        lines.append(f"| {n} | {f:.3f} |")
    if "cross_slope" in m:
        cs = m["cross_slope"]
        lines += ["",
                  f"- Test slope (cross-slope): `{cs['test_slope']}`  "
                  f"({cs['n_labeled_test']:,} labeled points)",
                  f"- **Cross-slope macro-F1: {cs['macro_f1']:.3f}**",
                  "",
                  "> Cross-slope < in-slope is expected: the two slopes differ in "
                  "lithology and lighting, so a model fit to one doesn't fully "
                  "transfer. Geometry-based features transfer better than color; "
                  "this gap is the deployment-realism finding to highlight."]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("make-template")
    t.add_argument("--stage1", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config", default=None)
    t.set_defaults(func=cmd_make_template)

    r = sub.add_parser("train")
    r.add_argument("--train-stage1", required=True)
    r.add_argument("--train-labels", default=None, help="cluster-map YAML")
    r.add_argument("--train-labeled-cloud", default=None, help="exported labeled cloud")
    r.add_argument("--test-stage1", default=None)
    r.add_argument("--test-labels", default=None)
    r.add_argument("--test-labeled-cloud", default=None)
    r.add_argument("--config", default=None, help="dataset.yaml for class order/colors")
    r.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
