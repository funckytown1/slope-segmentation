"""
Turn Stage 1 clusters into point labels for supervised training.

Primary path (fast, recommended): label whole clusters.
  1. Stage 1 wrote <slope>_stage1.parquet with a `cluster` column.
  2. `make_template()` emits a YAML listing every cluster id + its point
     count; you fill in a class name per cluster (looking at the cluster
     PLY in CloudCompare).
  3. `apply_cluster_map()` joins that mapping back to get a `label_name`
     per point. Unmapped/blank clusters (incl. noise -1) stay unlabeled
     and are excluded from training.

Fallback path (manual refinement): if you hand-paint labels in CloudCompare
and export a cloud with a per-point class scalar field, `labels_from_cloud()`
matches those labels onto the Stage 1 points by nearest neighbor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


NON_FEATURE_COLS = {"x", "y", "z", "r", "g", "b", "cluster",
                    "label", "label_name", "slope"}

# Stable fallback palette (RGB 0-1) if a class isn't in the config list.
_FALLBACK_PALETTE = [
    (0.85, 0.20, 0.20), (0.20, 0.55, 0.85), (0.25, 0.70, 0.30),
    (0.90, 0.70, 0.15), (0.55, 0.35, 0.75), (0.45, 0.45, 0.45),
    (0.90, 0.45, 0.75), (0.35, 0.75, 0.75),
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = everything that isn't metadata."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# --------------------------------------------------------------------------
# Class registry (name <-> id <-> color), seeded from configs/dataset.yaml
# --------------------------------------------------------------------------
def class_registry(config_classes: list[dict] | None,
                   present_names: list[str]) -> dict:
    """
    Build a stable name->id and name->color mapping. Config order fixes ids
    and colors; any labeled class not in config is appended.
    """
    ordered = []
    if config_classes:
        ordered = [c["name"] for c in config_classes]
    for n in present_names:
        if n not in ordered:
            ordered.append(n)

    name_to_id = {n: i for i, n in enumerate(ordered)}
    name_to_color = {
        n: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
        for i, n in enumerate(ordered)
    }
    return {"names": ordered, "name_to_id": name_to_id,
            "name_to_color": name_to_color}


# --------------------------------------------------------------------------
# Primary path: cluster -> class
# --------------------------------------------------------------------------
def make_template(stage1_parquet: str | Path, out_yaml: str | Path,
                  class_hint: list[str] | None = None) -> None:
    df = pd.read_parquet(stage1_parquet)
    if "cluster" not in df.columns:
        raise ValueError(f"{stage1_parquet} has no 'cluster' column -- run Stage 1 first.")
    counts = df["cluster"].value_counts().sort_index()

    hint = ", ".join(class_hint) if class_hint else "your class names"
    lines = [
        f"# Cluster -> class map for {Path(stage1_parquet).stem}",
        f"# Fill each cluster's class from: {hint}",
        "# Leave blank or null to exclude a cluster (e.g. noise -1, or mixed clusters).",
        "map:",
    ]
    for cid, n in counts.items():
        lines.append(f'  {int(cid)}: ""   # {int(n):,} pts')
    Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(out_yaml).write_text("\n".join(lines) + "\n")
    print(f"Wrote label template -> {out_yaml}")
    print("Open the Stage 1 cluster PLY in CloudCompare, then fill in the classes.")


def apply_cluster_map(stage1_parquet: str | Path,
                      label_yaml: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(stage1_parquet)
    mapping_raw = yaml.safe_load(Path(label_yaml).read_text()).get("map", {})
    # Keep only non-empty class names.
    mapping = {int(k): str(v).strip() for k, v in mapping_raw.items()
               if v is not None and str(v).strip() != ""}
    df["label_name"] = df["cluster"].map(mapping)
    n_labeled = df["label_name"].notna().sum()
    print(f"  {Path(stage1_parquet).stem}: {n_labeled:,} labeled points across "
          f"{df['label_name'].nunique()} classes "
          f"({sorted(df['label_name'].dropna().unique())})")
    return df


# --------------------------------------------------------------------------
# Fallback path: labels from an exported labeled cloud (nearest-neighbor)
# --------------------------------------------------------------------------
def labels_from_cloud(stage1_parquet: str | Path, labeled_cloud: str | Path,
                      class_names: dict[int, str],
                      label_field_index: int = 3) -> pd.DataFrame:
    """
    Match a CloudCompare-exported labeled cloud onto Stage 1 points.

    labeled_cloud: whitespace .txt/.asc where each row is X Y Z <class_id ...>;
    label_field_index is the column of the integer class id (default 3).
    class_names: {integer_id: "class_name"}.
    """
    from scipy.spatial import cKDTree

    df = pd.read_parquet(stage1_parquet)
    arr = np.loadtxt(labeled_cloud)
    lab_xyz = arr[:, :3]
    lab_id = arr[:, label_field_index].astype(int)

    tree = cKDTree(lab_xyz)
    _, idx = tree.query(df[["x", "y", "z"]].to_numpy(), k=1)
    ids = lab_id[idx]
    df["label_name"] = pd.Series(ids).map(class_names).values
    print(f"  matched {df['label_name'].notna().sum():,} labels via nearest neighbor")
    return df
