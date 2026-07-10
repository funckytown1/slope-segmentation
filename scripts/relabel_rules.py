"""
Re-apply the labeling rules to existing rules parquets, fast.

The expensive step (per-point feature extraction) is already cached in
<rules-dir>/<slope>_rules.parquet, so after tuning the rule thresholds in
segmentation.auto_label we can re-derive labels + previews without recomputing
features. Optionally writes to a separate --out-dir so a new variant can be
compared against the baseline instead of overwriting it.

    # in place (overwrite baseline):
    python scripts/relabel_rules.py --rules-dir outputs/rules

    # non-destructive v2 with the scale-invariant local-height feature:
    python scripts/relabel_rules.py --rules-dir outputs/rules \
        --out-dir outputs/rules_v2 --local-height
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from segmentation.auto_label import (   # noqa: E402
    CLASSES, rule_label_points, render_class_views, save_class_ply,
    add_local_height,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-dir", default="outputs/rules")
    ap.add_argument("--out-dir", default=None,
                    help="write here instead of overwriting rules-dir (non-destructive).")
    ap.add_argument("--local-height", action="store_true",
                    help="add the scale-invariant local_height feature before labeling.")
    ap.add_argument("--no-ply", action="store_true",
                    help="skip rewriting the (large) class PLYs")
    args = ap.parse_args()

    rules_dir = REPO / args.rules_dir
    out_dir = REPO / args.out_dir if args.out_dir else rules_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for pq in sorted(rules_dir.glob("*_rules.parquet")):
        stem = pq.stem.replace("_rules", "")
        df = pd.read_parquet(pq)
        if args.local_height:
            add_local_height(df)
        labels = rule_label_points(df).astype(str)
        df["label_name"] = labels.values
        df.to_parquet(out_dir / f"{stem}_rules.parquet", index=False)

        dist = labels.value_counts()
        total = int(dist.sum())
        print(f"{stem} ({total:,} pts):")
        for c in CLASSES:
            nc = int(dist.get(c, 0))
            print(f"  {c:<28} {nc:>10,} ({100*nc/total:4.1f}%)")

        render_class_views(df, labels, f"{stem}: rule-based draft segmentation",
                           out_dir / f"{stem}_rules_views.png")
        if not args.no_ply:
            save_class_ply(df, labels, out_dir / f"{stem}_rules.ply")
        print()


if __name__ == "__main__":
    main()
