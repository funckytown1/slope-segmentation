"""
Unify the Zenodo slopes into canonical parquet, one per slope.

Handles the two format wrinkles in one place so nothing downstream has to
care about .txt vs .pcd or multi-part slopes:
  - .txt  -> whitespace X Y Z R G B, RGB already 0-1
  - .pcd  -> Open3D, colors 0-1 (verified present)
  - multi-part slopes -> all parts concatenated into one cloud

After this runs, every later stage reads data/processed/zenodo/<slope>.parquet
and never touches raw formats again.

Usage:
    python scripts/ingest_slopes.py --config configs/dataset.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pcio.loaders import load_any          
from pcio.canonical import save_canonical, summarize  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--dataset", default="zenodo_rock_slopes")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["datasets"][args.dataset]
    out_dir = REPO / d["output_parquet_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    slopes = d["slopes"]
    for slope_name, parts in slopes.items():
        print(f"\n=== {slope_name} ({len(parts)} part(s)) ===")
        frames = []
        for part in parts:
            p = REPO / part
            if not p.exists():
                print(f"  MISSING: {p} -- skipping (edit config to match your files)")
                continue
            df = load_any(p)
            print(f"  loaded {p.name}: {len(df):,} points")
            frames.append(df)
        if not frames:
            print(f"  no parts found for {slope_name}; skipping.")
            continue
        merged = pd.concat(frames, ignore_index=True)
        summarize(merged, name=slope_name)
        out_path = out_dir / f"{slope_name}.parquet"
        save_canonical(merged, out_path)

    print("\nDone. Next:")
    print("  python scripts/verify_colorized_cloud.py --input data/processed/zenodo/slope_a.parquet")
    print("  (then Stage 1 unsupervised on each slope parquet)")


if __name__ == "__main__":
    main()
