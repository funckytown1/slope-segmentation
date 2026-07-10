# Rock-Slope Point Cloud Segmentation

Segments colorized 3D rock-slope point clouds into geohazard classes (intact
rock, discontinuity/fracture, loose block/debris, vegetation, shadow) using
scale-normalized geometric + color features, automated rule-based weak labels,
and a Random Forest evaluated leave-one-slope-out.

## Data

UAV-photogrammetry rock slopes, colorized XYZRGB, **not included in this repo**
(download and place under `data/raw/zenodo_slopes/`; raw/processed data and all
point clouds are git-ignored).

> Point cloud datasets from *"Automatic Planar Discontinuity Identification with
> Comprehensive Error Quantification: Case Study of Multi-structured Slopes
> Using 3D Point Cloud Analytics."*
> https://zenodo.org/records/16934958 · doi:10.5281/zenodo.16934958 · CC-BY-4.0

`.txt` = whitespace `X Y Z R G B` (RGB already 0–1). `.pcd` via Open3D. Slope_3
ships as 2 `.pcd`, Slope_4 as 4 `.txt` (concatenated per slope). Edit
`configs/dataset.yaml` to match the files you downloaded.

## Setup

```bash
conda create -n slopeseg -c conda-forge python=3.10 \
    numpy pandas scikit-learn scipy pyarrow matplotlib pyyaml tqdm hdbscan
conda activate slopeseg
pip install open3d jakteristics laspy      # not on conda-forge for this platform
```

## Run

```bash
# 1. Ingest raw .txt/.pcd -> one canonical parquet per slope, + review previews
python scripts/ingest_slopes.py  --config configs/dataset.yaml
python scripts/review_slopes.py  --config configs/dataset.yaml

# 2. Rule-based draft segmentation per slope (run from src/).
#    --voxel downsamples big clouds; 0 = full resolution.
cd src
python -m segmentation.auto_label --input ../data/processed/zenodo/slope_6.parquet --voxel 0
python -m segmentation.auto_label --input ../data/processed/zenodo/slope_5.parquet --voxel 0
python -m segmentation.auto_label --input ../data/processed/zenodo/slope_4.parquet --voxel 0.01
python -m segmentation.auto_label --input ../data/processed/zenodo/slope_2.parquet --voxel 0.03
python -m segmentation.auto_label --input ../data/processed/zenodo/slope_3.parquet --voxel 0.03
cd ..

# 3. Random Forest, leave-one-slope-out cross-validation
python scripts/loso_experiment.py --rules-dir outputs/rules --train-cap 80000 --save-ply
```

### v2 (scale-invariant local-height feature)

Re-labels from the cached features (no feature recompute) into a separate dir
so it can be compared against the baseline:

```bash
python scripts/relabel_rules.py   --rules-dir outputs/rules --out-dir outputs/rules_v2 --local-height --no-ply
python scripts/loso_experiment.py --rules-dir outputs/rules_v2 --out-dir outputs/loso_v2 --train-cap 80000 --save-ply
```

## Outputs

| dir | contents |
|---|---|
| `outputs/review/` | per-slope 3-view RGB previews + `slopes_summary.md` |
| `outputs/rules/` | rule-based draft per slope: `*_rules.parquet` (features + `label_name`), `*_rules_views.png`, `*_rules.ply` |
| `outputs/loso/` | `loso_metrics.md`/`.json`, difficulty + per-class-F1 figures, per-fold predicted `*.png`/`*.ply` |
| `outputs/loso_v2/` | same, after the local-height feature |

## Useful flags

- `auto_label --voxel <m>` — downsample size (0 = full res).
- `loso_experiment --train-cap N` — points per training slope (balance/speed).
- `loso_experiment --save-ply` — write predicted PLYs; big held-out slopes are
  subsampled to `--ply-max-points` (default 2M). `--full-ply` forces native res.

Notes: `jakteristics` makes the geometric features fast (there's a slower PCA
fallback if it's absent). `stage2_supervised.py` is an older single-split path
that takes labels from a cluster→class YAML or a hand-painted cloud
(`labeling.py`); LOSO is the primary automated evaluation.
