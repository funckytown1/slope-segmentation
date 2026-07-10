# Leave-one-slope-out (LOSO) cross-validation

Each slope held out once; RF trained on the others' rule labels (capped per slope for balance).

| held-out slope | macro-F1 (all) | macro-F1 (geometry-only) | train pts |
|---|---|---|---|
| slope_2 | 0.747 | 0.729 | 320,000 |
| slope_3 | 0.659 | 0.530 | 320,000 |
| slope_4 | 0.505 | 0.446 | 320,000 |
| slope_5 | 0.697 | 0.613 | 320,000 |
| slope_6 | 0.594 | 0.476 | 320,000 |

**Hardest held-out (all classes): `slope_4` (0.505)** — most out-of-distribution.
**Easiest: `slope_2` (0.747)**.

Per-class F1 by held-out slope: see `loso_perclass_f1.png` (blank = class absent in that slope's truth). Vegetation/shadow F1 depends on whether the class was present in the *training* slopes, which is a data-coverage effect, not a geometry-transfer one — hence the geometry-only column.
