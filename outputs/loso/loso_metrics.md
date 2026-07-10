# Leave-one-slope-out (LOSO) cross-validation

Each slope held out once; RF trained on the others' rule labels (capped per slope for balance).

| held-out slope | macro-F1 (all) | macro-F1 (geometry-only) | train pts |
|---|---|---|---|
| slope_2 | 0.680 | 0.639 | 320,000 |
| slope_3 | 0.666 | 0.541 | 320,000 |
| slope_4 | 0.462 | 0.367 | 320,000 |
| slope_5 | 0.608 | 0.480 | 320,000 |
| slope_6 | 0.588 | 0.447 | 320,000 |

**Hardest held-out (all classes): `slope_4` (0.462)** — most out-of-distribution.
**Easiest: `slope_2` (0.680)**.

Per-class F1 by held-out slope: see `loso_perclass_f1.png` (blank = class absent in that slope's truth). Vegetation/shadow F1 depends on whether the class was present in the *training* slopes, which is a data-coverage effect, not a geometry-transfer one — hence the geometry-only column.
