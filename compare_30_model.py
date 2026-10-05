"""
compare_30_model.py
===================
Trains and evaluates the final 30-feature model across three chronological
folds and reports R2 / MAE / RMSE / WAPE for train, validation and test.

Protocol, identical in every fold:
  * chronological ordering, never shuffled
  * 30-day purge gap between train/validation and validation/test
  * early stopping on the validation set only
  * the test segment is scored once, after fitting is complete

Fold C is the production fold used by final_pipeline_30.py. Folds A and B use
earlier cutoffs to show how performance varies by period - they are a stability
report, not a model-selection device. Hyperparameters and the feature list are
fixed across all folds.

Run:  python3 compare_30_model.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from feature_engineering_30 import SELECTED_FEATURES_30, TARGET
from final_pipeline_30 import (HERE, XGB_PARAMS, chronological_split,
                               per_company, predict, prepare, score, train_model)

FOLDS = [
    # name,      train_end,    val_start,    val_end,      test_start,   test_end
    ("fold_A", "2023-03-31", "2023-05-01", "2023-11-30", "2024-01-01", "2024-07-31"),
    ("fold_B", "2023-09-30", "2023-11-01", "2024-05-31", "2024-07-01", "2025-01-31"),
    ("fold_C", "2024-03-31", "2024-05-01", "2025-01-31", "2025-03-03", "2025-12-01"),
]
PRODUCTION_FOLD = "fold_C"


def main():
    df = prepare()
    print(f"rows with a complete target window: {len(df):,}   features: {len(SELECTED_FEATURES_30)}")

    results, per_co = {}, None
    for name, tre, vs, ve, ts, te_ in FOLDS:
        tr, va, te = chronological_split(df, tre, vs, ve, ts, te_)
        model = train_model(tr, va)
        fold = {"n": {"train": len(tr), "val": len(va), "test": len(te)},
                "best_iteration": int(model.best_iteration),
                "window": {"train_end": tre, "val": [vs, ve], "test": [ts, te_]}}
        for split_name, part in (("train", tr), ("val", va), ("test", te)):
            fold[split_name] = score(part[TARGET].values, predict(model, part))
        results[name] = fold
        if name == PRODUCTION_FOLD:
            per_co = per_company(te, predict(model, te))

    # ---------------- per-split table --------------------------------------
    print("\n" + "=" * 78)
    print("METRICS BY FOLD AND SPLIT")
    print("=" * 78)
    print(f"  {'fold':<8}{'split':>6}{'n':>7}{'R2':>9}{'MAE':>12}{'RMSE':>12}{'WAPE':>9}")
    for name in results:
        for sp in ("train", "val", "test"):
            s, n = results[name][sp], results[name]["n"][sp]
            print(f"  {name:<8}{sp:>6}{n:>7}{s['R2']:>9.4f}{s['MAE']:>12,.0f}"
                  f"{s['RMSE']:>12,.0f}{s['WAPE']:>9.4f}")

    # ---------------- stability across folds -------------------------------
    print("\n" + "=" * 78)
    print("STABILITY ACROSS CHRONOLOGICAL FOLDS (test segments)")
    print("=" * 78)
    agg = {}
    print(f"  {'metric':<8}{'fold_A':>11}{'fold_B':>11}{'fold_C':>11}{'mean':>11}{'std':>11}")
    for metric in ("R2", "MAE", "RMSE", "WAPE"):
        vals = [results[f]["test"][metric] for f in results]
        agg[metric] = {"folds": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals))}
        fmt = (lambda v: f"{v:>11.4f}") if metric in ("R2", "WAPE") else (lambda v: f"{v:>11,.0f}")
        print(f"  {metric:<8}" + "".join(fmt(v) for v in vals)
              + fmt(np.mean(vals)) + fmt(np.std(vals)))

    gaps = [results[f]["train"]["R2"] - results[f]["test"]["R2"] for f in results]
    print(f"\n  train-to-test R2 gap by fold: "
          f"{', '.join(f'{g:+.4f}' for g in gaps)}   mean {np.mean(gaps):+.4f}")

    print("\nPer-company TEST metrics (production fold)")
    print(per_co.round(3).to_string(float_format=lambda v: f"{v:,.3f}"))

    out = {"feature_count": len(SELECTED_FEATURES_30),
           "features": SELECTED_FEATURES_30,
           "hyperparameters": {k: v for k, v in XGB_PARAMS.items()},
           "purge_gap_days": 30,
           "folds": results,
           "test_across_folds": agg,
           "production_fold": PRODUCTION_FOLD,
           "final_test_metrics": results[PRODUCTION_FOLD]["test"],
           "final_test_per_company": per_co.round(4).to_dict(orient="index")}
    path = os.path.join(HERE, "final_model_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
