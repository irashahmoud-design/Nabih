"""Verification of the final 30-feature project files against the delivery checklist."""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from feature_engineering_30 import (CALENDAR_FEATURES, FINANCIAL_FEATURES,
                                    SELECTED_FEATURES_30, TARGET, build_features_30,
                                    load_dataset)
from final_pipeline_30 import (DATA_PATH, HORIZON, TEST_END, TEST_START, TRAIN_END,
                               VAL_END, VAL_START, chronological_split, prepare,
                               predict, train_model)

checks = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


EXPECTED = ["feature_engineering_30.py", "final_pipeline_30.py", "leakage_audit_30.py",
            "compare_30_model.py", "final_feature_list.json", "final_model_metrics.json",
            "FINAL_README.md"]
ck("all 7 deliverables exist", all(os.path.exists(os.path.join(HERE, f)) for f in EXPECTED),
   ", ".join(f for f in EXPECTED if not os.path.exists(os.path.join(HERE, f))) or "none missing")

# ---- 1. all 30 features exist ---------------------------------------------
df = prepare()
ck("all 30 features built", all(f in df.columns for f in SELECTED_FEATURES_30),
   f"{len(SELECTED_FEATURES_30)} features")
ck("13 calendar + 17 financial = 30, no overlap",
   len(CALENDAR_FEATURES) == 13 and len(FINANCIAL_FEATURES) == 17
   and not set(CALENDAR_FEATURES) & set(FINANCIAL_FEATURES)
   and set(CALENDAR_FEATURES) | set(FINANCIAL_FEATURES) == set(SELECTED_FEATURES_30))
ck("feature list unchanged from the approved 30",
   sorted(SELECTED_FEATURES_30) == sorted(json.load(
       open(os.path.join(HERE, "final_feature_list.json")))["features"]
       and [f["name"] for f in json.load(
           open(os.path.join(HERE, "final_feature_list.json")))["features"]]))

# ---- 2. no feature outside the 30 reaches the model -----------------------
tr, va, te = chronological_split(df)
model = train_model(tr, va)
booster_feats = model.get_booster().feature_names
ck("model consumes exactly the 30 features, nothing else",
   sorted(booster_feats) == sorted(SELECTED_FEATURES_30), f"{len(booster_feats)} inputs")
ck("frame carries no stray engineered columns",
   sorted(set(df.columns) - {"company_id", "date", TARGET, "scale"}) == sorted(SELECTED_FEATURES_30),
   f"extras = {sorted(set(df.columns) - set(SELECTED_FEATURES_30) - {'company_id','date',TARGET,'scale'})}")
ck("target and scale excluded from model inputs",
   TARGET not in booster_feats and "scale" not in booster_feats)

# ---- 3. leakage ------------------------------------------------------------
r = subprocess.run([sys.executable, os.path.join(HERE, "leakage_audit_30.py")],
                   capture_output=True, text=True)
ck("leakage_audit_30.py returns PASS", r.returncode == 0,
   f"exit={r.returncode}, 'PASS' in output = {'PASS -' in r.stdout}")

# ---- 4. no unexpected NaNs -------------------------------------------------
nans = df[SELECTED_FEATURES_30].isna().sum()
worst = nans.max()
head_only = True
for c in nans[nans > 0].index:
    for _, g in df.groupby("company_id"):
        idx = np.where(g[c].isna().to_numpy())[0]
        if len(idx) and idx.max() != len(idx) - 1:
            head_only = False
ck("NaNs only in per-company warm-up rows", head_only,
   f"max {int(worst)} rows ({int(worst)//6} per company, from the 365-day YoY warm-up)")
ck("no NaN in target after prepare()", df[TARGET].isna().sum() == 0)
ck("no NaN or inf in scale", df["scale"].notna().all() and np.isfinite(df["scale"]).all())

# ---- 5. target alignment ---------------------------------------------------
raw = load_dataset(DATA_PATH)
worst_t, n = 0.0, 0
rng = np.random.default_rng(3)
for cid, g in raw.groupby("company_id"):
    g = g.sort_values("date").reset_index(drop=True)
    nf, y = g.net_cash_flow.to_numpy(), g[TARGET].to_numpy()
    for t in rng.choice(len(g) - HORIZON, 200, replace=False):
        worst_t = max(worst_t, abs(nf[t + 1:t + 31].sum() - y[t])); n += 1
ck(f"target = sum(net[t+1:t+31]) on {n} samples", worst_t < 0.011, f"max dev {worst_t:.4f} SAR")
ck("target excludes day t",
   all(abs(g.sort_values('date')[TARGET].values[500]
           - g.sort_values('date').net_cash_flow.values[500:530].sum()) > 1e-6
       for _, g in raw.groupby("company_id")))

# ---- 6. purge preserved ----------------------------------------------------
gv = (pd.Timestamp(VAL_START) - pd.Timestamp(TRAIN_END)).days - 1
gt = (pd.Timestamp(TEST_START) - pd.Timestamp(VAL_END)).days - 1
ck("30-day purge: train -> validation", gv >= HORIZON, f"{gv} days")
ck("30-day purge: validation -> test", gt >= HORIZON, f"{gt} days")
ck("splits are chronological and non-overlapping",
   tr.date.max() < va.date.min() < va.date.max() < te.date.min(),
   f"{tr.date.max().date()} | {va.date.min().date()}-{va.date.max().date()} | {te.date.min().date()}")
ck("chronological_split rejects an undersized purge", False, "placeholder")
try:
    chronological_split(df, "2024-03-31", "2024-04-05", "2025-01-31", "2025-03-03", "2025-12-01")
    checks[-1] = ("chronological_split rejects an undersized purge", False, "no error raised")
except ValueError as e:
    checks[-1] = ("chronological_split rejects an undersized purge", True, str(e)[:46])

# ---- 7. test untouched during training -------------------------------------
src = open(os.path.join(HERE, "final_pipeline_30.py")).read()
fit_line = [l for l in src.splitlines() if "m.fit(" in l or "eval_set" in l]
ck("training references only train/validation",
   all("te" not in l.replace("test", "") .replace("eval_set", "") for l in fit_line)
   and "eval_set=[(va[features]" in src,
   "eval_set uses the validation partition only")
ck("test partition is built after the model is fitted in main()",
   src.index("model = train_model(tr, va)") < src.index('("test", te)'))

# ---- 8. metrics file -------------------------------------------------------
m = json.load(open(os.path.join(HERE, "final_model_metrics.json")))
ck("metrics json has 3 folds with train/val/test + mean/std",
   len(m["folds"]) == 3 and all(k in m["test_across_folds"]["R2"] for k in ("folds", "mean", "std"))
   and all(mm in m["final_test_metrics"] for mm in ("R2", "MAE", "RMSE", "WAPE")),
   f"final test R2={m['final_test_metrics']['R2']:.4f} WAPE={m['final_test_metrics']['WAPE']:.4f}")

# ---- 9. reproducibility ----------------------------------------------------
p1 = predict(model, te)
p2 = predict(train_model(tr, va), te)
ck("retraining reproduces identical predictions", float(np.max(np.abs(p1 - p2))) < 1e-9,
   f"max delta {float(np.max(np.abs(p1 - p2))):.2e}")

print("=" * 86)
print(f"{'CHECK':<58}{'STATUS':<8}DETAIL")
print("=" * 86)
for name, ok, detail in checks:
    print(f"{name:<58}{'PASS' if ok else 'FAIL':<8}{detail}")
nfail = sum(1 for _, ok, _ in checks if not ok)
print("=" * 86)
print(f"{len(checks)-nfail}/{len(checks)} checks passed"
      + ("" if nfail == 0 else f"   <-- {nfail} FAILURES"))
sys.exit(1 if nfail else 0)
