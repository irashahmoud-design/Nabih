"""
leakage_audit_30.py
===================
Independent, empirical leakage audit of the 30 production features.

The audit does not reason from feature names. It perturbs the data and checks
whether features move when they must not.

  T1  FUTURE-PERTURBATION   corrupt every financial column AFTER a cut date,
                            rebuild, compare rows dated <= cut.
                            Any feature that moves consumed future data.
  T2  TARGET-PERTURBATION   replace future_30d_net_cash_flow with noise,
                            rebuild, compare all rows.
                            Any feature that moves reads the target.
  T3  CORRUPT-ALL-FINANCIAL replace every financial column with noise.
                            Features that stay identical are provably
                            pure-calendar; used to confirm the fwd30_* block
                            carries no financial content.
  T4  CONTROL               inject a deliberately leaky shift(-5) feature and
                            confirm T1 detects it. Without this, a passing
                            audit proves nothing.

Additional checks: window-span recomputation of the YoY family from the raw
series, and calendar determinism (identical across companies on a given date).

Run:  python3 leakage_audit_30.py      # exit code 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from feature_engineering_30 import (CALENDAR_FEATURES, FINANCIAL_FEATURES,
                                    SELECTED_FEATURES_30, TARGET,
                                    build_features_30, load_dataset)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "sme_cashflow_dataset.csv")
FINANCIAL_COLUMNS = ["cash_inflow", "cash_outflow", "opening_balance",
                     "closing_balance", "net_cash_flow", "salary_effect", TARGET]
TOL = 1e-9
rng = np.random.default_rng(20260911)


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, mask, cols) -> dict:
    """Max |a-b| per column over masked rows. NaN opposite a number counts as inf."""
    out = {}
    for c in cols:
        x = a.loc[mask, c].astype(float).to_numpy()
        y = b.loc[mask, c].astype(float).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.where(both_nan, 0.0, np.abs(x - y))
        out[c] = float(np.nanmax(np.where(np.isnan(d), np.inf, d)))
    return out


def main() -> int:
    raw = load_dataset(DATA_PATH)
    base = build_features_30(raw)
    all_rows = pd.Series(True, index=base.index)
    checks = []

    print("=" * 74)
    print("LEAKAGE AUDIT - 30 production features")
    print("=" * 74)

    # ---------------- T1 future-perturbation -------------------------------
    cut = pd.Timestamp("2024-06-30")
    r1 = raw.copy()
    fut = r1.date > cut
    for c in FINANCIAL_COLUMNS:
        r1.loc[fut, c] = (r1.loc[fut, c].to_numpy()
                          * rng.uniform(0.1, 8.0, int(fut.sum())) - 555_555.0)
    p1 = build_features_30(r1)
    past = base.date <= cut
    d1 = {k: v for k, v in max_abs_diff(base, p1, past, SELECTED_FEATURES_30).items()
          if v > TOL}
    checks.append(("T1 future-perturbation", not d1,
                   f"{int(fut.sum())} future rows corrupted, {int(past.sum())} past rows "
                   f"compared, {len(d1)} features changed"))
    if d1:
        print("  LEAKING:", d1)

    # ---------------- T4 control (validates T1 can fail) -------------------
    ctrl_base = base.copy()
    ctrl_p1 = p1.copy()
    for frame in (ctrl_base, ctrl_p1):
        frame["DELIBERATE_LEAK"] = frame.groupby("company_id")["closing_balance"].shift(-5)
    d4 = max_abs_diff(ctrl_base, ctrl_p1, past, ["DELIBERATE_LEAK"])
    checks.append(("T4 control (leaky shift(-5) must be caught)",
                   d4["DELIBERATE_LEAK"] > TOL,
                   f"control moved by {d4['DELIBERATE_LEAK']:,.2f} -> detector is live"))

    # ---------------- T2 target-perturbation -------------------------------
    r2 = raw.copy()
    r2[TARGET] = rng.normal(0, 5e6, len(r2))
    p2 = build_features_30(r2)
    d2 = {k: v for k, v in max_abs_diff(base, p2, all_rows, SELECTED_FEATURES_30).items()
          if v > TOL}
    checks.append(("T2 target-perturbation", not d2,
                   f"target replaced by noise, {len(d2)} of 30 features changed"))
    if d2:
        print("  LEAKING:", d2)

    # ---------------- T3 corrupt all financials ----------------------------
    r3 = raw.copy()
    for c in FINANCIAL_COLUMNS:
        r3[c] = rng.normal(0, 1e6, len(r3))
    p3 = build_features_30(r3)
    d3 = max_abs_diff(base, p3, all_rows, SELECTED_FEATURES_30)
    frozen = sorted(k for k, v in d3.items() if v <= 1e-12)
    moved = sorted(k for k, v in d3.items() if v > 1e-12)
    checks.append(("T3 calendar block is financially inert",
                   frozen == sorted(CALENDAR_FEATURES) and moved == sorted(FINANCIAL_FEATURES),
                   f"{len(frozen)} pure-calendar / {len(moved)} financial, as declared"))

    # ---------------- window-span recomputation ----------------------------
    ok_span = True
    for cid in raw.company_id.unique():
        g = raw[raw.company_id == cid].sort_values("date").reset_index(drop=True)
        nf, inf = g.net_cash_flow.to_numpy(), g.cash_inflow.to_numpy()
        rows = base[base.company_id == cid].reset_index(drop=True)
        for t in (800, 1100, 1400):
            man_yoy = nf[t - 364:t - 334].sum()                    # spans [t-364, t-335]
            man_in = inf[t - 394:t - 364].mean()                   # spans [t-394, t-365]
            man_growth = inf[t - 29:t + 1].mean() / (man_in + 1e-6)
            ok_span &= abs(rows.yoy_target[t] - man_yoy) < 1e-6
            ok_span &= abs(rows.yoy_in_mean[t] - man_in) < 1e-6
            ok_span &= abs(rows.yoy_target_scaled[t] - man_yoy * man_growth) < 1e-6
    checks.append(("YoY window spans recomputed from raw series", ok_span,
                   "yoy_target spans [t-364, t-335]; newest index t-335, never [t+1, t+30]"))

    # ---------------- calendar determinism ---------------------------------
    det = max(base.groupby("date")[c].nunique().max() for c in CALENDAR_FEATURES)
    checks.append(("calendar features identical across companies per date", det == 1,
                   f"max distinct values per date = {det}"))

    # ---------------- no unexpected NaNs -----------------------------------
    model_rows = base.dropna(subset=[TARGET])
    nan_counts = model_rows[SELECTED_FEATURES_30].isna().sum()
    warmup_only = True
    for c, n in nan_counts.items():
        if n == 0:
            continue
        # NaNs must sit only at the head of each company series (warm-up)
        for cid, g in model_rows.groupby("company_id"):
            idx = np.where(g[c].isna().to_numpy())[0]
            if len(idx) and idx.max() != len(idx) - 1:
                warmup_only = False
    checks.append(("NaNs confined to per-company warm-up", warmup_only,
                   f"max NaN count in a feature = {int(nan_counts.max())} "
                   f"({int(nan_counts.max()) / model_rows.company_id.nunique():.0f} rows/company)"))

    # ---------------- target alignment -------------------------------------
    worst = 0.0
    for cid, g in raw.groupby("company_id"):
        g = g.sort_values("date").reset_index(drop=True)
        nf, y = g.net_cash_flow.to_numpy(), g[TARGET].to_numpy()
        for t in (0, 137, 640, 1295, 1795):
            worst = max(worst, abs(nf[t + 1:t + 31].sum() - y[t]))
    checks.append(("target = sum(net_cash_flow[t+1 : t+31])", worst < 0.011,
                   f"max deviation {worst:.4f} SAR"))

    # ---------------- report ------------------------------------------------
    print()
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {detail}")

    passed = all(ok for _, ok, _ in checks)
    print("\n" + "=" * 74)
    print("FINAL RESULT:", "PASS - no future-data leakage detected in any of the 30 features"
          if passed else "FAIL - leakage detected, see above")
    print("=" * 74)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
