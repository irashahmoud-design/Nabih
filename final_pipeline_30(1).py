"""
final_pipeline_30.py
====================
End-to-end pipeline for the SME 30-day cash-flow forecasting model.

    load -> feature engineering (30 features) -> chronological split with purge
         -> train (XGBoost) -> predict -> evaluate -> persist artefacts

Target
------
    future_30d_net_cash_flow[t] = sum(net_cash_flow[t+1 .. t+30])

Temporal protocol
-----------------
Strictly chronological, never shuffled. A 30-day purge gap separates each
segment so that a training row's forward window cannot overlap the next
segment. The test set is evaluated once, after training and early stopping
have completed against the validation set only.

Run:  python3 final_pipeline_30.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from feature_engineering_30 import (HORIZON, SELECTED_FEATURES_30, TARGET,
                                    build_features_30, load_dataset)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "sme_cashflow_dataset.csv")

# --- production split (fold C). Purge = HORIZON days between segments -------
TRAIN_END = "2024-03-31"
VAL_START, VAL_END = "2024-05-01", "2025-01-31"
TEST_START, TEST_END = "2025-03-03", "2025-12-01"

# --- model. Frozen on validation only; never tuned against the test set ----
XGB_PARAMS = dict(
    n_estimators=4000, learning_rate=0.02, max_depth=6,
    subsample=0.85, colsample_bytree=0.7, min_child_weight=15,
    reg_lambda=2.0, reg_alpha=0.5, gamma=0.05,
    tree_method="hist", early_stopping_rounds=150,
    eval_metric="rmse", random_state=42, n_jobs=4,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))


def score(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": float(r2_score(y, p)),
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "WAPE": wape(y, p)}


# ---------------------------------------------------------------------------
# Preprocessing / splitting
# ---------------------------------------------------------------------------
def prepare(path: str = DATA_PATH) -> pd.DataFrame:
    """Load, engineer the 30 features, drop rows with no complete target window."""
    df = build_features_30(load_dataset(path))
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def chronological_split(df, train_end=TRAIN_END, val_start=VAL_START,
                        val_end=VAL_END, test_start=TEST_START, test_end=TEST_END):
    """Chronological split; the caller's dates must already embed the purge gap."""
    ts = pd.Timestamp
    gap_v = (ts(val_start) - ts(train_end)).days - 1
    gap_t = (ts(test_start) - ts(val_end)).days - 1
    if gap_v < HORIZON or gap_t < HORIZON:
        raise ValueError(f"purge gap too small: {gap_v}d / {gap_t}d, need >= {HORIZON}d")
    tr = df[df.date <= ts(train_end)]
    va = df[(df.date >= ts(val_start)) & (df.date <= ts(val_end))]
    te = df[(df.date >= ts(test_start)) & (df.date <= ts(test_end))]
    return tr, va, te


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def train_model(tr, va, features=SELECTED_FEATURES_30, params=XGB_PARAMS):
    """
    Fit on the scale-normalised target.

    The model learns target / (30 x trailing 30-day mean inflow), because
    gradient-boosted trees cannot extrapolate a growing absolute level beyond
    the training range. The scale is trailing-only, so this introduces no
    forward information. Early stopping uses the validation set only.
    """
    m = XGBRegressor(**params)
    m.fit(tr[features], tr[TARGET].values / tr["scale"].values,
          eval_set=[(va[features], va[TARGET].values / va["scale"].values)],
          verbose=False)
    return m


def predict(model, part, features=SELECTED_FEATURES_30) -> np.ndarray:
    """Predict the ratio, then convert straight back to SAR."""
    return model.predict(part[features]) * part["scale"].values


def per_company(part, pred) -> pd.DataFrame:
    rows = []
    for cid in sorted(part.company_id.unique()):
        m = (part.company_id == cid).values
        rows.append({"company": cid, **score(part[TARGET].values[m], pred[m])})
    return pd.DataFrame(rows).set_index("company")


# ---------------------------------------------------------------------------
# Downstream risk layer (derived AFTER prediction; not part of training)
# ---------------------------------------------------------------------------
def risk_layer(part, pred, threshold_days: float = 30.0) -> pd.DataFrame:
    """Turn a cash-flow forecast into a liquidity verdict."""
    daily_out = part["closing_balance"].values / part["days_of_coverage"].values
    projected = part["closing_balance"].values + pred
    cov_pred = projected / daily_out
    return pd.DataFrame({
        "company_id": part.company_id.values,
        "date": part.date.values,
        "day_of_coverage": part["days_of_coverage"].values.round(1),
        "predicted_30d_net_cash_flow": pred.round(0),
        "projected_coverage_30d": cov_pred.round(1),
        "Liquidity_Risk_Score": np.clip(100.0 * (1 - cov_pred / 60.0), 0, 100).round(0),
        "Liquidity_Risk_30D": np.where(cov_pred < threshold_days, "At Risk", "Safe"),
    })


# ---------------------------------------------------------------------------
def main():
    df = prepare()
    tr, va, te = chronological_split(df)

    print("SME 30-day cash-flow forecasting - final pipeline (30 features)")
    print(f"  features : {len(SELECTED_FEATURES_30)}")
    print(f"  train    : {tr.date.min().date()} .. {tr.date.max().date()}  n={len(tr):,}")
    print(f"  purge    : {HORIZON} days")
    print(f"  val      : {va.date.min().date()} .. {va.date.max().date()}  n={len(va):,}")
    print(f"  purge    : {HORIZON} days")
    print(f"  test     : {te.date.min().date()} .. {te.date.max().date()}  n={len(te):,}")

    model = train_model(tr, va)
    print(f"  best_iteration = {model.best_iteration}")

    results = {}
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        p = predict(model, part)
        results[name] = score(part[TARGET].values, p)
        s = results[name]
        print(f"\n{name.upper():<11} R2={s['R2']:.4f}  MAE={s['MAE']:,.0f}  "
              f"RMSE={s['RMSE']:,.0f}  WAPE={s['WAPE']:.4f}")

    pred_te = predict(model, te)
    pc = per_company(te, pred_te)
    print("\nPer-company TEST metrics")
    print(pc.round(3).to_string(float_format=lambda v: f"{v:,.3f}"))
    results["test_per_company"] = pc.round(4).to_dict(orient="index")

    pd.DataFrame({"company_id": te.company_id.values, "date": te.date.values,
                  "y_true": te[TARGET].values, "y_pred": pred_te}
                 ).to_csv(os.path.join(HERE, "predictions_test_30.csv"), index=False)
    rl = risk_layer(te, pred_te)
    rl.to_csv(os.path.join(HERE, "risk_layer_test_30.csv"), index=False)
    print(f"\nRisk layer: {(rl.Liquidity_Risk_30D == 'At Risk').mean():.1%} of test days "
          f"flagged At Risk")

    model.save_model(os.path.join(HERE, "model_30.json"))
    with open(os.path.join(HERE, "final_model_metrics_production_fold.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: model_30.json, predictions_test_30.csv, risk_layer_test_30.csv")


if __name__ == "__main__":
    main()
