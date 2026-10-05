"""
feature_engineering_30.py
=========================
Clean implementation of the 30 production features for the SME 30-day
cash-flow forecasting model.

Design contract
---------------
Every feature at row t is computed from one of exactly two sources:

  (A) DETERMINISTIC CALENDAR  - 13 features
      Pure functions of the date. Contain no financial information at all.
      Some describe the t+1..t+30 window; that is wall-calendar arithmetic
      (paydays, tax dates, weekends, Ramadan), which is genuinely known in
      advance and carries zero realised financial content.

  (B) HISTORICAL FINANCIAL    - 17 features
      Built only from financial values dated <= t. Every rolling window ends
      at t, every shift is positive (backward). The newest index any of these
      touches is t.

Nothing here reads `future_30d_net_cash_flow`, and no window overlaps the
target span [t+1, t+30].

`build_features_30()` also returns a `scale` column. This is NOT a model
feature - it is the target normaliser (30 x trailing 30-day mean inflow) used
so the tree model learns a ratio instead of an absolute level it cannot
extrapolate. It is trailing-only and is never fed to the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HORIZON = 30                     # forecast horizon in days
TARGET = "future_30d_net_cash_flow"
EPS = 1e-6

# Business calendar of the SME environment. In production these come from the
# client's own payroll / filing configuration, not from constants.
PAYDAY_DOM = 27                  # salaries credited on the 27th
TAX_DOM = 15                     # quarterly VAT / zakat settlement day
TAX_MONTHS = (1, 4, 7, 10)
WEEKEND_DOW = (4, 5)             # Saudi weekend: Friday, Saturday

# Hijri anchors (Gregorian). Known months ahead to within +/- 1 day.
RAMADAN_START = {
    2021: "2021-04-13", 2022: "2022-04-02", 2023: "2023-03-23",
    2024: "2024-03-11", 2025: "2025-03-01", 2026: "2026-02-18",
}
EID_FITR = {
    2021: "2021-05-13", 2022: "2022-05-02", 2023: "2023-04-21",
    2024: "2024-04-10", 2025: "2025-03-30", 2026: "2026-03-20",
}

# ---------------------------------------------------------------------------
# The 30 features
# ---------------------------------------------------------------------------
CALENDAR_FEATURES = [
    "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
    "days_to_month_end", "days_from_eid_fitr",
    "fwd30_is_payday", "fwd30_is_tax", "fwd30_is_wknd", "fwd30_in_ramadan",
    "fwd30_doy_s1", "fwd30_doy_c1", "fwd30_doy_c2",
]

FINANCIAL_FEATURES = [
    "opening_balance", "closing_balance", "net_lag30",
    "net_mean90", "net_mean180", "net_std30", "net_std90",
    "momentum_in", "momentum_out", "burn_ratio90",
    "days_of_coverage", "balance_to_in30", "in_seasonal_dev",
    "yoy_in_mean", "yoy_growth", "yoy_target", "yoy_target_scaled",
]

SELECTED_FEATURES_30 = [
    "in_seasonal_dev", "doy_cos2", "doy_sin1", "fwd30_is_payday",
    "yoy_growth", "burn_ratio90", "days_of_coverage", "fwd30_is_tax",
    "balance_to_in30", "yoy_target", "net_mean180", "fwd30_in_ramadan",
    "opening_balance", "doy_sin2", "fwd30_doy_c2", "yoy_target_scaled",
    "closing_balance", "net_std90", "net_mean90", "fwd30_doy_s1",
    "fwd30_is_wknd", "net_std30", "momentum_in", "doy_cos1",
    "yoy_in_mean", "momentum_out", "net_lag30", "days_from_eid_fitr",
    "fwd30_doy_c1", "days_to_month_end",
]

assert len(SELECTED_FEATURES_30) == 30
assert set(SELECTED_FEATURES_30) == set(CALENDAR_FEATURES) | set(FINANCIAL_FEATURES)
assert len(CALENDAR_FEATURES) == 13 and len(FINANCIAL_FEATURES) == 17


# ---------------------------------------------------------------------------
# (A) Deterministic calendar
# ---------------------------------------------------------------------------
def _nearest_signed_offset(dates: pd.Series, anchors: dict, clip: int = 120) -> np.ndarray:
    """Signed day offset to the nearest anchor date (negative = anchor ahead)."""
    o = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    best = np.full(len(o), 9999.0)
    for v in anchors.values():
        off = o - pd.Timestamp(v).toordinal()
        best = np.where(np.abs(off) < np.abs(best), off, best)
    return np.clip(best, -clip, clip)


def _forward_window_calendar() -> pd.DataFrame:
    """
    Mean of each calendar indicator over the window [t+1, t+30].

    Built on a standalone date index that never touches the financial data.
    The reversed rolling window at reversed position i covers original
    [t, t+29]; the trailing shift(-1) moves it to [t+1, t+30].
    """
    cal = pd.DataFrame({"date": pd.date_range("2020-01-01", "2026-12-31", freq="D")})
    day, month, dow = cal.date.dt.day, cal.date.dt.month, cal.date.dt.dayofweek
    doy = cal.date.dt.dayofyear

    cal["is_payday"] = (day == PAYDAY_DOM).astype(float)
    cal["is_tax"] = ((day == TAX_DOM) & month.isin(TAX_MONTHS)).astype(float)
    cal["is_wknd"] = dow.isin(WEEKEND_DOW).astype(float)
    cal["doy_s1"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_c1"] = np.cos(2 * np.pi * doy / 365.25)
    cal["doy_c2"] = np.cos(4 * np.pi * doy / 365.25)

    ram_off = _nearest_signed_offset(cal.date, RAMADAN_START, clip=9999)
    cal["in_ramadan"] = ((ram_off >= 0) & (ram_off <= 29)).astype(float)

    cols = ["is_payday", "is_tax", "is_wknd", "in_ramadan", "doy_s1", "doy_c1", "doy_c2"]
    cal = cal.set_index("date")[cols]
    fwd = (cal.iloc[::-1]
              .rolling(HORIZON, min_periods=HORIZON).mean()
              .iloc[::-1]
              .shift(-1))
    fwd.columns = [f"fwd30_{c}" for c in cols]
    return fwd


def add_calendar_features(out: pd.DataFrame, d: pd.Series) -> pd.DataFrame:
    """13 deterministic calendar features. No financial input."""
    doy = d.dt.dayofyear
    out["doy_sin1"] = np.sin(2 * np.pi * 1 * doy / 365.25)
    out["doy_cos1"] = np.cos(2 * np.pi * 1 * doy / 365.25)
    out["doy_sin2"] = np.sin(2 * np.pi * 2 * doy / 365.25)
    out["doy_cos2"] = np.cos(2 * np.pi * 2 * doy / 365.25)
    out["days_to_month_end"] = d.dt.days_in_month - d.dt.day
    out["days_from_eid_fitr"] = _nearest_signed_offset(d, EID_FITR)
    return out.merge(_forward_window_calendar(), left_on="date",
                     right_index=True, how="left")


# ---------------------------------------------------------------------------
# (B) Historical financial  (all windows end at t; all shifts are backward)
# ---------------------------------------------------------------------------
def add_financial_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    17 features from financial history only.

    Intermediates (in_mean30/90/180, out_mean30/90, net_sum30_past) are computed
    here but dropped before the model sees the frame - they exist only to derive
    the ratio features and the target scale.
    """
    g = out.groupby("company_id", group_keys=False)

    def roll(col, w, stat):
        mp = max(3, w // 3)
        r = g[col].rolling(w, min_periods=mp)
        return (r.mean() if stat == "mean" else r.std()).values

    # --- trailing intermediates: window [t-w+1, t], newest index = t ---------
    out["_in_mean30"] = roll("cash_inflow", 30, "mean")
    out["_in_mean90"] = roll("cash_inflow", 90, "mean")
    out["_in_mean180"] = roll("cash_inflow", 180, "mean")
    out["_out_mean30"] = roll("cash_outflow", 30, "mean")
    out["_out_mean90"] = roll("cash_outflow", 90, "mean")
    out["_net_sum30_past"] = g["net_cash_flow"].rolling(
        HORIZON, min_periods=HORIZON).sum().values          # spans [t-29, t]

    # --- level and volatility of realised net flow --------------------------
    out["net_lag30"] = g["net_cash_flow"].shift(30)          # single value at t-30
    out["net_mean90"] = roll("net_cash_flow", 90, "mean")    # [t-89,  t]
    out["net_mean180"] = roll("net_cash_flow", 180, "mean")  # [t-179, t]
    out["net_std30"] = roll("net_cash_flow", 30, "std")      # [t-29,  t]
    out["net_std90"] = roll("net_cash_flow", 90, "std")      # [t-89,  t]

    # --- momentum: recent activity against its own longer baseline ----------
    out["momentum_in"] = out["_in_mean30"] / (out["_in_mean90"] + EPS)
    out["momentum_out"] = out["_out_mean30"] / (out["_out_mean90"] + EPS)
    out["in_seasonal_dev"] = out["_in_mean30"] / (out["_in_mean180"] + EPS)

    # --- liquidity state at end of day t ------------------------------------
    out["burn_ratio90"] = out["_out_mean90"] / (out["_in_mean90"] + EPS)
    out["days_of_coverage"] = out["closing_balance"] / (out["_out_mean30"] + EPS)
    out["balance_to_in30"] = out["closing_balance"] / (out["_in_mean30"] + EPS)

    # --- same period one year ago (realised PAST windows) -------------------
    gg = out.groupby("company_id", group_keys=False)
    # _net_sum30_past[t-335] spans [t-364, t-335]: newest index is t-335,
    # i.e. 335 days BEFORE t. It never reaches into [t+1, t+30].
    out["yoy_target"] = gg["_net_sum30_past"].shift(335)
    out["yoy_in_mean"] = gg["_in_mean30"].shift(365)         # spans [t-394, t-365]
    out["yoy_growth"] = out["_in_mean30"] / (out["yoy_in_mean"] + EPS)
    out["yoy_target_scaled"] = out["yoy_target"] * out["yoy_growth"]

    # --- target normaliser (NOT a model feature) ----------------------------
    exp_in = gg["cash_inflow"].expanding().mean().reset_index(level=0, drop=True)
    out["scale"] = HORIZON * out["_in_mean30"].fillna(exp_in)

    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_features_30(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the frame with the 30 features plus company_id, date, target, scale.

    Rows are sorted by (company_id, date) so that every groupby-rolling result
    aligns positionally with the frame.
    """
    df = df.sort_values(["company_id", "date"]).reset_index(drop=True)
    out = df.copy()
    out = add_calendar_features(out, out["date"])
    out = add_financial_features(out)

    missing = [f for f in SELECTED_FEATURES_30 if f not in out.columns]
    if missing:
        raise RuntimeError(f"features failed to build: {missing}")

    keep = ["company_id", "date", TARGET, "scale"] + SELECTED_FEATURES_30
    return out[keep]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
