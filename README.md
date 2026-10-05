# SME Cash-Flow Forecasting & Early Liquidity-Risk Detection

Final project version — 30-feature model.

---

## 1. Objective

A forecasting and decision-support system for small and medium-sized enterprises that answers, for any business day:

- How much net cash flow is expected over the next 30 days?
- Will there be enough liquidity to cover upcoming expenses?
- Is a cash-flow gap likely?
- Is the business currently **Safe** or **At Risk**?
- Which patterns in historical cash flow are driving that risk?

The forecast feeds a downstream risk layer that converts a predicted cash-flow figure into a liquidity verdict an owner can act on.

## 2. Target definition

```
future_30d_net_cash_flow[t] = sum(net_cash_flow[t+1] … net_cash_flow[t+30])
```

Strictly forward-looking and starting at **t+1** — the current day is excluded. The final 30 days of each company carry no complete forward window and are left as `NaN` rather than imputed; those rows are dropped before modelling.

## 3. Forecast horizon

**30 days**, predicted as a single aggregate figure (not a day-by-day path).

## 4. Dataset

6 SMEs × 1,826 daily observations, 2021-01-01 → 2025-12-31 (10,956 rows), amounts in SAR. Three business behaviour patterns — stable-growing, seasonal, volatile — with two companies each at different scales.

Accounting identities hold exactly: `net = inflow − outflow`, `closing = opening + net`, `opening[t] = closing[t−1]`.

## 5. The 30 final features

**13 deterministic calendar features** — pure functions of the date, containing no financial information whatsoever:

| Feature | Meaning |
|---|---|
| `doy_sin1`, `doy_cos1`, `doy_sin2`, `doy_cos2` | First and second annual harmonics of day-of-year |
| `days_to_month_end` | Governs how many monthly events fall inside the forecast window |
| `days_from_eid_fitr` | Signed days to the nearest Eid al-Fitr, clipped ±120 |
| `fwd30_is_payday` | Share of the next 30 days falling on the salary date |
| `fwd30_is_tax` | Share falling on a quarterly VAT/zakat settlement date |
| `fwd30_is_wknd` | Share that are Friday/Saturday |
| `fwd30_in_ramadan` | Share falling inside Ramadan |
| `fwd30_doy_s1`, `fwd30_doy_c1`, `fwd30_doy_c2` | Mean seasonal harmonics across the forecast window |

**17 historical financial features** — every window ends at t, every shift is backward:

| Feature | Source window |
|---|---|
| `opening_balance`, `closing_balance` | t |
| `net_lag30` | t−30 |
| `net_mean90`, `net_std90` | [t−89, t] |
| `net_mean180` | [t−179, t] |
| `net_std30`, `days_of_coverage`, `balance_to_in30` | [t−29, t] |
| `momentum_in`, `momentum_out`, `burn_ratio90` | [t−89, t] |
| `in_seasonal_dev` | [t−179, t] |
| `yoy_in_mean` | [t−394, t−365] |
| `yoy_target` | [t−364, t−335] |
| `yoy_growth`, `yoy_target_scaled` | [t−394, t−335] and [t−29, t] |

Full descriptions in `final_feature_list.json`.

**Note on the `fwd30_*` block.** These describe the calendar of the next 30 days — how many paydays, weekends and tax dates it contains, and where it sits in the seasonal year. That is wall-calendar arithmetic available on day t, and it is built on a standalone date index that never joins the financial data. Test T3 below proves it empirically. In production these schedules come from the client's own payroll and filing configuration.

**Target normaliser.** The model learns `target / (30 × trailing 30-day mean inflow)` and converts back to SAR immediately. Gradient-boosted trees cannot extrapolate a growing absolute level beyond the training range; the ratio formulation removes that failure mode. The scale is trailing-only and is **not** a model feature.

## 6. Model

**XGBoost regressor**, single pooled model across all six companies.

```
n_estimators=4000 (early stopping at 150)   learning_rate=0.02
max_depth=6          subsample=0.85         colsample_bytree=0.7
min_child_weight=15  reg_lambda=2.0         reg_alpha=0.5
gamma=0.05           tree_method=hist       random_state=42
```

Hyperparameters were frozen using train/validation only and are held fixed across every fold.

## 7. Temporal evaluation methodology

Strictly chronological, never shuffled. A **30-day purge gap** separates each segment, so a training row's forward window cannot overlap the next segment — without it, the last 30 training rows would share realised days with the start of validation.

| Segment | Dates | Rows |
|---|---|---|
| Train | 2021-01-01 → 2024-03-31 | 7,116 |
| *purge* | 30 days | — |
| Validation | 2024-05-01 → 2025-01-31 | 1,656 |
| *purge* | 30 days | — |
| Test | 2025-03-03 → 2025-12-01 | 1,644 |

Early stopping uses the validation set only. The test segment is scored once, after fitting completes, and was never used for feature selection, hyperparameter tuning, model selection or threshold setting.

Two earlier folds (A: cutoff 2023-03-31, B: cutoff 2023-09-30) are evaluated under the identical protocol as a stability report.

## 8. Leakage-audit result

**PASS — no future-data leakage detected.** Run `python3 leakage_audit_30.py` (exit code 0 = pass).

The audit is empirical, not name-based:

| Test | Method | Result |
|---|---|---|
| T1 future-perturbation | Corrupt every financial column after 2024-06-30, rebuild, compare the 7,662 rows at or before the cut | 0 of 30 features changed |
| T2 target-perturbation | Replace the target with noise, rebuild, compare all rows | 0 of 30 features changed |
| T3 corrupt-all-financials | Replace all financial columns with noise | 13 calendar features frozen, 17 financial moved — exactly as declared |
| T4 control | Inject a deliberate `shift(-5)` leak and confirm T1 catches it | Detected (moved by 11,339,171) — the detector is live |

Plus: YoY window spans recomputed from the raw NumPy series (`yoy_target` spans [t−364, t−335]; its newest index is 335 days before t), calendar determinism (identical across all companies on every date), NaNs confined to per-company warm-up, and target alignment verified to 0.0000 SAR.

T4 matters most. An audit that passes everything is worthless unless you prove it can fail.

**One disclosure.** `fwd30_is_payday` and `fwd30_is_tax` encode a known business schedule (27th; 15 Jan/Apr/Jul/Oct). This is not future financial information, but it does assume the schedule is known — true for a bookkeeping product with access to the client's payroll calendar, and worth stating rather than leaving for a reviewer to find. Ramadan and Eid dates depend on moon sighting and are known in advance to within ±1 day; since `fwd30_in_ramadan` is a fraction over 30 days, a one-day error shifts it by at most 1/30.

## 9. Final performance

**Production fold — test set (scored once):**

| Metric | Train | Validation | **Test** |
|---|---|---|---|
| R² | 0.8848 | 0.7615 | **0.8524** |
| MAE | 25,271 | 43,040 | **41,627** |
| RMSE | 40,402 | 60,970 | **53,291** |
| WAPE | 0.2936 | 0.4510 | **0.3745** |

Train-to-test R² gap of **+0.032** — the model generalises rather than memorises.

**Per-company test metrics:**

| Company | R² | MAE | RMSE | WAPE |
|---|---|---|---|---|
| C002 | 0.900 | 40,518 | 57,086 | 0.256 |
| C001 | 0.868 | 30,061 | 37,718 | 0.332 |
| C004 | 0.857 | 45,128 | 55,774 | 0.330 |
| C005 | 0.778 | 41,100 | 49,494 | 0.472 |
| C006 | 0.759 | 27,021 | 35,219 | 0.447 |
| C003 | 0.706 | 65,935 | 74,550 | 0.494 |

The stable and cleanly seasonal businesses are the most predictable; the volatile pair is hardest, as expected.

**Stability across three chronological folds (test segments):**

| Metric | fold A | fold B | fold C | mean | std |
|---|---|---|---|---|---|
| R² | 0.4881 | 0.6361 | 0.8524 | 0.6589 | 0.1496 |
| MAE | 54,495 | 52,211 | 41,627 | 49,444 | 5,606 |
| RMSE | 75,985 | 74,564 | 53,291 | 67,947 | 10,379 |
| WAPE | 0.6499 | 0.5227 | 0.3745 | 0.5157 | 0.1125 |

**Read this honestly.** Test R² ranges 0.49 → 0.85 across periods. Folds A and B train on less history (2.25 and 2.75 years versus 3.25) and face harder test windows. The 0.85 figure is the production fold, not a guaranteed expectation. A realistic operating range for this system is **R² 0.5–0.85 depending on regime**, and accuracy should be re-measured as new data arrives rather than assumed constant.

## 10. Downstream risk layer

Derived **after** prediction — these are not columns in the dataset and not model features:

```
projected_balance_30d = closing_balance + predicted_30d_net_cash_flow
projected_coverage    = projected_balance_30d / mean daily outflow (trailing 30d)
Liquidity_Risk_30D    = "At Risk" if projected_coverage < 30 days else "Safe"
Liquidity_Risk_Score  = clip(100 * (1 - projected_coverage / 60), 0, 100)
```

On the test period this flags **18.9%** of days as At Risk, concentrated in the two companies whose economics genuinely deteriorate — non-trivial, and driven by the forecast rather than by the current balance alone.

## 11. Files

| File | Purpose |
|---|---|
| `feature_engineering_30.py` | The 30 features, calendar and financial blocks kept separate |
| `final_pipeline_30.py` | End-to-end: load → features → split → train → predict → evaluate |
| `leakage_audit_30.py` | Four-test empirical audit, exits non-zero on failure |
| `compare_30_model.py` | Three-fold evaluation with mean/std |
| `final_feature_list.json` | The 30 names, types, descriptions and source windows |
| `final_model_metrics.json` | All fold metrics, aggregates, final test metrics |
| `sme_cashflow_dataset.csv` | The dataset (10,956 × 12) |

## 12. Reproducing

```bash
python3 leakage_audit_30.py     # PASS / FAIL, exit code 0 on pass
python3 final_pipeline_30.py    # trains the production model, writes predictions + risk layer
python3 compare_30_model.py     # three-fold stability report, writes final_model_metrics.json
```

Requires `numpy`, `pandas`, `scikit-learn`, `xgboost`. All randomness is seeded (`random_state=42`); the same data produces the same metrics.
