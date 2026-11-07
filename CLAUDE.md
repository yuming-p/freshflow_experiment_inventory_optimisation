# Freshflow inventory management demo

Demo for a Data Scientist (Operations Research) application at Freshflow (inventory optimisation for perishables in grocery retail). Use FreshRetailNet-50K stratified samples, experiment with LightGBM quantile forecasts and calibration, evaluate with simulated data on waste, stockouts and profit.

Principles: pragmatic and simple over elegant. Every reported number reproducible from code. 

## Status
- Done: data preparation, LightGBM quantile models, conformal correction, 5-week rolling backtest,
  simulator + tests, `run_simulation.py`, README.

## Data
FreshRetailNet-50K (Hugging Face `Dingdong-Inc/FreshRetailNet-50K`, CC BY 4.0).
Daily rows per store-product, hourly sales and hourly stockout flags. IDs anonymised.
Preparation is done in `notebooks/prepare_data_freshnet_retail50k.ipynb` (don't redo unless asked).
Key facts:
- Categories 29, 16, 4. Sales × 10 as integers.
- `sale_amount_recovered` = stockout-corrected demand; the demand truth for target,
  calibration, evaluation and simulation. Raw sales: `sale_amount`.
- Lag features built from `sale_amount_recovered`, shifted by 1 day.
- Splits: train to 2024-06-11, val to 2024-06-25, eval 2024-06-26 to 07-02.
- Files: `data/{train,val,eval}.parquet`, `_small` versions used for modelling.

## Assumptions (illustrative, state them in any write-up)
| Category | Role | Shelf life | Price | Cost | τ |
|---|---|---|---|---|---|
| 29 | bread | 1 | 2.00 | 0.80 | 0.60 |
| 16 | meat | 3 | 6.00 | 4.00 | 0.75 |
| 4 | vegetables | 5 | 2.50 | 1.50 | 0.88 |

τ = critical ratio with carry-over: margin / (margin + cost × P(unit eventually expires)).
Daily delivery before opening; order placed the evening before with known end-of-day stock.
Expired stock worth 0. Effective price = price × discount. No case packs, holding cost 0.
Customer picking: 70% oldest first, 30% freshest first.

## Modelling (notebook)
- LightGBM `objective="quantile"`, one global model per τ (trained on all categories, used for its own).
  Features: lags/rollings, weekday, discount, holiday_flag, activity_flag, weather, categorical IDs
  (category levels fixed from train).
- Val split: first 7 days early stopping (`val_es`), last 7 days conformal calibration (`val_cal`).
- Conformal: per-category shift = finite-sample τ-quantile of (y − q) on `val_cal`;
  `q_lgbm_conf = q_lgbm + shift`, clipped at 0.
- Backtest: 7-day blocks counted back from 2024-06-25; for test week w, train on weeks ≥ w+2,
  calibrate on w+1; 5 folds; n_estimators reused from main models' best_iteration_.

## Results so far
Coverage (target = τ), eval week raw → conformal: bread 0.48 → 0.59, meat 0.71 → 0.79, veg 0.79 → 0.88.
Backtest means raw → conformal: bread 0.55 → 0.59, meat 0.72 → 0.74, veg 0.87 → 0.86;
pinball changes ≤ 1%. Interpretation: raw quantiles nearly calibrated; eval week had a demand jump;
remaining under-coverage comes from trend, not miscalibration.

## Code
- `src/simulator.py`: `simulate_series` (one item, stock by remaining shelf life),
  `run_policy`, `summarise` (fill_rate, waste_rate, profit; end-of-horizon stock valued at cost),
  `compare_policies` (adds profit_vs_perfect).
- `run_simulation.py`: reads `data/eval_predictions.parquet` (eval set + target columns
  mean_7, q_lgbm, q_lgbm_conf), writes `results/*.csv`. Policies: naive 7-day mean, LightGBM,
  LightGBM + conformal, perfect foresight.
- `tests/test_simulator.py`: hand-computed meat trace, bread, picking, rounding, unit conservation,
  perfect foresight. Run `uv run pytest -v` (pytest config in pyproject.toml).


## Rules
- No leakage: features at day t use only days before t; nothing learned from eval data.
- Don't tune on `val_cal` or eval. Fixed seeds.
- Small steps; show a table or plot after each and wait for review.
- Logic in `src/` with tests; notebooks only run and display.