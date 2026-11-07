# Freshflow inventory optimisation Simple demo

A small demo that uses FreshRetailNet-50K stratified sample data, LightGBM quantile model for frecasting, and simulation with naive rules for decision making.


## Data
[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong, CC BY 4.0):
Key facts:
- Categories 29, 16, 4. Sales × 10 as integers.
- `sale_amount_recovered` = stockout-corrected demand; the demand truth for target,
  calibration, evaluation and simulation. Raw sales: `sale_amount`.
- Lag features built from `sale_amount_recovered`, shifted by 1 day.
- Splits: train to 2024-06-11, val to 2024-06-25, eval 2024-06-26 to 07-02.
- Files: `data/{train,val,eval}.parquet`, `_small` versions used for modelling.
daily sales with hourly stockout flags. IDs are anonymised, so categories get illustrative roles:

| Category | Role | Shelf life | Price / cost | Target quantile τ |
|---|---|---|---|---|
| 29 | bread | 1 day | 2.00 / 0.80 | 0.60 |
| 16 | meat | 3 days | 6.00 / 4.00 | 0.75 |
| 4 | vegetables | 5 days | 2.50 / 1.50 | 0.88 |



## Approach
1. **Censored demand.** Sales understate demand on stockout days (43–65% of days have some).
   Sales are scaled up by in-stock hours (capped at 1.6×); full-day stockouts use the last valid day.
2. **Forecast.** LightGBM quantile regression, one model per τ, with lag and rolling features,
   calendar, discount and weather.
3. **Calibrate (the "complex" step).** Split conformal: a per-category shift learned on a held-out week.
4. **Decide.** Order-up-to: each evening, order `max(0, predicted τ-quantile − stock on hand)`.
   τ accounts for carry-over: leftovers of longer-life items rarely expire, so they get a higher target.
5. **Simulate.** Stock tracked by remaining shelf life; 70% of customers take the oldest units;
   expired units are wasted. Compared against a naive rule (7-day mean) and perfect foresight.

Splits: 76 days train, 7 days early stopping, 7 days calibration, 7 days final test.

## Results so far
Coverage = share of days where demand ≤ predicted quantile (should equal τ).

| Category | τ | Raw, test week | + conformal, test week | Raw, 5-week backtest | + conformal, backtest |
|---|---|---|---|---|---|
| bread | 0.60 | 0.48 | 0.59 | 0.55 | 0.59 |
| meat | 0.75 | 0.71 | 0.79 | 0.72 | 0.74 |
| vegetables | 0.88 | 0.79 | 0.88 | 0.87 | 0.86 |

- Raw quantiles run slightly low on average, implying mild under-ordering.
- In the test week, conformal cut the newsvendor (pinball) loss by 4–7% for bread and vegetables.
- Across a 5-week rolling backtest, the gain shrinks to about 1% or less: the test week had a demand
  jump that the calibration week happened to anticipate.
- **Takeaway: plain LightGBM quantiles are nearly calibrated; the remaining gap comes from
  demand trends, not miscalibration.** The simulation below does show conformal helping meat and vegetables
  in the test week, but that week had an upward demand jump the shift happened to anticipate, so treat
  the gain as indicative.

### Simulation (test week, 70% oldest-first picking)
Fill rate = sold / demand; waste rate = expired units / units ordered (source: `results/policy_comparison.csv`).
Profit is shown as % of perfect foresight (orders exactly the true demand).

| Category | Policy | Fill rate | Waste rate | Profit, % of perfect |
|---|---|---|---|---|
| bread | naive (7-day mean) | 84.9% | 15.4% | 73.2% |
| bread | LightGBM | 80.2% | 12.0% | 71.9% |
| bread | LightGBM + conformal | 84.8% | 16.5% | 72.1% |
| meat | naive (7-day mean) | 85.2% | 0.1% | 85.2% |
| meat | LightGBM | 91.2% | 0.8% | 89.5% |
| meat | LightGBM + conformal | 93.8% | 1.3% | 91.0% |
| vegetables | naive (7-day mean) | 84.8% | 0.0% | 85.4% |
| vegetables | LightGBM | 94.3% | 0.0% | 94.2% |
| vegetables | LightGBM + conformal | 96.4% | 0.1% | 96.2% |

- **Meat and vegetables:** LightGBM beats the naive rule by 4–9 profit points, and conformal adds another
  1.5–2 points through higher fill rate at almost no extra waste (long shelf life absorbs the extra stock).
- **Bread:** all policies land at 72–73% of perfect. One-day shelf life means every over-order is wasted
  (12–17% of units), so better forecasts barely move profit; conformal buys fill rate back
  (80% → 85%) at the price of more waste, which nets out to about +0.2 points.
- Picking rule matters little for bread (nothing carries over) but for meat, moving from 100% to 50% oldest-first
  raises conformal waste from 0.7% to 2.0% and cuts profit from 92.2% to 89.4% of perfect
  (`results/picking_sensitivity.csv`).
- Caveats: one test week with a demand jump (which favours the upward conformal shift), no confidence
  intervals, assumed prices and costs. Treat the ranking as indicative, not proven.

## Run it
```bash
uv sync
uv run pytest -v                                              # simulator tests
uv run python src/run_simulation.py data/eval_predictions.parquet --out results/
```
Data preparation and modelling are in `notebooks/`; simulator logic in `src/simulator.py`.

## Limitations and next steps
- Demand on stockout days is an estimate; categories, prices and shelf lives are assumptions.
- One test week plus a 5-week backtest; the simulation starts with empty shelves.
- Policies sit at different service levels (fill rate vs waste), so a profit gap mixes forecast quality
  with how aggressively each policy orders. Read `results/tradeoff.png` with that in mind; a full
  sweep over service levels would separate the two.
- Model choice kept deliberately simple; a hierarchical model (shared structure across stores and products) is a possible next step.
- Next: compare one-day vs. carry-over τ, handle trends (relative targets, recency weighting),
  and add case packs and shared constraints, where a MILP starts to beat simple rules.



## Resources
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
https://www.kaggle.com/datasets/likithagedipudi/perishable-goods-management
https://arxiv.org/pdf/2210.12496
https://arxiv.org/pdf/2507.23390
https://www.lokad.com/blog/2026/2/27/when-you-think-you-need-an-inventory-forecast/
https://arxiv.org/pdf/2210.03137