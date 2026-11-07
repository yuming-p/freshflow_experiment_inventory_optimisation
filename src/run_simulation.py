"""Turn forecasts into ordering decisions and evaluate them in the inventory simulator.

Usage:
    python src/run_simulation.py data/eval_predictions.parquet --out results/

The input file is the eval set with one column per policy target, saved from the
modelling notebook, for example:
    eval_.to_parquet("data/eval_predictions.parquet", index=False)

It must contain: store_id, product_id, dt, first_category_id, shelf_life, price,
cost, discount, sale_amount_recovered, and the target columns listed in POLICIES.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import retail_configs
from simulator import DEMAND_COL, compare_policies, run_policy, summarise

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# policy name -> column holding the order-up-to target for each item-day
POLICIES = {
    "naive (7-day mean)": "mean_7",
    "naive (mean + k*std)": "naive_k",
    "LightGBM": "q_lgbm",
    "LightGBM + conformal": "q_lgbm_conf",
    "perfect foresight": DEMAND_COL,
}

K_GRID = np.arange(-1.0, 3.01, 0.25)  # safety-stock multipliers tried for the simple rule

FIFO_SHARES = [1.0, 0.7, 0.5]  # share of customers taking the oldest units first
BASE_FIFO_SHARE = 0.7

SUMMARY_COLS = ["category", "policy", "fill_rate", "waste_rate", "profit_vs_perfect"]


def category_names():
    """Map category IDs to names from retail_configs."""
    return {c: cfg["category_name"] for c, cfg in retail_configs.CATEGORY_CONFIG.items()}


def add_category_names(results):
    names = category_names()
    results["category"] = results["first_category_id"].map(lambda c: names.get(c, str(c)))
    return results


def k_target(df, k):
    """Order-up-to level: 7-day mean plus k recent standard deviations, never below 0."""
    return (df["mean_7"] + k * df["std_7"].fillna(0)).clip(lower=0)


def tune_k(train):
    """Per category, the k with the highest simulated profit on training-period data.

    Eval data is never used. Rows without a 7-day mean (start of each series) are dropped.
    """
    train = train.dropna(subset=["mean_7"])
    profits = {}
    for k in K_GRID:
        sim = run_policy(train.assign(target=k_target(train, k)), "target",
                         fifo_share=BASE_FIFO_SHARE)
        profits[k] = summarise(sim)["profit"]
    return pd.DataFrame(profits).idxmax(axis=1).to_dict()


def load_predictions(path, tune_path):
    df = pd.read_parquet(path)
    df["dt"] = pd.to_datetime(df["dt"])
    train = pd.read_parquet(tune_path)
    train["dt"] = pd.to_datetime(train["dt"])
    ks = tune_k(train)
    print("Tuned k per category (training period):", ks)
    k_per_row = df["first_category_id"].map(ks)
    df["naive_k"] = (df["mean_7"] + k_per_row * df["std_7"].fillna(0)).clip(lower=0)
    missing = [col for col in POLICIES.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in {path}: {missing}")
    return df


def main_comparison(df):
    """All policies at the base customer-picking assumption."""
    results = compare_policies(df, POLICIES, fifo_share=BASE_FIFO_SHARE)
    return add_category_names(results)


def picking_sensitivity(df):
    """How waste and profit change with customer picking behaviour."""
    tables = []
    for share in FIFO_SHARES:
        r = compare_policies(df, POLICIES, fifo_share=share)
        r["fifo_share"] = share
        tables.append(r)
    results = add_category_names(pd.concat(tables, ignore_index=True))
    return results[["fifo_share"] + SUMMARY_COLS]


def plot_tradeoff(results, path):
    """Fill rate vs waste rate, one point per policy, one panel per category."""
    shown = results[results["policy"] != "perfect foresight"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, ["bread", "meat", "vegetables"]):
        for i, (_, r) in enumerate(shown[shown["category"] == name].iterrows()):
            ax.scatter(100 * r["waste_rate"], 100 * r["fill_rate"], s=60,
                       color=f"C{i}", label=r["policy"])
        ax.set_title(name)
        ax.set_xlabel("waste (% of units ordered)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fill rate (% of demand served)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_trace(df, target_col="q_lgbm"):
    """Day-by-day simulation for the first store-product, for illustration."""
    detail = run_policy(df, target_col, fifo_share=BASE_FIFO_SHARE)
    first = detail.iloc[0]
    one = detail[(detail["store_id"] == first["store_id"]) &
                 (detail["product_id"] == first["product_id"])]
    return one[["store_id", "product_id", "dt", DEMAND_COL,
                "order", "sold", "lost", "waste", "end_stock"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", help="parquet file with eval data and target columns")
    parser.add_argument("--tune-data", default="data/train_small.parquet",
                        help="training-period parquet used to pick k for the simple rule")
    parser.add_argument("--out", default="results", help="folder for the output CSV files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions, args.tune_data)

    with pd.option_context("display.width", 120, "display.max_columns", 20):
        results = main_comparison(df)
        results.to_csv(out / "policy_comparison.csv", index=False)
        print(f"\nPolicy comparison (fifo_share = {BASE_FIFO_SHARE})")
        print(results[SUMMARY_COLS].round(3).to_string(index=False))

        plot_tradeoff(results, out / "tradeoff.png")

        sensitivity = picking_sensitivity(df)
        sensitivity.to_csv(out / "picking_sensitivity.csv", index=False)
        print("\nSensitivity to customer picking behaviour")
        print(sensitivity.round(3).to_string(index=False))

        trace = example_trace(df)
        trace.to_csv(out / "example_trace.csv", index=False)
        print("\nExample: one store-product, day by day (LightGBM)")
        print(trace.to_string(index=False))

    print(f"\nSaved CSV files to {out.resolve()}")


if __name__ == "__main__":
    main()