"""Inventory simulation for perishable items under an order-up-to policy.

Each evening the policy orders enough to bring stock up to a target level for the
next day; the order arrives the next morning with full shelf life. During the day
customers buy (a share takes the oldest units first, the rest the freshest), and at
night units on their last day of shelf life are thrown away.
"""

import math

import numpy as np
import pandas as pd

DEMAND_COL = "sale_amount_recovered"


def simulate_series(demand, target, shelf_life, fifo_share=0.7):
    """Simulate one store-product over consecutive days.

    Args:
        demand: array of daily demand in units.
        target: array of order-up-to levels, one per day (same length as demand).
        shelf_life: days a unit can be sold, counting the delivery day.
        fifo_share: share of customers who take the oldest units first.

    Returns:
        DataFrame with one row per day: order, sold, lost, waste, end_stock.
    """
    L = int(shelf_life)
    stock = np.zeros(L + 1, dtype=int)  # stock[r] = units with r days left (r = 1..L)
    rows = []
    for t in range(len(demand)):
        # evening before day t: order up to the target
        on_hand = int(stock.sum())
        order = max(0, math.ceil(round(float(target[t]) - on_hand, 6)))
        stock[L] += order  # morning delivery, full shelf life

        # during day t: serve demand
        d = int(demand[t])
        d_fifo = round(fifo_share * d)
        d_lifo = d - d_fifo
        sold = 0
        for r in range(1, L + 1):  # these customers take the oldest first
            take = min(stock[r], d_fifo)
            stock[r] -= take
            d_fifo -= take
            sold += take
        for r in range(L, 0, -1):  # the rest take the freshest first
            take = min(stock[r], d_lifo)
            stock[r] -= take
            d_lifo -= take
            sold += take
        lost = d - sold

        # night: last-day units are thrown away, the rest age by one day
        waste = int(stock[1])
        stock[1:L] = stock[2 : L + 1]
        stock[L] = 0

        rows.append(
            {"order": order, "sold": int(sold), "lost": int(lost),
             "waste": waste, "end_stock": int(stock.sum())}
        )
    return pd.DataFrame(rows)


def run_policy(df, target_col, demand_col=DEMAND_COL, fifo_share=0.7):
    """Simulate every store-product in df, ordering up to df[target_col] each day.

    df needs: store_id, product_id, dt, first_category_id, shelf_life,
    price, cost, discount, the demand column and the target column.
    """
    keep = ["store_id", "product_id", "dt", "first_category_id",
            "price", "cost", "discount", demand_col]
    out = []
    groups = df.sort_values("dt").groupby(["store_id", "product_id"], observed=True)
    for _, g in groups:
        sim = simulate_series(
            g[demand_col].to_numpy(),
            g[target_col].fillna(0).to_numpy(),
            int(g["shelf_life"].iloc[0]),
            fifo_share,
        )
        sim.index = g.index
        out.append(pd.concat([g[keep], sim], axis=1))
    res = pd.concat(out)
    res["revenue"] = res["sold"] * res["price"] * res["discount"]  # promo days earn less
    res["purchase_cost"] = res["order"] * res["cost"]
    return res


def summarise(res, demand_col=DEMAND_COL):
    """Aggregate a run_policy result per category.

    Stock left after the last day is valued at cost: it is not waste yet and could
    still be sold the following week.
    """
    last_day = res["dt"].max()
    res = res.assign(
        end_value=np.where(res["dt"] == last_day, res["end_stock"] * res["cost"], 0.0)
    )
    cols = [demand_col, "sold", "lost", "waste", "order",
            "revenue", "purchase_cost", "end_value"]
    s = res.groupby("first_category_id", observed=True)[cols].sum()
    s["fill_rate"] = s["sold"] / s[demand_col]   # share of demand served
    s["waste_rate"] = s["waste"] / s["order"]    # share of ordered units thrown away
    s["profit"] = s["revenue"] - s["purchase_cost"] + s["end_value"]
    return s


def compare_policies(df, policies, demand_col=DEMAND_COL, fifo_share=0.7,
                     reference="perfect foresight"):
    """Run several policies and report each relative to the reference policy's profit.

    policies: dict of {policy name: target column}.
    """
    results = []
    for name, col in policies.items():
        s = summarise(run_policy(df, col, demand_col, fifo_share), demand_col).reset_index()
        s["policy"] = name
        results.append(s)
    results = pd.concat(results, ignore_index=True)

    if reference in policies:
        best = (results[results["policy"] == reference]
                .set_index("first_category_id")["profit"])
        results["profit_vs_perfect"] = results["profit"] / results["first_category_id"].map(best)
    return results