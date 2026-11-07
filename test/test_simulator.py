import numpy as np
import pandas as pd

from simulator import compare_policies, simulate_series


def test_meat_trace():
    """Hand-computed case: shelf life 3, everyone takes the oldest first, target 12."""
    sim = simulate_series(demand=np.array([10, 13, 6, 4, 1]),
                          target=np.full(5, 12.0), shelf_life=3, fifo_share=1.0)
    assert sim["order"].tolist() == [12, 10, 12, 6, 4]
    assert sim["sold"].tolist() == [10, 12, 6, 4, 1]
    assert sim["lost"].tolist() == [0, 1, 0, 0, 0]
    assert sim["waste"].tolist() == [0, 0, 0, 0, 1]


def test_bread_one_day_shelf_life():
    """Nothing carries over: every leftover is wasted the same night."""
    sim = simulate_series(demand=np.array([3, 7]),
                          target=np.array([5.0, 5.0]), shelf_life=1)
    assert sim["order"].tolist() == [5, 5]
    assert sim["sold"].tolist() == [3, 5]
    assert sim["lost"].tolist() == [0, 2]
    assert sim["waste"].tolist() == [2, 0]


def test_freshest_first_wastes_more():
    """Customers taking the freshest units leave old stock to expire."""
    demand = np.array([5, 5, 5, 5, 5, 5])
    target = np.full(6, 8.0)
    fifo = simulate_series(demand, target, shelf_life=3, fifo_share=1.0)
    lifo = simulate_series(demand, target, shelf_life=3, fifo_share=0.0)
    assert lifo["waste"].sum() > fifo["waste"].sum()


def test_fractional_target_rounds_up():
    sim = simulate_series(demand=np.array([0]), target=np.array([4.2]), shelf_life=2)
    assert sim["order"].tolist() == [5]


def test_units_are_conserved():
    """Everything ordered is either sold, wasted, or still on the shelf."""
    rng = np.random.default_rng(0)
    demand = rng.poisson(10, size=30)
    target = rng.uniform(5, 15, size=30)
    sim = simulate_series(demand, target, shelf_life=4, fifo_share=0.7)
    assert sim["order"].sum() == (sim["sold"].sum() + sim["waste"].sum()
                                  + sim["end_stock"].iloc[-1])
    assert (sim["sold"] + sim["lost"]).tolist() == demand.tolist()


def test_perfect_foresight_has_no_loss_or_waste():
    dates = pd.date_range("2024-06-26", periods=4)
    df = pd.DataFrame({
        "store_id": 1, "product_id": 1, "dt": dates, "first_category_id": 16,
        "shelf_life": 3, "price": 6.0, "cost": 4.0, "discount": 1.0,
        "sale_amount_recovered": [7, 3, 9, 0],
    })
    res = compare_policies(df, {"perfect foresight": "sale_amount_recovered"})
    row = res.iloc[0]
    assert row["lost"] == 0
    assert row["waste"] == 0
    assert row["profit_vs_perfect"] == 1.0