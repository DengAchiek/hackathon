import numpy as np
import pandas as pd
from forecasting import Config, forecast_products, make_demo_data, recommend_reorder


def test_demo_runs_and_has_all_horizons():
    req, sal, stk = make_demo_data(days=300)
    out = forecast_products(req, sal, stk)
    t = out["forecasts"]
    assert {"demand_7d", "demand_30d", "demand_90d", "trend", "recommended_reorder_qty"} <= set(t.columns)
    assert len(t) == 6 and (t[["demand_7d", "demand_30d", "demand_90d"]] >= 0).all().all()
    assert len(out["model_report"]) == 3


def test_out_of_stock_product_is_flagged():
    req, sal, stk = make_demo_data(days=300)
    t = forecast_products(req, sal, stk)["forecasts"].set_index("product")
    assert bool(t.loc["Size 42 sneakers", "restock_priority"])


def test_rising_and_declining_trends():
    dates = pd.date_range("2026-01-01", periods=120)
    rows = []
    for i, d in enumerate(dates):
        rows += [(d, "up", 1, "")] * int(1 + i / 10)
        rows += [(d, "down", 1, "")] * int(max(1, 12 - i / 10))
    req = pd.DataFrame(rows, columns=["date", "product", "quantity", "blocker"])
    t = forecast_products(req).get("forecasts").set_index("product")
    assert t.loc["up", "trend"] == "rising"
    assert t.loc["down", "trend"] == "declining"


def test_reorder_is_zero_when_stock_is_plentiful():
    assert recommend_reorder(2.0, 1.0, 7, on_hand=10_000, on_order=0, cfg=Config()) == 0
    assert recommend_reorder(2.0, 1.0, 7, on_hand=0, on_order=0, cfg=Config()) > 0


def test_short_history_and_no_sales_do_not_crash():
    req = pd.DataFrame({"date": pd.date_range("2026-03-01", periods=10), "product": "x"})
    out = forecast_products(req)["forecasts"]
    assert out.loc[0, "confidence"] == "low"
    assert out.loc[0, "demand_30d"] >= 0
