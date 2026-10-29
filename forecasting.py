"""Demand forecasting and stock recommendations for GhostDemand AI.

Inputs (pandas DataFrames)
  requests: date, product, [quantity], [blocker]      <- customer requests (a signal of interest)
  sales:    date, product, units                      <- optional, real sales
  stock:    product, on_hand, lead_time_days, [on_order]  <- optional

Run `python forecasting.py` for a demo on synthetic data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:                                    # optional
    from xgboost import XGBRegressor
except Exception:                       # pragma: no cover
    XGBRegressor = None


@dataclass
class Config:
    horizons: tuple = (7, 30, 90)
    request_conversion: float = 0.3     # share of requests assumed to turn into sales
    service_z: float = 1.65             # about 95% service level
    review_days: int = 30               # how many days one order should cover beyond lead time
    trend_threshold: float = 0.2        # +/-20% change counts as rising or declining
    min_trend_volume: float = 5.0
    min_train_rows: int = 80
    beat_naive_margin: float = 0.05     # a model must beat the baseline by 5% to be used
    restock_unmet_requests: int = 20    # out-of-stock requests in 30 days that trigger a flag
    seed: int = 42


# ----------------------------------------------------------------------------- data prep
def prepare_daily(requests, sales=None, as_of=None):
    """Return {product: DataFrame(requests, sales, oos_requests)} with one row per day."""
    r = requests.copy()
    r["date"] = pd.to_datetime(r["date"]).dt.normalize()
    if "quantity" in r.columns:
        r["quantity"] = pd.to_numeric(r["quantity"], errors="coerce").fillna(1.0).clip(lower=0)
    else:
        r["quantity"] = 1.0
    blocker = r["blocker"].astype(str).str.lower() if "blocker" in r.columns else pd.Series("", index=r.index)
    r["oos"] = np.where(blocker.str.contains("out_of_stock|out of stock"), r["quantity"], 0.0)

    s = None
    if sales is not None and len(sales):
        s = sales.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        s["units"] = pd.to_numeric(s["units"], errors="coerce").fillna(0.0)

    last = r["date"].max() if s is None else max(r["date"].max(), s["date"].max())
    end = pd.Timestamp(as_of).normalize() if as_of is not None else last
    first = r["date"].min() if s is None else min(r["date"].min(), s["date"].min())
    r = r[r["date"] <= end]
    if s is not None:
        s = s[s["date"] <= end]
    idx = pd.date_range(first, end, freq="D")

    products = set(r["product"])
    if s is not None:
        products |= set(s["product"])
    out = {}
    for p in sorted(products):
        rp = r[r["product"] == p]
        req = rp.groupby("date")["quantity"].sum().reindex(idx, fill_value=0.0)
        oos = rp.groupby("date")["oos"].sum().reindex(idx, fill_value=0.0)
        if s is not None:
            sal = s[s["product"] == p].groupby("date")["units"].sum().reindex(idx, fill_value=0.0)
        else:
            sal = pd.Series(0.0, index=idx)
        df = pd.DataFrame({"requests": req, "sales": sal, "oos_requests": oos})
        active = (df["requests"] + df["sales"]) > 0
        if active.any():
            df = df.loc[active.idxmax():]          # drop days before the product first appears
        out[p] = df
    return out


# ----------------------------------------------------------------------------- features
MIN_ORIGIN = 55   # need 56 days of history to build features


def _features(y, t):
    """Features using only days up to and including index t."""
    def s(n):
        return float(y[max(0, t - n + 1): t + 1].sum())
    last28 = s(28)
    prev28 = float(y[t - 55: t - 27].sum())
    slope = (last28 - prev28) / (prev28 + 1.0)
    nonzero = float((y[t - 27: t + 1] > 0).mean())
    return [s(7), s(14), last28, s(56), slope, nonzero]


def _samples(series, h):
    X, Y, test = [], [], []
    for p, y in series.items():
        T = len(y)
        n_orig = T - h - 1 - MIN_ORIGIN
        if n_orig < 1:
            continue
        test_start = MIN_ORIGIN + int(n_orig * 0.75)
        for t in range(MIN_ORIGIN, T - h):
            if t < test_start and t + h > test_start:
                continue                           # window overlaps the test period: skip (no leakage)
            X.append(_features(y, t))
            Y.append(float(y[t + 1: t + 1 + h].sum()))
            test.append(t >= test_start)
    return np.array(X), np.array(Y), np.array(test, dtype=bool)


def _candidates(seed):
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         subsample=0.8, random_state=seed, verbosity=0)
    return models


def _select_model(series, h, cfg):
    """Backtest the candidates against a naive baseline and pick one honestly."""
    X, Y, is_test = _samples(series, h)
    report = {"horizon": h, "model": "naive_rate", "reason": "", "mae": {}}
    if len(X) == 0 or (~is_test).sum() < cfg.min_train_rows or is_test.sum() < 10:
        report["reason"] = "not enough history to train; using recent daily rate"
        return None, report
    Xtr, Ytr, Xte, Yte = X[~is_test], Y[~is_test], X[is_test], Y[is_test]
    naive = Xte[:, 2] * h / 28.0
    report["mae"]["naive_rate"] = float(np.mean(np.abs(Yte - naive)))
    fitted = {}
    for name, est in _candidates(cfg.seed).items():
        est.fit(Xtr, Ytr)
        pred = np.clip(est.predict(Xte), 0, None)
        report["mae"][name] = float(np.mean(np.abs(Yte - pred)))
        fitted[name] = est
    best = min(fitted, key=lambda n: report["mae"][n])
    if report["mae"][best] > report["mae"]["naive_rate"] * (1 - cfg.beat_naive_margin):
        report["reason"] = "no model beat the recent-rate baseline by the required margin"
        return None, report
    final = _candidates(cfg.seed)[best]
    final.fit(X, Y)                                  # refit on all history
    report.update(model=best, reason="best backtest MAE", n_train=int((~is_test).sum()), n_test=int(is_test.sum()))
    return final, report


def _predict(est, y, h):
    if len(y) > MIN_ORIGIN:
        f = _features(y, len(y) - 1)
        if est is None:
            return max(0.0, f[2] * h / 28.0)
        return max(0.0, float(est.predict(np.array([f]))[0]))
    n = max(1, min(28, len(y)))
    return float(y[-n:].mean() * h)


# ----------------------------------------------------------------------------- recommendations
def recommend_reorder(rate_per_day, sigma_per_day, lead_time_days, on_hand, on_order, cfg):
    """Order-up-to: cover lead time + review period, plus safety stock, minus stock held."""
    cover = lead_time_days + cfg.review_days
    target = rate_per_day * cover + cfg.service_z * sigma_per_day * math.sqrt(cover)
    return max(0, math.ceil(target - on_hand - on_order))


def _trend(y, cfg):
    if len(y) < 56:
        return "not enough history", None
    last, prev = float(y[-28:].sum()), float(y[-56:-28].sum())
    if max(last, prev) < cfg.min_trend_volume:
        return "low volume", None
    change = (last - prev) / max(prev, 1.0)
    label = "rising" if change > cfg.trend_threshold else "declining" if change < -cfg.trend_threshold else "stable"
    return label, round(change * 100, 1)


def forecast_products(requests, sales=None, stock=None, as_of=None, config=None):
    """Return {'forecasts': DataFrame, 'model_report': list} for every product."""
    cfg = config or Config()
    daily = prepare_daily(requests, sales, as_of)
    k = cfg.request_conversion
    series = {p: (d["sales"] + k * d["requests"]).to_numpy(float) for p, d in daily.items()}

    estimators, reports = {}, []
    for h in cfg.horizons:
        est, rep = _select_model(series, h, cfg)
        estimators[h] = est
        reports.append(rep)

    stock_map = {}
    if stock is not None and len(stock):
        stock_map = stock.set_index("product").to_dict("index")

    rows = []
    for p, y in series.items():
        d = daily[p]
        preds = {h: _predict(estimators[h], y, h) for h in cfg.horizons}
        rate = preds[30] / 30.0 if 30 in preds else float(y[-28:].mean())
        sigma = float(np.std(y[-56:])) if len(y) > 1 else 0.0
        trend, change = _trend(y, cfg)
        has_sales = bool(d["sales"].sum() > 0)
        unmet30 = float(d["oos_requests"].iloc[-30:].sum())

        st = stock_map.get(p)
        notes, qty, cover_days = [], None, None
        stockout_risk = False
        if st is not None:
            on_hand = float(st.get("on_hand", 0) or 0)
            on_order = float(st.get("on_order", 0) or 0)
            lead = float(st.get("lead_time_days", 0) or 0)
            qty = recommend_reorder(rate, sigma, lead, on_hand, on_order, cfg)
            cover_days = round(on_hand / rate, 1) if rate > 0 else None
            stockout_risk = rate > 0 and on_hand + on_order < rate * lead
            if stockout_risk:
                notes.append("stock will run out before a new order could arrive")
        else:
            notes.append("add stock and lead time for a reorder quantity")
        if not has_sales:
            notes.append("no sales data: demand is estimated from requests only")

        beat = any(r["model"] != "naive_rate" for r in reports)
        confidence = "low"
        if len(y) >= 60:
            confidence = "medium"
        if len(y) >= 180 and has_sales and beat:
            confidence = "high"

        rows.append({
            "product": p, "history_days": len(y),
            **{f"demand_{h}d": round(preds[h], 1) for h in cfg.horizons},
            "trend": trend, "trend_change_pct": change,
            "requests_30d": float(d["requests"].iloc[-30:].sum()),
            "unmet_requests_30d": unmet30,
            "days_of_cover": cover_days,
            "recommended_reorder_qty": qty,
            "restock_priority": bool(unmet30 >= cfg.restock_unmet_requests or stockout_risk),
            "confidence": confidence, "notes": "; ".join(notes),
        })
    table = pd.DataFrame(rows).sort_values(["restock_priority", "demand_30d"], ascending=False).reset_index(drop=True)
    return {"forecasts": table, "model_report": reports}


# ----------------------------------------------------------------------------- demo data
def make_demo_data(days=365, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    spec = {   # name: (base daily requests, trend per day, out-of-stock share)
        "Size 42 sneakers": (2.0, 0.004, 0.55), "Phone charger": (3.0, 0.0, 0.05),
        "School bag": (1.5, -0.002, 0.05), "Cooking oil 5L": (2.5, 0.001, 0.10),
        "Blue denim jacket": (1.0, 0.003, 0.20), "Solar lamp": (1.2, 0.0, 0.08),
    }
    req_rows, sale_rows = [], []
    weekly = np.array([0.8, 0.9, 0.9, 1.0, 1.2, 1.4, 0.8])
    for name, (base, slope, oos) in spec.items():
        for i, dt in enumerate(dates):
            lam = max(0.05, (base + slope * i) * weekly[dt.dayofweek])
            n = rng.poisson(lam)
            for _ in range(n):
                blk = "out_of_stock" if rng.random() < oos else rng.choice(["price_too_high", "not_stocked", ""])
                req_rows.append((dt, name, 1, blk))
            units = rng.poisson(lam * 1.8)
            if units:
                sale_rows.append((dt, name, units))
    requests = pd.DataFrame(req_rows, columns=["date", "product", "quantity", "blocker"])
    sales = pd.DataFrame(sale_rows, columns=["date", "product", "units"])
    stock = pd.DataFrame({
        "product": list(spec),
        "on_hand": [4, 300, 90, 120, 25, 60],
        "lead_time_days": [14, 7, 21, 10, 14, 10],
        "on_order": [0, 0, 0, 0, 0, 0],
    })
    return requests, sales, stock


if __name__ == "__main__":
    req, sal, stk = make_demo_data()
    result = forecast_products(req, sal, stk)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(result["forecasts"].drop(columns=["notes"]).to_string(index=False))
    print("\nBacktest (mean absolute error, lower is better):")
    for rep in result["model_report"]:
        print(f"  {rep['horizon']:>2}-day: used {rep['model']:<18} {rep['reason']}")
        for name, mae in rep["mae"].items():
            print(f"        {name:<18} {mae:8.2f}")
