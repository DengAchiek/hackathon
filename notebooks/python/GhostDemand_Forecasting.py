# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Python cell notebook: GhostDemand Forecasting
#
# This is ordinary Python, arranged into `# %%` notebook cells.
# Open it in VS Code with the Python and Jupyter extensions and select
# **Run Cell** or **Run All**, or run `python GhostDemand_Forecasting.py`.
# Install `requirements.txt` first. See `README.md` for the full setup.
#
# Source: `notebooks/GhostDemand_Forecasting.ipynb`. Original explanations and model
# logic are retained. Outputs go to a separate `runs/GhostDemand_Forecasting/`
# folder. The scripts use practice data and never deploy a live model.

# %%
from pathlib import Path
import os
from IPython import get_ipython
from IPython.display import display

# In VS Code, open this file and use Run Cell or Run All. In Jupyter, first
# change to the folder containing the Python notebooks. Python scripts also work.
NOTEBOOK_DIR = (Path(__file__).resolve().parent if "__file__" in globals()
                else globals().get("NOTEBOOK_DIR", Path.cwd().resolve()))
DATA_DIR = NOTEBOOK_DIR / "data"
RUN_DIR = NOTEBOOK_DIR / "runs" / "GhostDemand_Forecasting"
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)
os.environ.setdefault("MPLCONFIGDIR", str(RUN_DIR / ".matplotlib"))

IN_NOTEBOOK = get_ipython() is not None
import matplotlib
if IN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

_plot_number = 0
def show_plot():
    """Display charts interactively, or save them when run with python."""
    global _plot_number
    if IN_NOTEBOOK:
        plt.show()
        return
    for number in plt.get_fignums():
        _plot_number += 1
        path = RUN_DIR / f"figure_{_plot_number:02d}.png"
        plt.figure(number).savefig(path, bbox_inches="tight", dpi=140)
        print("Saved chart:", path)
    plt.close("all")

print("Learning outputs:", RUN_DIR)


# %% [markdown]
# # GhostDemand AI: Demand Forecasting and Stock Recommendations
#
# This notebook forecasts demand for each product over the next 7, 30 and 90 days, spots products with rising or falling interest, and suggests how much to reorder.
#
# **Important:** customer requests show interest, not guaranteed sales. The notebook combines requests with real sales, stock and supplier lead times wherever they are available.
#
# **How to use it**
# - Run all cells to see a demo on synthetic data.
# - To use your own data, put `requests.csv`, and optionally `sales.csv` and `stock.csv`, next to this notebook. The column names are listed in section 2.

# %% [markdown]
# ## 1. Set up
# The next cell defines the forecasting helpers directly in Python. It does not write or replace an application module.

# %%
"""Demand forecasting and stock recommendations for GhostDemand AI.

Inputs (pandas DataFrames)
  requests: date, product, [quantity], [blocker]      <- customer requests (a signal of interest)
  sales:    date, product, units                      <- optional, real sales
  stock:    product, on_hand, lead_time_days, [on_order]  <- optional

Run `python forecasting.py` for a demo on synthetic data.
"""

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


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Forecasting helpers were defined in the preceding Python cell.

sns.set_style("whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## 2. Load the data
# Expected columns:
#
# | File | Columns |
# |---|---|
# | `requests.csv` | `date`, `product`, optional `quantity`, optional `blocker` (use `out_of_stock` for items that were out of stock) |
# | `sales.csv` (optional) | `date`, `product`, `units` |
# | `stock.csv` (optional) | `product`, `on_hand`, `lead_time_days`, optional `on_order` |
#
# If `requests.csv` is not found, synthetic demo data is used.

# %%
if (DATA_DIR / "requests.csv").exists():
    requests = pd.read_csv(DATA_DIR / "requests.csv")
    sales = pd.read_csv(DATA_DIR / "sales.csv") if (DATA_DIR / "sales.csv").exists() else None
    stock = pd.read_csv(DATA_DIR / "stock.csv") if (DATA_DIR / "stock.csv").exists() else None
    print("Using your files.")
else:
    requests, sales, stock = make_demo_data()
    print("requests.csv not found. Using SYNTHETIC demo data.")

print("Requests:", len(requests), "rows,", requests["product"].nunique(), "products")
print("Date range:", pd.to_datetime(requests["date"]).min().date(), "to", pd.to_datetime(requests["date"]).max().date())
print("Sales data:", "yes" if sales is not None else "no", "| Stock data:", "yes" if stock is not None else "no")
display(requests.head())

# %% [markdown]
# ## 3. Explore the data
# Before forecasting I check which products get the most requests and whether there is a weekly pattern.

# %%
req = requests.copy()
req["date"] = pd.to_datetime(req["date"])
if "quantity" not in req.columns:
    req["quantity"] = 1

recent = req[req["date"] >= req["date"].max() - pd.Timedelta(days=90)]
by_product = recent.groupby("product")["quantity"].sum().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
by_product.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Requests per product (last 90 days)")

weekday = req.groupby(req["date"].dt.day_name())["quantity"].sum()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday.reindex(order).plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_title("Requests by day of the week")
axes[1].set_xlabel("")
plt.tight_layout()
show_plot()

# %%
# How many requests are blocked by stock-outs?
if "blocker" in req.columns:
    blockers = req["blocker"].fillna("").replace("", "none").value_counts()
    print(blockers)
else:
    print("No blocker column in this data.")

# %% [markdown]
# ## 4. Forecast demand and recommend stock
# How it works, in plain words:
# 1. Demand per day is **sales plus 30% of requests** (requests are only a signal of interest).
# 2. For each horizon (7, 30, 90 days) the code tests Linear Regression, Random Forest and (if installed) XGBoost on recent history it did not train on.
# 3. A model is used only if it beats a simple baseline ("the recent daily rate continues") by at least 5%.
# 4. Reorder quantity covers the supplier lead time plus a 30-day review period, with a safety margin, minus the stock held.

# %%
result = forecast_products(requests, sales, stock)
forecasts = result["forecasts"]
display(forecasts.drop(columns=["notes"]))

# %% [markdown]
# ### 4.1 Which model was used?
# Mean absolute error (MAE) on the hold-out period. Lower is better. If no model beats `naive_rate` by enough, the baseline is used.

# %%
rows = []
for rep in result["model_report"]:
    row = {"horizon_days": rep["horizon"], "model_used": rep["model"]}
    row.update({k: round(v, 2) for k, v in rep["mae"].items()})
    rows.append(row)
backtest = pd.DataFrame(rows)
display(backtest)

mae_cols = [c for c in backtest.columns if c not in ("horizon_days", "model_used")]
if mae_cols:
    backtest.set_index("horizon_days")[mae_cols].plot(kind="bar", figsize=(8, 4))
    plt.title("Backtest error by horizon (lower is better)")
    plt.ylabel("Mean absolute error (units)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    show_plot()
else:
    print("Not enough history to run a backtest, so the recent daily rate was used.")

# %% [markdown]
# ## 5. Products to restock first
# A product is flagged when many recent requests were blocked by "out of stock", or when its stock would run out before a new order could arrive.

# %%
cols = ["product", "demand_30d", "unmet_requests_30d", "days_of_cover", "recommended_reorder_qty", "confidence", "notes"]
priority = forecasts[forecasts["restock_priority"]][cols]
if len(priority):
    display(priority)
else:
    print("No products flagged for restocking.")

# %% [markdown]
# ## 6. Rising and declining products

# %%
trend_table = forecasts[["product", "trend", "trend_change_pct", "demand_30d"]].sort_values("trend_change_pct", ascending=False)
display(trend_table)

plot_df = trend_table.dropna(subset=["trend_change_pct"])
if len(plot_df):
    colors = ["seagreen" if v > 0 else "indianred" for v in plot_df["trend_change_pct"]]
    plt.figure(figsize=(8, 4))
    plt.barh(plot_df["product"], plot_df["trend_change_pct"], color=colors)
    plt.axvline(0, color="grey", linewidth=1)
    plt.title("Change in demand: last 28 days vs the 28 days before (%)")
    plt.tight_layout()
    show_plot()

# %% [markdown]
# ## 7. Look at one product
# The line shows daily demand (7-day average) and the dashed line shows the forecast daily rate for the next 30 days.

# %%
product = forecasts.iloc[0]["product"]      # change this to look at another product
cfg = Config()
daily = prepare_daily(requests, sales)[product]
demand = daily["sales"] + cfg.request_conversion * daily["requests"]
smooth = demand.rolling(7).mean()

rate = forecasts.loc[forecasts["product"] == product, "demand_30d"].iloc[0] / 30
future = pd.date_range(demand.index[-1], periods=31)

plt.figure(figsize=(11, 4))
plt.plot(smooth.index[-120:], smooth.iloc[-120:], label="daily demand (7-day average)")
plt.plot(future, [rate] * len(future), "--", color="darkorange", label="forecast, next 30 days")
plt.title(f"{product}: demand and forecast")
plt.ylabel("Units per day")
plt.legend()
plt.tight_layout()
show_plot()

# %% [markdown]
# ## 8. How sensitive are the results to the request weighting?
# The 30% request weighting is an assumption. This check shows how much the 30-day forecast moves if it is lower or higher. If your team has data on how many requests really turn into sales, use that number instead.

# %%
rows = []
for k in [0.1, 0.3, 0.5]:
    out = forecast_products(requests, sales, stock, config=Config(request_conversion=k))["forecasts"]
    rows.append(out.set_index("product")["demand_30d"].rename(f"weight {int(k*100)}%"))
sensitivity = pd.concat(rows, axis=1)
display(sensitivity)

# %% [markdown]
# ## 9. Limitations
# - Customer requests show interest and are not guaranteed sales. The 30% weighting is an assumption.
# - Products with less than about 60 days of history get a simple recent-rate forecast and low confidence.
# - The forecast does not know about promotions, price changes or holidays. Add them as features if you have them.
# - The reorder rule assumes a steady lead time and does not account for supplier minimum order quantities.
# - Results on the synthetic demo data show that the code works, not how accurate it will be on real data. Check the backtest table again with your own data.
#
# ## 10. Next steps
# - Run it on real requests, sales and stock data.
# - Add promotions and holidays as extra features.
# - Try Prophet or ARIMA for products with long histories.
# - Connect it to the GhostDemand app as a `/api/forecast` route.
