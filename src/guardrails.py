from typing import Tuple, Dict
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0: return float("nan")
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100.0

def rolling_backtest(y: pd.Series, horizon: int = 1, start: int = 24):
    y = y.copy(); y.index = pd.to_datetime(y.index)
    rows = []
    for i in range(start, len(y)-horizon+1):
        train, test = y.iloc[:i], y.iloc[i:i+horizon]
        fc = float(train.iloc[-1])  # naive; swap with your model
        for ts, actual in test.items():
            rows.append({"date": ts, "actual": float(actual), "forecast": fc})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mape"] = np.abs((df["actual"]-df["forecast"]) / df["actual"].replace(0, np.nan))*100.0
    return df

def drift_alert(mape_series: pd.Series, recent_k=3, baseline_window=6, sigma=2.0):
    s = mape_series.dropna()
    if len(s) < recent_k + baseline_window:
        return False, {"reason": "not_enough_data"}
    recent_mean = s.iloc[-recent_k:].mean()
    baseline = s.iloc[-(recent_k+baseline_window):-recent_k]
    base_mean, base_std = baseline.mean(), baseline.std(ddof=1) if len(baseline)>1 else 0.0
    threshold = base_mean + sigma*base_std
    return (recent_mean > threshold), {"recent_mean": float(recent_mean), "baseline_mean": float(base_mean), "baseline_std": float(base_std), "threshold": float(threshold)}

def cone_plot(df: pd.DataFrame, path: str):
    import matplotlib.dates as mdates
    x = mdates.date2num(pd.to_datetime(df["date"]).dt.to_pydatetime())
    actual = df["actual"].astype(float).values
    forecast = df["forecast"].astype(float).values
    lower = df["lower"].astype(float).values
    upper = df["upper"].astype(float).values
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.plot_date(x, actual, "-", label="Actual")
    ax.plot_date(x, forecast, "-", label="Forecast")
    ax.fill_between(x, lower, upper, alpha=0.2, label="Cone")
    ax.set_title("Forecast vs Actuals (Cone View)"); ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); fig.autofmt_xdate()
    ax.legend(loc="best"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
