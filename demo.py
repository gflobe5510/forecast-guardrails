import pandas as pd
from src.guardrails import rolling_backtest, drift_alert, cone_plot

df = pd.read_csv("data/sample_timeseries.csv", parse_dates=["date"])
series = df.set_index("date")["actual"].dropna()

bt = rolling_backtest(series, horizon=1, start=24)
alert, stats = drift_alert(bt["mape"], recent_k=3, baseline_window=6, sigma=2.0)
print("Drift alert:", alert); print("Stats:", stats)

cone_df = df.dropna(subset=["lower","upper","forecast"], how="any")
cone_plot(cone_df, "assets/forecast_cone.png")
print("Saved chart to assets/forecast_cone.png")
