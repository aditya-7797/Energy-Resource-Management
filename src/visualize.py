from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(forecast_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df["date"], forecast_df["actual"], label="Actual", linewidth=2)
    plt.plot(forecast_df["date"], forecast_df["arima_forecast"], label="ARIMA", linestyle="--")

    if forecast_df["prophet_forecast"].notna().any():
        plt.plot(forecast_df["date"], forecast_df["prophet_forecast"], label="Prophet", linestyle=":")

    plt.title("Renewable Energy Generation Forecast")
    plt.xlabel("Date")
    plt.ylabel("Generation (MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
