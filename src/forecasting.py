from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

try:
    from prophet import Prophet
except Exception:
    Prophet = None


@dataclass
class ForecastResult:
    metrics: pd.DataFrame
    forecasts: pd.DataFrame


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(y_true == 0, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def _metric_row(model: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(_mape(y_true, y_pred), 4),
    }


def run_forecasts(df: pd.DataFrame, horizon_days: int = 30) -> ForecastResult:
    ts = df[["date", "generation_mwh"]].copy().sort_values("date")
    ts = ts.reset_index(drop=True)

    if len(ts) <= horizon_days + 14:
        raise ValueError("Not enough rows for train/test split. Increase date range.")

    train = ts.iloc[:-horizon_days]
    test = ts.iloc[-horizon_days:]

    metrics = []
    output = test[["date", "generation_mwh"]].rename(columns={"generation_mwh": "actual"})

    # ARIMA baseline
    try:
        arima = ARIMA(train["generation_mwh"], order=(2, 1, 2)).fit()
    except Exception:
        arima = ARIMA(train["generation_mwh"], order=(1, 1, 1)).fit()

    arima_pred = arima.forecast(steps=horizon_days).to_numpy()
    output["arima_forecast"] = arima_pred
    metrics.append(_metric_row("ARIMA", test["generation_mwh"].to_numpy(), arima_pred))

    # Prophet (optional, skips gracefully if package unavailable)
    if Prophet is not None:
        prophet_train = train.rename(columns={"date": "ds", "generation_mwh": "y"})
        prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        prophet_model.fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=horizon_days, freq="D", include_history=False)
        prophet_fcst = prophet_model.predict(future)[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prophet_forecast"})
        output = output.merge(prophet_fcst, on="date", how="left")
        metrics.append(
            _metric_row(
                "Prophet",
                test["generation_mwh"].to_numpy(),
                output["prophet_forecast"].to_numpy(),
            )
        )
    else:
        output["prophet_forecast"] = np.nan
        metrics.append({"model": "Prophet", "mae": np.nan, "rmse": np.nan, "mape": np.nan})

    metrics_df = pd.DataFrame(metrics)
    return ForecastResult(metrics=metrics_df, forecasts=output)
