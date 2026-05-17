from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import build_supervised_frame, project_exogenous


FEATURE_COLUMNS = [
    "year",
    "year_index",
    "total_generation",
    "renewable_share",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
]


@dataclass
class ResourceModelArtifact:
    resource: str
    model_name: str
    model: object
    leaderboard: pd.DataFrame
    feature_columns: list[str]
    holdout_predictions: pd.DataFrame


class LastValueForecaster:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LastValueForecaster":
        self.last_value_ = float(y.iloc[-1])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.last_value_, dtype=float)


def _candidate_models(small_data: bool = False) -> dict[str, object]:
    if small_data:
        return {
            "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=4, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
        }

    return {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_best_model_for_resource(country_df: pd.DataFrame, resource: str) -> ResourceModelArtifact:
    supervised = build_supervised_frame(country_df, resource)

    if len(supervised) < 6:
        fallback_model = LastValueForecaster().fit(supervised[["year"]], supervised["target"])
        holdout = supervised[["year", "target"]].copy()
        holdout["prediction"] = fallback_model.predict(holdout[["year"]])
        leaderboard = pd.DataFrame(
            [
                {
                    "resource": resource,
                    "model": "LastValueForecaster",
                    "mae": float(mean_absolute_error(holdout["target"], holdout["prediction"])),
                    "rmse": float(np.sqrt(mean_squared_error(holdout["target"], holdout["prediction"]))),
                    "r2": float(r2_score(holdout["target"], holdout["prediction"])),
                }
            ]
        )
        return ResourceModelArtifact(
            resource=resource,
            model_name="LastValueForecaster",
            model=fallback_model,
            leaderboard=leaderboard,
            feature_columns=FEATURE_COLUMNS,
            holdout_predictions=holdout,
        )

    small_data = len(supervised) < 18
    n_test = max(2, int(round(len(supervised) * 0.2)))
    if len(supervised) - n_test < 3:
        n_test = max(1, len(supervised) - 3)

    train_df = supervised.iloc[:-n_test].copy()
    test_df = supervised.iloc[-n_test:].copy()

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    rows = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, model in _candidate_models(small_data=small_data).items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions_by_model[model_name] = pred
        metrics = _score(y_test.to_numpy(), pred)
        rows.append({"resource": resource, "model": model_name, **metrics})

    leaderboard = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_name = str(leaderboard.loc[0, "model"])

    best_model = _candidate_models(small_data=small_data)[best_name]
    best_model.fit(supervised[FEATURE_COLUMNS], supervised["target"])

    holdout = test_df[["year", "target"]].copy()
    holdout["prediction"] = predictions_by_model[best_name]

    return ResourceModelArtifact(
        resource=resource,
        model_name=best_name,
        model=best_model,
        leaderboard=leaderboard,
        feature_columns=FEATURE_COLUMNS,
        holdout_predictions=holdout,
    )


def forecast_resource(
    country_df: pd.DataFrame,
    artifact: ResourceModelArtifact,
    horizon: int,
) -> pd.DataFrame:
    exog_future = project_exogenous(country_df, horizon)

    history_vals = country_df[artifact.resource].astype(float).tolist()
    rows = []

    for _, exog in exog_future.iterrows():
        lag_1 = history_vals[-1]
        lag_2 = history_vals[-2] if len(history_vals) >= 2 else history_vals[-1]
        lag_3 = history_vals[-3] if len(history_vals) >= 3 else history_vals[-1]
        recent = history_vals[-3:] if len(history_vals) >= 3 else history_vals
        rolling_mean_3 = float(np.mean(recent))
        rolling_std_3 = float(np.std(recent))

        feature_row = pd.DataFrame(
            [
                {
                    "year": exog["year"],
                    "year_index": exog["year_index"],
                    "total_generation": exog["total_generation"],
                    "renewable_share": exog["renewable_share"],
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean_3,
                    "rolling_std_3": rolling_std_3,
                }
            ]
        )
        pred = float(artifact.model.predict(feature_row[artifact.feature_columns])[0])
        pred = max(0.0, pred)
        history_vals.append(pred)

        rows.append({"year": int(exog["year"]), "resource": artifact.resource, "prediction_twh": pred})

    return pd.DataFrame(rows)


def train_models(country_df: pd.DataFrame, resources: list[str]) -> tuple[dict[str, ResourceModelArtifact], pd.DataFrame]:
    artifacts: dict[str, ResourceModelArtifact] = {}
    leaderboard_frames = []

    for resource in resources:
        artifact = train_best_model_for_resource(country_df, resource)
        artifacts[resource] = artifact
        leaderboard_frames.append(artifact.leaderboard)

    full_leaderboard = pd.concat(leaderboard_frames, ignore_index=True)
    return artifacts, full_leaderboard
