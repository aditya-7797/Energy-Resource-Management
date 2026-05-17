from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_COUNTRY, DEFAULT_HORIZON, DEFAULT_MIN_YEAR, RESOURCE_COLUMNS
from .data_pipeline import download_energy_dataset, prepare_country_dataset
from .eda import run_eda
from .modeling import forecast_resource, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy resource analytics and forecasting pipeline")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, help="Country/Entity name from dataset")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help="Minimum year to include")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast years ahead")
    parser.add_argument(
        "--resources",
        default=",".join(RESOURCE_COLUMNS),
        help="Comma-separated resource columns to model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_resources = [r.strip() for r in args.resources.split(",") if r.strip()]

    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "owid_electricity_source_data.csv"
    processed_path = root / "data" / "processed" / "country_generation_features.csv"
    metrics_path = root / "outputs" / "tableau" / "model_leaderboard.csv"
    forecast_path = root / "outputs" / "tableau" / "resource_forecasts.csv"
    eda_dir = root / "outputs" / "plots"
    model_bundle_path = root / "outputs" / "models" / "model_bundle.joblib"

    model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = download_energy_dataset(raw_path)
    country_df = prepare_country_dataset(raw_df, args.country, args.min_year, selected_resources)
    country_df.to_csv(processed_path, index=False)

    eda_outputs = run_eda(country_df, selected_resources, eda_dir)

    artifacts, leaderboard = train_models(country_df, selected_resources)
    leaderboard.to_csv(metrics_path, index=False)

    forecasts = []
    for resource in selected_resources:
        forecasts.append(forecast_resource(country_df, artifacts[resource], args.horizon))
    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.to_csv(forecast_path, index=False)

    model_bundle = {
        "country": args.country,
        "min_year": args.min_year,
        "resources": selected_resources,
        "country_df": country_df,
        "artifacts": artifacts,
        "leaderboard": leaderboard,
    }
    joblib.dump(model_bundle, model_bundle_path)

    print("Pipeline complete.")
    print(f"Raw dataset: {raw_path}")
    print(f"Processed features: {processed_path}")
    print(f"Model leaderboard: {metrics_path}")
    print(f"Forecast output: {forecast_path}")
    print(f"Model bundle: {model_bundle_path}")
    print("EDA plot files:")
    for name, path in eda_outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
