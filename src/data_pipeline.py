from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import DATA_URL, REQUEST_HEADERS, RENEWABLE_COLUMNS


def download_energy_dataset(raw_path: Path) -> pd.DataFrame:
    """Download OWID electricity generation by source dataset."""
    response = requests.get(DATA_URL, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    return df


def prepare_country_dataset(
    df: pd.DataFrame,
    country: str,
    min_year: int,
    resources: Iterable[str],
) -> pd.DataFrame:
    resources = [r for r in resources if r in df.columns]
    if not resources:
        raise ValueError("No valid resource columns were selected.")

    country_df = df.loc[df["Entity"] == country, ["Entity", "Code", "Year", *resources]].copy()
    country_df = country_df.rename(columns={"Entity": "country", "Code": "code", "Year": "year"})
    country_df = country_df[country_df["year"] >= min_year].sort_values("year").reset_index(drop=True)

    if country_df.empty:
        raise ValueError(f"No rows found for country '{country}' with min_year={min_year}.")

    # Forward-fill sparse values and then remaining missing as 0 for cleaner annual panels.
    for col in resources:
        country_df[col] = country_df[col].astype(float)
    country_df[resources] = country_df[resources].ffill().fillna(0.0)

    country_df["total_generation"] = country_df[resources].sum(axis=1)
    renewable_cols = [c for c in RENEWABLE_COLUMNS if c in resources]
    country_df["renewable_generation"] = country_df[renewable_cols].sum(axis=1) if renewable_cols else 0.0
    country_df["renewable_share"] = np.where(
        country_df["total_generation"] > 0,
        country_df["renewable_generation"] / country_df["total_generation"],
        0.0,
    )
    country_df["total_growth_pct"] = country_df["total_generation"].pct_change().fillna(0.0)
    return country_df


def build_supervised_frame(country_df: pd.DataFrame, resource: str) -> pd.DataFrame:
    data = country_df[["year", "total_generation", "renewable_share", resource]].copy()
    data = data.rename(columns={resource: "target"})

    data["year_index"] = data["year"] - int(data["year"].min())
    data["lag_1"] = data["target"].shift(1)
    data["lag_2"] = data["target"].shift(2)
    data["lag_3"] = data["target"].shift(3)
    data["rolling_mean_3"] = data["target"].shift(1).rolling(3).mean()
    data["rolling_std_3"] = data["target"].shift(1).rolling(3).std().fillna(0.0)

    data = data.dropna().reset_index(drop=True)
    return data


def project_exogenous(country_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_year = int(country_df["year"].max())
    future_years = np.arange(last_year + 1, last_year + horizon + 1)

    total = country_df["total_generation"].to_numpy()
    share = country_df["renewable_share"].to_numpy()

    total_slope = float(np.mean(np.diff(total))) if len(total) > 1 else 0.0
    share_slope = float(np.mean(np.diff(share))) if len(share) > 1 else 0.0

    total_proj = [float(total[-1] + (i + 1) * total_slope) for i in range(horizon)]
    share_proj = [float(np.clip(share[-1] + (i + 1) * share_slope, 0.0, 1.0)) for i in range(horizon)]

    return pd.DataFrame(
        {
            "year": future_years,
            "year_index": future_years - int(country_df["year"].min()),
            "total_generation": total_proj,
            "renewable_share": share_proj,
        }
    )
