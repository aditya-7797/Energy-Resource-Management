from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_eda(country_df: pd.DataFrame, resources: list[str], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}

    summary = country_df[["year", *resources, "total_generation", "renewable_share"]].describe().T
    summary_path = output_dir / "eda_summary_stats.csv"
    summary.to_csv(summary_path)
    created["summary_stats"] = summary_path

    # Trend plot by resource
    trend_path = output_dir / "resource_trends.png"
    plt.figure(figsize=(12, 6))
    for resource in resources:
        plt.plot(country_df["year"], country_df[resource], marker="o", linewidth=1.8, label=resource)
    plt.title("Historical Electricity Generation by Resource")
    plt.xlabel("Year")
    plt.ylabel("Generation (TWh)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(trend_path, dpi=160)
    plt.close()
    created["resource_trends"] = trend_path

    corr_cols = resources + ["total_generation", "renewable_share", "total_growth_pct"]
    corr = country_df[corr_cols].corr(numeric_only=True)
    corr_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=70, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=160)
    plt.close()
    created["correlation_heatmap"] = corr_path

    share_df = country_df[["year", *resources]].copy()
    row_sums = share_df[resources].sum(axis=1)
    share_pct = share_df[resources].div(row_sums.where(row_sums != 0, 1), axis=0) * 100

    share_path = output_dir / "resource_share_stacked.png"
    plt.figure(figsize=(12, 6))
    plt.stackplot(country_df["year"], [share_pct[c] for c in resources], labels=resources, alpha=0.85)
    plt.title("Resource Share of Generation (%)")
    plt.xlabel("Year")
    plt.ylabel("Share (%)")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(share_path, dpi=160)
    plt.close()
    created["share_stacked"] = share_path

    return created
