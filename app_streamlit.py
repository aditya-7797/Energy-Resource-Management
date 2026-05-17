from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_COUNTRY, DEFAULT_HORIZON, DEFAULT_MIN_YEAR, RESOURCE_COLUMNS
from src.data_pipeline import download_energy_dataset, prepare_country_dataset
from src.modeling import forecast_resource, train_models


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "owid_electricity_source_data.csv"


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return download_energy_dataset(RAW_PATH)


def run_analysis(country: str, min_year: int, resources: list[str], horizon: int) -> dict:
    raw_df = load_raw_data()
    country_df = prepare_country_dataset(raw_df, country, min_year, resources)
    artifacts, leaderboard = train_models(country_df, resources)

    forecasts = []
    for resource in resources:
        forecasts.append(forecast_resource(country_df, artifacts[resource], horizon))
    forecast_df = pd.concat(forecasts, ignore_index=True)

    return {
        "country_df": country_df,
        "artifacts": artifacts,
        "leaderboard": leaderboard,
        "forecast_df": forecast_df,
    }


st.set_page_config(page_title="Energy Resource Analytics", layout="wide")
st.title("Energy Resource Analytics and Forecasting")
st.caption("Real web dataset + EDA + feature engineering + ML model selection")

raw_data = load_raw_data()
available_countries = sorted(raw_data["Entity"].dropna().unique().tolist())
default_country_index = available_countries.index(DEFAULT_COUNTRY) if DEFAULT_COUNTRY in available_countries else 0

with st.sidebar:
    st.header("Controls")
    country = st.selectbox("Country", options=available_countries, index=default_country_index)
    min_year = st.slider("Minimum Year", min_value=1965, max_value=2022, value=DEFAULT_MIN_YEAR, step=1)
    horizon = st.slider("Forecast Horizon (Years)", min_value=3, max_value=15, value=DEFAULT_HORIZON, step=1)
    resources = st.multiselect(
        "Resources to Model",
        options=RESOURCE_COLUMNS,
        default=["Solar", "Wind", "Hydropower", "Nuclear", "Coal", "Gas"],
    )
    run_clicked = st.button("Run Analysis")

if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked:
    if len(resources) < 2:
        st.error("Please choose at least two resources for meaningful analytics.")
    else:
        with st.spinner("Running EDA, feature engineering, model selection, and forecasting..."):
            st.session_state.result = run_analysis(country, min_year, resources, horizon)

if st.session_state.result is None:
    st.info("Select parameters and click 'Run Analysis' to generate analytics and predictions.")
else:
    result = st.session_state.result
    country_df = result["country_df"]
    leaderboard = result["leaderboard"]
    forecast_df = result["forecast_df"]

    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "EDA Plots", "Model Selection", "Predictions"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(country_df)}")
        c2.metric("Last Historical Year", f"{int(country_df['year'].max())}")
        c3.metric("Latest Total Generation (TWh)", f"{country_df['total_generation'].iloc[-1]:.2f}")
        st.dataframe(country_df.tail(12), use_container_width=True)

    with tab2:
        trend_long = country_df.melt(id_vars=["year"], value_vars=resources, var_name="resource", value_name="generation_twh")
        trend_fig = px.line(
            trend_long,
            x="year",
            y="generation_twh",
            color="resource",
            markers=True,
            title="Historical Generation Trend by Resource",
        )
        st.plotly_chart(trend_fig, use_container_width=True)

        corr_cols = resources + ["total_generation", "renewable_share", "total_growth_pct"]
        corr = country_df[corr_cols].corr(numeric_only=True)
        corr_fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        corr_fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(corr_fig, use_container_width=True)

        box_fig = px.box(
            trend_long,
            x="resource",
            y="generation_twh",
            color="resource",
            title="Resource Generation Distribution",
        )
        st.plotly_chart(box_fig, use_container_width=True)

    with tab3:
        st.subheader("Model Leaderboard")
        st.dataframe(leaderboard.sort_values(["resource", "rmse"]), use_container_width=True)
        best_models = leaderboard.sort_values("rmse").groupby("resource", as_index=False).first()
        bar_fig = px.bar(best_models, x="resource", y="rmse", color="model", title="Best Model by Resource (Lower RMSE is Better)")
        st.plotly_chart(bar_fig, use_container_width=True)

    with tab4:
        st.subheader("Forecasted Generation by Resource")
        st.dataframe(forecast_df, use_container_width=True)

        history_long = country_df.melt(id_vars=["year"], value_vars=resources, var_name="resource", value_name="generation_twh")
        forecast_plot_df = forecast_df.rename(columns={"prediction_twh": "generation_twh"}).copy()
        forecast_plot_df["series"] = "Forecast"
        history_long["series"] = "Historical"

        combined_plot = pd.concat(
            [
                history_long[["year", "resource", "generation_twh", "series"]],
                forecast_plot_df[["year", "resource", "generation_twh", "series"]],
            ],
            ignore_index=True,
        )
        pred_fig = px.line(
            combined_plot,
            x="year",
            y="generation_twh",
            color="resource",
            line_dash="series",
            markers=True,
            title="Historical vs Forecast Generation",
        )
        st.plotly_chart(pred_fig, use_container_width=True)
