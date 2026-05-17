# Energy Resource Analytics and Forecasting

This project uses a real web dataset from Our World in Data and builds a full analytics workflow:

- Data ingestion from web
- EDA with exported plots
- Feature engineering (lag and rolling features)
- Model selection across multiple ML models
- Forecasting for multiple energy resources
- Streamlit web app for predictions and analysis plots

## Web Dataset Used

- Source: https://ourworldindata.org/grapher/electricity-prod-source-stacked.csv
- Energy resources available include Solar, Wind, Hydropower, Bioenergy, Other renewables, Nuclear, Coal, Gas, Oil.

## Project Outputs

- Raw dataset: data/raw/owid_electricity_source_data.csv
- Processed country dataset: data/processed/country_generation_features.csv
- Model leaderboard: outputs/tableau/model_leaderboard.csv
- Forecast results: outputs/tableau/resource_forecasts.csv
- Model bundle: outputs/models/model_bundle.joblib
- EDA plots:
	- outputs/plots/resource_trends.png
	- outputs/plots/correlation_heatmap.png
	- outputs/plots/resource_share_stacked.png
	- outputs/plots/eda_summary_stats.csv

## Setup

```powershell
cd "c:\RE Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Analytics Pipeline

```powershell
python src/main.py --country India --min-year 1990 --horizon 8
```

Optional custom resources:

```powershell
python src/main.py --country India --resources "Solar,Wind,Hydropower,Nuclear,Coal,Gas"
```

## Run Web App (Streamlit)

```powershell
streamlit run app_streamlit.py
```

The app includes:

- Analysis tab with interactive trend/correlation/distribution plots
- Model performance tab with per-resource model comparison
- Prediction tab with future generation forecasts for selected resources
