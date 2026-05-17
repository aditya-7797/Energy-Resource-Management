# Energy Resource Analytics and Forecasting Project Documentation

## 1. Project Title

**Energy Resource Analytics and Forecasting using Real-World Web Data**

This project is a complete data analytics and forecasting workflow built on electricity generation data from Our World in Data. It combines data ingestion, preprocessing, exploratory data analysis, feature engineering, model comparison, forecasting, and a Streamlit user interface into one end-to-end pipeline.

---

## 2. Introduction

Electricity generation is one of the most important indicators of economic activity, infrastructure development, industrial growth, and energy transition. Governments, energy companies, researchers, and policy planners constantly need to understand how electricity generation from different sources changes over time. They also need to estimate future generation patterns to support planning, investment, sustainability analysis, and risk management.

This project addresses that need by analyzing historical electricity generation data by source and forecasting how resource-wise generation is likely to evolve in the future. Instead of using synthetic or toy data, it uses a real public dataset published by Our World in Data, which makes the results more meaningful and closer to practical analytical work.

The project is not only a forecasting notebook or a static report. It is an engineering-style analytics system that includes:

- Automated data download from the web
- Cleaning and transformation of the raw dataset
- Country-level resource analysis
- Derived features such as total generation, renewable share, growth rate, lag variables, and rolling statistics
- Model benchmarking across multiple algorithms
- Resource-wise forecasting for selected future years
- Exported charts, summary tables, and model artifacts
- A Streamlit application for interactive analysis

---

## 3. Problem Statement

Energy systems are complex and dynamic. Electricity generation by source changes because of policy decisions, fuel prices, technological advances, climate conditions, infrastructure investment, and demand patterns. Decision makers often face the following problems:

- Historical generation data is spread across sources and time periods.
- It is difficult to understand which energy resources are growing or declining.
- Renewable adoption trends are not always obvious from raw tabular data.
- There is no simple way to compare the predictive performance of different models for each energy source.
- Future generation estimates are needed for planning but are not readily available.

This project solves these problems by building a reproducible analytics pipeline that turns raw electricity source data into interpretable trends and forecasts.

---

## 4. Project Objectives

The main objectives of the project are:

1. To download a real electricity generation dataset from Our World in Data.
2. To filter the dataset for a selected country and time range.
3. To prepare a clean annual resource-level dataset for analysis.
4. To perform exploratory data analysis and generate plots automatically.
5. To engineer forecasting features such as lag values and rolling statistics.
6. To compare multiple machine learning models for each energy resource.
7. To forecast future generation for selected resources.
8. To store outputs in reusable files for reporting and dashboarding.
9. To provide an interactive Streamlit interface for users.

---

## 5. Scope of the Project

The project focuses on country-level electricity generation analysis and forecasting. It is designed for a user who wants to study one country at a time and choose one or more energy resources for modeling.

### In Scope

- Electricity generation by source analysis
- Country-specific filtering
- Renewable and non-renewable resource comparison
- Statistical visualization
- Supervised learning for resource forecasting
- Streamlit-based interactive analysis

### Out of Scope

- Hourly or sub-daily forecasting
- Regional grid optimization
- Power market price prediction
- Deep learning sequence models such as LSTM or transformers
- External macroeconomic or weather-driven causal modeling beyond the available features

---

## 6. System Overview

The project follows a structured analytics architecture:

1. Raw data is downloaded from the OWID CSV endpoint.
2. Data is filtered for a chosen country and year range.
3. Derived variables are computed, such as total generation and renewable share.
4. EDA outputs are generated and saved to disk.
5. Supervised learning frames are created for each resource.
6. Multiple models are trained and evaluated.
7. The best model for each resource is used for future forecasting.
8. Results are exported as CSV files and a serialized model bundle.
9. A Streamlit app provides interactive analysis and forecasts.

---

## 7. Real Dataset Description

### Data Source

The project uses the following real web dataset:

- **Source**: Our World in Data electricity generation by source dataset
- **URL**: https://ourworldindata.org/grapher/electricity-prod-source-stacked.csv

### Resource Categories

The project works with the following electricity generation sources:

- Solar
- Wind
- Hydropower
- Bioenergy
- Other renewables
- Nuclear
- Coal
- Gas
- Oil

### Why This Dataset Is Valuable

- It is public and widely used in energy analytics.
- It contains long-term historical data.
- It supports cross-country energy analysis.
- It allows renewable transition studies.
- It is suitable for both descriptive analytics and forecasting.

---

## 8. Software Requirements Specification (SRS)

### 8.1 Purpose

The purpose of the system is to provide a complete analytics and forecasting solution for electricity generation by source using a real-world dataset.

### 8.2 Intended Users

- Students working on data analytics or energy forecasting projects
- Researchers studying energy transitions
- Analysts preparing dashboards and reports
- Policy and planning teams evaluating generation trends

### 8.3 Functional Requirements

#### FR1: Data Download

The system shall download electricity generation data from the OWID CSV endpoint and store it locally.

#### FR2: Country Filtering

The system shall filter the dataset for a selected country/entity.

#### FR3: Year Filtering

The system shall allow the user to define a minimum year for analysis.

#### FR4: Resource Selection

The system shall allow the user to select which electricity resources to model.

#### FR5: Feature Engineering

The system shall generate derived fields such as total generation, renewable generation, renewable share, total growth percentage, lag features, and rolling statistics.

#### FR6: EDA Generation

The system shall generate summary statistics and visual plots.

#### FR7: Model Training

The system shall train and compare multiple predictive models for each selected resource.

#### FR8: Forecasting

The system shall forecast future generation for the selected horizon.

#### FR9: Result Export

The system shall save processed data, forecasts, leaderboard metrics, plots, and model bundles to disk.

#### FR10: Interactive Visualization

The system shall expose the analysis in a Streamlit application.

### 8.4 Non-Functional Requirements

#### NFR1: Usability

The UI must be easy to use and suitable for non-technical users.

#### NFR2: Reproducibility

Running the pipeline with the same parameters should generate consistent outputs.

#### NFR3: Maintainability

The code should remain modular and easy to extend.

#### NFR4: Performance

The pipeline should execute within a practical time for a single-country annual dataset.

#### NFR5: Reliability

The system should handle missing values and low-data cases gracefully.

#### NFR6: Portability

The project should run locally in a Python virtual environment.

### 8.5 Constraints

- The project depends on the availability of the OWID dataset URL.
- Forecast quality is limited by annual resolution and feature simplicity.
- Some advanced forecasting packages, such as Prophet, are optional.
- The project is designed for one-country analysis rather than simultaneous global optimization.

### 8.6 Assumptions

- The dataset contains enough historical observations for meaningful modeling.
- Annual resource values are sufficiently stable to support lag-based forecasting.
- Missing values can be forward-filled or treated as zero where appropriate.
- The future trend of total generation and renewable share can be approximated from historical movement.

### 8.7 Acceptance Criteria

The project is considered successful if it can:

- Download the dataset automatically.
- Produce country-specific processed data.
- Generate EDA plots and summary statistics.
- Train and rank models for selected resources.
- Produce forecast tables for the requested horizon.
- Launch the Streamlit app without errors.

---

## 9. Literature Survey

The following survey summarizes representative research and foundational ideas relevant to energy forecasting, renewable analytics, and time-series modeling. These references motivate the methods used in the project.

| No. | Paper / Work | Contribution | Relevance to This Project |
|---|---|---|---|
| 1 | Hyndman and Khandakar, 2008, "Automatic Time Series Forecasting: The forecast Package for R" | Introduced automated ARIMA model selection and practical forecasting workflows | Supports the idea of using classical forecasting baselines for time series prediction |
| 2 | Taylor and McSharry, 2007, "Short-Term Load Forecasting Methods: An Evaluation Based on European Data" | Compared statistical forecasting approaches for electricity-related time series | Reinforces the importance of benchmarking multiple models |
| 3 | Weron, 2014, "Electricity Price Forecasting: A Review of the State-of-the-Art" | Surveyed forecasting methodologies in the electricity domain | Shows that energy forecasting benefits from strong evaluation and feature engineering |
| 4 | Mellit and Kalogirou, 2008, "Artificial Intelligence Techniques for Photovoltaic Applications" | Reviewed AI methods for renewable energy prediction | Motivates machine learning in renewable generation analysis |
| 5 | Safari and Gasore, 2010, "A Statistical Investigation of Wind Power Forecasting" | Discussed challenges in forecasting wind power | Highlights variability and the need for robust models |
| 6 | Deb et al., 2017, works on renewable energy forecasting using machine learning | Demonstrated the value of ML models in forecasting renewable generation | Supports use of tree-based and linear models for resource forecasting |
| 7 | Hippert, Pedreira, and Souza, 2001, "Neural Networks for Short-Term Load Forecasting: A Review and Evaluation" | Reviewed neural network applications in forecasting | Establishes that forecasting accuracy improves when models are compared systematically |
| 8 | Fan, 2018, studies on renewable energy forecasting and grid integration | Discussed forecasting as a key enabler for clean energy integration | Connects directly to renewable share estimation in this project |
| 9 | Ahmad and Chen, 2018, review studies on energy consumption and forecasting | Examined machine learning in energy prediction tasks | Supports the use of machine learning for structured energy data |
| 10 | Recent OWID and global energy transition analyses using public datasets | Demonstrated practical use of open energy data for analytics dashboards | Confirms the value of building transparent, reproducible data pipelines |

### Literature Survey Observations

- Classical models such as ARIMA remain important as baselines.
- Machine learning models often improve flexibility when exogenous and lagged features are present.
- Renewable generation forecasting is influenced by trend, seasonality, policy shifts, and structural changes.
- Public datasets like OWID are well suited for educational and analytical prototypes.
- Model comparison is essential because no single method is best for every resource.

---

## 10. Proposed Solution

The proposed solution is a modular analytics pipeline that transforms raw generation data into forecasts and visual insights.

### Core Idea

The system takes historical electricity generation data, isolates a selected country, engineers time-based features for each resource, trains multiple models, compares them using holdout metrics, and then predicts future values for each selected resource.

### Why This Solution Works

- It uses real data instead of synthetic examples.
- It combines analytical insight with predictive modeling.
- It stays simple enough to be explainable.
- It is modular, so each part can be tested independently.
- It produces both machine-readable outputs and human-readable visuals.

### Design Philosophy

The solution is intentionally practical. Rather than using a complex deep learning model that may be harder to interpret, it uses:

- Strong data preprocessing
- Clear feature engineering
- Multiple classical ML regressors
- Simple recursive forecasting logic
- Optional Prophet support for an additional forecasting baseline

---

## 11. Technology Stack

### Programming Language

- Python 3

### Core Libraries

- Pandas for data handling
- NumPy for numerical operations
- Requests for web download
- Matplotlib for static visualizations
- Plotly for interactive charts
- Scikit-learn for machine learning models and evaluation
- Statsmodels for ARIMA forecasting
- Joblib for model serialization
- Streamlit for the web application

### Optional Library

- Prophet, if installed, for an additional forecasting baseline

### Development and Execution Environment

- Windows local environment
- Python virtual environment stored in `.venv`
- Command-line execution through PowerShell

### File Formats Used

- CSV for raw data, processed data, metrics, forecasts, and summary statistics
- PNG for plots
- Joblib for serialized model bundles

---

## 12. Project Structure

The repository is organized into logical folders:

- `app_streamlit.py` - Streamlit UI entry point
- `src/config.py` - Constants and configuration values
- `src/data_pipeline.py` - Download and preprocessing logic
- `src/eda.py` - EDA generation utilities
- `src/modeling.py` - Model training and forecasting logic
- `src/forecasting.py` - Time-series benchmark forecasting helpers
- `src/main.py` - Main CLI pipeline runner
- `outputs/` - Generated reports, plots, metrics, and model artifacts
- `data/raw/` - Raw downloaded dataset
- `data/processed/` - Cleaned and feature-engineered dataset

---

## 13. Detailed Methodology

### 13.1 Data Acquisition

The pipeline begins by downloading the electricity generation dataset directly from the OWID URL. The raw CSV is saved to the local project directory so the pipeline can be rerun and audited later.

### 13.2 Data Filtering

After download, the dataset is filtered by:

- Selected country/entity
- Minimum year chosen by the user
- Selected resource columns

This ensures that only relevant rows are processed.

### 13.3 Data Cleaning

The country-level dataset may contain missing values. The pipeline addresses this by:

- Converting resource columns to float
- Forward-filling sparse values
- Filling remaining missing values with zero

This is appropriate for annual panels where occasional gaps may occur.

### 13.4 Feature Engineering

The project derives several analytical variables:

- `total_generation`: sum of selected resources
- `renewable_generation`: sum of renewable resource columns present in the selection
- `renewable_share`: renewable generation divided by total generation
- `total_growth_pct`: percentage change in total generation year over year

For forecasting, the supervised learning dataset adds:

- `year_index`: year offset from the first observation
- `lag_1`: previous year target value
- `lag_2`: two years ago target value
- `lag_3`: three years ago target value
- `rolling_mean_3`: mean of the previous three target values
- `rolling_std_3`: standard deviation of the previous three target values

These features allow the model to learn trend and recent-history behavior.

### 13.5 Exploratory Data Analysis

The EDA step generates:

- Summary statistics table
- Historical trend plot by resource
- Correlation heatmap
- Stacked resource share plot

This helps users understand:

- How each source changes over time
- Which sources dominate the generation mix
- Which features move together
- Whether renewable sources are increasing in share

### 13.6 Model Training Strategy

The project trains multiple candidate regressors for each resource and compares them using holdout metrics.

Candidate models include:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

The exact candidate set may vary depending on the amount of data available.

### 13.7 Model Selection Strategy

The training process uses a time-aware split:

- The final portion of the supervised dataset is reserved for testing.
- Models are trained on the earlier portion.
- Predictions are made on the holdout set.
- Metrics such as MAE, RMSE, and R2 are computed.

The best model is selected by lowest RMSE.

### 13.8 Forecasting Strategy

Forecasting is done recursively. For each future year:

1. Exogenous future values are projected from historical trends.
2. Lag features are created from the latest known or predicted values.
3. The trained best model predicts the next value.
4. The predicted value is added back into the history.
5. The process repeats for the next horizon year.

This recursive logic allows the model to forecast multiple years ahead using its own outputs as inputs.

### 13.9 Forecast for Exogenous Variables

The system projects total generation and renewable share forward using a simple slope-based trend approximation derived from historical differences. This provides the model with a future context for features that cannot be directly known in advance.

### 13.10 Optional Time-Series Baseline

The separate forecasting utility can also evaluate:

- ARIMA as a classical baseline
- Prophet, if available in the environment

These models provide a traditional benchmark for comparison.

---

## 14. Execution Plan

The execution plan describes how the project is run end to end.

### Phase 1: Environment Setup

- Create a Python virtual environment
- Activate the environment
- Install dependencies from `requirements.txt`

### Phase 2: Data Download

- Download the OWID electricity generation dataset
- Save it under `data/raw/`

### Phase 3: Data Preparation

- Select country and year range
- Choose the resources to model
- Clean missing values
- Compute derived fields
- Save processed data to `data/processed/`

### Phase 4: EDA Generation

- Calculate descriptive statistics
- Create trend plots
- Create a correlation heatmap
- Create a stacked share plot
- Save all outputs to `outputs/plots/`

### Phase 5: Supervised Dataset Creation

- Build resource-specific modeling tables
- Add lag and rolling features
- Remove rows with missing engineered values

### Phase 6: Model Training and Comparison

- Split the data into train and test sections
- Train several candidate models
- Evaluate them using MAE, RMSE, and R2
- Rank them in a leaderboard

### Phase 7: Forecast Generation

- Use the selected model for each resource
- Forecast forward for the requested horizon
- Concatenate all resource forecasts into a single table

### Phase 8: Artifact Serialization

- Save the leaderboard as CSV
- Save the forecast output as CSV
- Save the model bundle using Joblib

### Phase 9: Streamlit Deployment

- Launch the app using Streamlit
- Allow the user to pick country, year, horizon, and resources
- Display tables and interactive charts

---

## 15. Module-by-Module Technical Details

### 15.1 `src/config.py`

This module defines the global constants used across the project, including:

- Dataset URL
- HTTP request headers
- List of resource columns
- Renewable resource subset
- Default country
- Default minimum year
- Default forecast horizon
- Minimum rows required for training

### 15.2 `src/data_pipeline.py`

This module handles:

- Dataset download
- Country filtering
- Missing-value handling
- Derived feature creation
- Supervised frame generation
- Future exogenous projection

### 15.3 `src/eda.py`

This module generates:

- Summary CSV statistics
- Resource trend plot
- Correlation heatmap
- Stacked resource-share plot

### 15.4 `src/modeling.py`

This module performs:

- Candidate model creation
- Model scoring
- Resource-specific training
- Best-model selection
- Recursive forecasting
- Multi-resource training orchestration

### 15.5 `src/forecasting.py`

This module provides a separate time-series forecasting utility using:

- ARIMA
- Prophet, when available

It is useful as a classical benchmarking layer.

### 15.6 `src/main.py`

This is the CLI entry point. It coordinates:

- Argument parsing
- Data download and preparation
- EDA execution
- Model training
- Forecast generation
- Artifact saving

### 15.7 `app_streamlit.py`

This is the user-facing dashboard. It provides:

- Sidebar controls
- Data overview tab
- EDA plots tab
- Model selection tab
- Forecast predictions tab

---

## 16. Output Artifacts

The pipeline writes several outputs to disk.

### Raw Data

- `data/raw/owid_electricity_source_data.csv`

### Processed Data

- `data/processed/country_generation_features.csv`

### EDA Outputs

- `outputs/plots/eda_summary_stats.csv`
- `outputs/plots/resource_trends.png`
- `outputs/plots/correlation_heatmap.png`
- `outputs/plots/resource_share_stacked.png`

### Model Evaluation Outputs

- `outputs/tableau/model_leaderboard.csv`

### Forecast Outputs

- `outputs/tableau/resource_forecasts.csv`

### Serialized Model Bundle

- `outputs/models/model_bundle.joblib`

---

## 17. Technical Flow of the Application

### CLI Flow

1. User runs `python src/main.py` with arguments.
2. The script downloads the dataset.
3. It filters and prepares the country data.
4. It creates EDA outputs.
5. It trains models for each selected resource.
6. It forecasts future values.
7. It saves all artifacts.

### Streamlit Flow

1. User opens the Streamlit app.
2. The app loads the raw dataset.
3. The user chooses country, year, horizon, and resources.
4. The app runs the analysis pipeline.
5. Tables and charts are displayed interactively.

---

## 18. Key Modeling Considerations

### Why Lag Features Matter

Electricity generation tends to be autocorrelated. A source value in one year is often influenced by the previous years. Lag features help the model capture this memory.

### Why Rolling Statistics Matter

Rolling mean and rolling standard deviation capture short-term trend and variability. They help the model understand not only the last value but also the recent pattern.

### Why Multiple Models Are Compared

Different resources may have different dynamics:

- Solar may show stronger growth trends.
- Coal may evolve more slowly.
- Wind and hydro may be more volatile in some datasets.

No single model is always best, so the project compares several algorithms and chooses the best one per resource.

### Why Recursive Forecasting Is Used

Multi-year forecasting requires using predicted values as new inputs. Recursive forecasting is a practical way to extend one-step models into multi-step prediction.

---

## 19. Benefits of the Project

### Analytical Benefits

- Provides a structured view of energy mix changes
- Makes renewable transition patterns easier to understand
- Helps identify long-term trends and correlations

### Technical Benefits

- Modular and reusable codebase
- End-to-end automated pipeline
- Reproducible outputs saved to disk
- Easy integration with dashboards and reports

### Educational Benefits

- Demonstrates full data science workflow
- Shows practical feature engineering for time series
- Teaches model comparison and evaluation
- Connects analytics with a real public dataset

### Business and Policy Benefits

- Supports energy planning discussions
- Helps compare renewable and non-renewable source behavior
- Can inform scenario analysis for investment or policy

---

## 20. Limitations

The current solution has some limitations:

- It works at annual granularity only.
- It depends on historical trend continuation.
- It does not model policy shocks or macroeconomic changes explicitly.
- It does not include external drivers such as population, GDP, weather, or fuel prices.
- The forecast quality may vary across resources.

These limitations are acceptable for a project of this scope but can be improved in future versions.

---

## 21. Conclusion

This project provides a complete example of applied energy analytics using a real-world dataset. It demonstrates how raw electricity generation data can be transformed into a useful forecasting and visualization system through careful preprocessing, feature engineering, model comparison, and interactive reporting.

The project is valuable because it is not only technically sound but also practical and explainable. It brings together the main components of a modern analytics workflow:

- Data ingestion
- Cleaning and transformation
- Exploratory analysis
- Predictive modeling
- Forecast generation
- User-facing visualization

The result is a strong foundation for energy trend analysis and a useful base for future enhancements.

---

## 22. Future Scope

The project can be extended in several ways.

### 22.1 Add More Exogenous Variables

- Temperature
- Population
- GDP
- Energy prices
- Industrial output

### 22.2 Improve Forecasting Models

- SARIMA
- XGBoost or LightGBM
- LSTM or GRU
- Transformer-based forecasting
- Hybrid statistical-ML models

### 22.3 Increase Granularity

- Monthly forecasting
- Quarterly forecasting
- Multi-region comparison

### 22.4 Add Scenario Simulation

- High-renewables scenario
- Conservative growth scenario
- Policy-driven transition scenario

### 22.5 Add Better Validation

- Walk-forward validation
- Backtesting across multiple cutoffs
- Cross-country benchmarking

### 22.6 Enhance the UI

- Download buttons for charts
- Export filters for tables
- Country comparison view
- Forecast confidence intervals

### 22.7 Add Automated Reporting

- PDF report generation
- PowerPoint export
- Email delivery of results

---

## 23. Recommended Project Presentation Structure

If this project is used in a report or viva, the presentation can be structured as:

1. Introduction
2. Problem statement
3. Dataset description
4. SRS and objectives
5. Literature survey
6. Methodology
7. System architecture
8. Model comparison
9. Forecast results
10. Streamlit demo
11. Limitations
12. Future scope
13. Conclusion

---

## 24. Final Summary

This project is a complete energy resource analytics and forecasting system built on a real open dataset. It combines data engineering, feature engineering, machine learning, time-series prediction, and dashboarding into one coherent application. The approach is practical, reproducible, and suitable for academic submission, technical demonstration, or portfolio presentation.

If you want this document converted into a formal report style with abstract, acknowledgements, references, and chapter numbering, that can be added as a second version.