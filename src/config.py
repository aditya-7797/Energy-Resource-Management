from __future__ import annotations

DATA_URL = "https://ourworldindata.org/grapher/electricity-prod-source-stacked.csv"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

RESOURCE_COLUMNS = [
    "Solar",
    "Wind",
    "Hydropower",
    "Bioenergy",
    "Other renewables",
    "Nuclear",
    "Coal",
    "Gas",
    "Oil",
]

RENEWABLE_COLUMNS = ["Solar", "Wind", "Hydropower", "Bioenergy", "Other renewables"]
DEFAULT_COUNTRY = "India"
DEFAULT_MIN_YEAR = 1990
DEFAULT_HORIZON = 8
MIN_TRAIN_ROWS = 18
