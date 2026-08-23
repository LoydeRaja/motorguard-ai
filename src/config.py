from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "autoinsurance_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "Churn"

FEATURES = [
    "curr_ann_amt",
    "days_tenure",
    "age_in_years",
    "latitude",
    "longitude",
    "city",
    "county",
    "income",
    "has_children",
    "length_of_residence",
    "marital_status",
    "home_market_value",
    "home_owner",
    "college_degree",
    "good_credit",
]

NUMERIC_FEATURES = [
    "curr_ann_amt",
    "days_tenure",
    "age_in_years",
    "latitude",
    "longitude",
    "income",
    "has_children",
    "length_of_residence",
    "home_owner",
    "college_degree",
    "good_credit",
]

CATEGORICAL_FEATURES = [
    "city",
    "county",
    "marital_status",
    "home_market_value",
]
