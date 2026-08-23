from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "autoinsurance_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = REPORT_DIR / "model_metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"

TARGET_COLUMN = "Churn"
DROP_COLUMNS = [
    "individual_id",
    "address_id",
    "date_of_birth",
    "cust_orig_date",
    "acct_suspd_date",
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
    "state",
    "county",
    "marital_status",
    "home_market_value",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
