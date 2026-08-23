from pathlib import Path
import pandas as pd
from .config import FEATURES, TARGET

REQUIRED_COLUMNS = FEATURES + [TARGET]

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put autoinsurance_churn.csv in data/raw/."
        )
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = out[out[TARGET].isin([0, 1])]

    for col in [
        "curr_ann_amt", "days_tenure", "age_in_years",
        "latitude", "longitude", "income",
        "has_children", "length_of_residence",
        "home_owner", "college_degree", "good_credit"
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["city", "county", "marital_status", "home_market_value"]:
        # Keep categorical columns as object dtype and use None instead of
        # pandas.NA so scikit-learn's SimpleImputer can process them safely.
        out[col] = out[col].astype(object)
        out[col] = out[col].where(out[col].notna(), None)

    return out.reset_index(drop=True)
