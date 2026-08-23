from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DROP_COLUMNS, MODEL_PATH, PREDICTIONS_PATH, TARGET_COLUMN


def parse_args():
    parser = argparse.ArgumentParser(description="Run churn predictions on a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default=str(PREDICTIONS_PATH), help="Path to output CSV file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(args.input)

    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if TARGET_COLUMN in df.columns:
        columns_to_drop.append(TARGET_COLUMN)

    features = df.drop(columns=columns_to_drop, errors="ignore")
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)

    result = df.copy()
    result["churn_probability"] = probabilities
    result["churn_prediction"] = predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
