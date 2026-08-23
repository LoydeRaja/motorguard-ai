# MotorGuard AI

MotorGuard AI is a machine learning project for predicting churn in auto insurance customers using structured customer, policy, and demographic data.

## Project goal

The goal of this project is to predict whether an auto insurance customer is likely to churn, and to provide a reproducible training and inference pipeline that can be extended into a retention analytics workflow.

## Dataset

This project uses the Kaggle *Auto Insurance Churn Analysis Dataset* and assumes the merged file is available at:

```text
data/raw/autoinsurance_churn.csv
```

### Target

- `Churn` is used as the target variable.
- `acct_suspd_date` is excluded from model features to avoid leakage.

## Features used in the MVP

### Numeric features

- `curr_ann_amt`
- `days_tenure`
- `age_in_years`
- `latitude`
- `longitude`
- `income`
- `has_children`
- `length_of_residence`
- `home_owner`
- `college_degree`
- `good_credit`

### Categorical features

- `city`
- `state`
- `county`
- `marital_status`
- `home_market_value`

## Excluded columns

- `individual_id`
- `address_id`
- `date_of_birth`
- `cust_orig_date`
- `acct_suspd_date`

## Project structure

```text
motorguard-ai/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
│   └── motorguard_ai/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python -m src.motorguard_ai.train
```

The script will:

- load the churn dataset
- split train and test data
- build a preprocessing pipeline
- train logistic regression and random forest models
- evaluate both models
- save the best model to `models/best_model.joblib`
- save metrics to `reports/model_metrics.json`

## Prediction

To score a new CSV file:

```bash
python -m src.motorguard_ai.predict --input path/to/new_data.csv --output reports/predictions.csv
```

## Suggested next improvements

- add hyperparameter tuning
- add SHAP-based explainability
- compare with XGBoost or LightGBM
- build a retention dashboard or API
