# MotorGuard-AI Architecture

```text
autoinsurance_churn.csv
        |
        v
Validation / cleaning
        |
        v
Stratified train/test split
        |
        v
Preprocessing
  numeric -> median imputation + scaling
  categorical -> most-frequent imputation + one-hot encoding
        |
        +---- Logistic Regression
        +---- XGBoost
        +---- HistGradientBoosting
        |
        v
Evaluation
  ROC-AUC / PR-AUC / F1 / Recall / Precision
        |
        +---- MLflow
        |
        v
Best model (highest PR-AUC)
        |
        v
FastAPI
        |
        v
Docker
```

## Leakage prevention

`acct_suspd_date` is deliberately excluded. A suspension date can encode information that is only known after or around the churn event, which can artificially inflate performance.

Identifiers and redundant date fields are also excluded.
