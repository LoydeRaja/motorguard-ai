# MotorGuard-AI

## Vehicle Insurance Churn Prediction

MotorGuard-AI is an end-to-end machine-learning project that predicts whether an auto-insurance customer is likely to **churn**.

The project uses the original `autoinsurance_churn.csv` dataset and treats `Churn` as the binary classification target.

### Dataset

The supplied dataset contains **1,680,909 rows and 22 columns**. The target distribution is:

- No churn: 1,487,453
- Churn: 193,456
- Churn rate: ~11.5%

The following fields are intentionally excluded from the model:
- `individual_id` / `address_id`: identifiers
- `date_of_birth`: redundant with age
- `cust_orig_date`: redundant with tenure
- `acct_suspd_date`: potentially post-outcome information and therefore leakage risk
- `state`: constant in the supplied data (`TX`)

### Model features

- `curr_ann_amt`
- `days_tenure`
- `age_in_years`
- `latitude`
- `longitude`
- `city`
- `county`
- `income`
- `has_children`
- `length_of_residence`
- `marital_status`
- `home_market_value`
- `home_owner`
- `college_degree`
- `good_credit`

### ML pipeline

```text
autoinsurance_churn.csv
        |
        v
Data validation + cleaning
        |
        v
Train / validation / test split
        |
        v
Preprocessing
  ├── numeric imputation
  └── categorical imputation + encoding
        |
        +---- Logistic Regression
        |
        +---- HistGradientBoosting
        |
        +---- XGBoost
        |
        v
Cross-validation + test evaluation
        |
        +---- MLflow experiment tracking
        |
        v
Best model
        |
        v
FastAPI /predict
        |
        v
Docker
```

### Setup

Use a Python 3.12 virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put the dataset here:

```text
data/raw/autoinsurance_churn.csv
```

Then run:

```powershell
python -m src.train
```

The script creates:

```text
models/churn_model.joblib
models/metrics.json
```

### API

After training:

```powershell
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict
```

Example request:

```json
{
  "curr_ann_amt": 818.88,
  "days_tenure": 1454,
  "age_in_years": 44,
  "latitude": 32.578829,
  "longitude": -96.305006,
  "city": "Kaufman",
  "county": "Kaufman",
  "income": 22500,
  "has_children": true,
  "length_of_residence": 15,
  "marital_status": "Married",
  "home_market_value": "50000 - 74999",
  "home_owner": true,
  "college_degree": true,
  "good_credit": true
}
```

### MLflow

Run:

```powershell
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

The training script logs model metrics and parameters to MLflow.

### Evaluation

Because the target is imbalanced, the project reports:

- ROC-AUC
- PR-AUC / Average Precision
- F1
- Precision
- Recall
- Accuracy
- Confusion Matrix

For churn detection, **ROC-AUC and especially PR-AUC/Recall should be considered alongside accuracy**.

### Docker

After a model has been trained:

```powershell
docker compose up --build
```

The API is available at:

```text
http://127.0.0.1:8000
```

### Tests

```powershell
pytest -q
```

### GitHub Actions

Every push and pull request to `main` runs the test suite automatically.

### Important modeling note

`acct_suspd_date` is not used as a feature because it can contain information that only becomes available after or around the churn event. Using it could produce data leakage and unrealistic model performance.

This is an educational portfolio project. It is not an insurance underwriting or pricing system.


### Windows troubleshooting

If you are already inside the `motorguard-ai-classification` directory, do **not**
run `cd motorguard-ai` again. Start directly with:

```powershell
.venv\Scripts\Activate.ps1
python -m src.train
```

The training pipeline converts missing categorical values to `None` before
scikit-learn preprocessing to avoid the `TypeError: boolean value of NA is ambiguous`
error with pandas `StringDtype`.
