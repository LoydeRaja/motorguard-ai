from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .config import FEATURES, MODEL_PATH

app = FastAPI(
    title="MotorGuard-AI",
    description="Vehicle insurance customer churn prediction API",
    version="1.1.0",
)

class ChurnRequest(BaseModel):
    curr_ann_amt: float = Field(..., ge=0)
    days_tenure: float = Field(..., ge=0)
    age_in_years: float = Field(..., ge=16)
    latitude: float | None = None
    longitude: float | None = None
    city: str | None = None
    county: str | None = None
    income: float = Field(..., ge=0)
    has_children: bool
    length_of_residence: float = Field(..., ge=0)
    marital_status: str
    home_market_value: str | None = None
    home_owner: bool
    college_degree: bool
    good_credit: bool

def load_model():
    if not Path(MODEL_PATH).exists():
        raise RuntimeError("Model not found. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": Path(MODEL_PATH).exists()}

@app.post("/predict")
def predict(request: ChurnRequest):
    try:
        model = load_model()
        row = pd.DataFrame([request.model_dump()])[FEATURES]
        probability = float(model.predict_proba(row)[0, 1])
        return {
            "churn_probability": round(probability, 6),
            "churn_prediction": int(probability >= 0.5),
            "risk_level": (
                "high" if probability >= 0.5
                else "medium" if probability >= 0.25
                else "low"
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
