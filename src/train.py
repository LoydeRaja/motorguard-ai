import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from .config import FEATURES, METRICS_PATH, MODEL_DIR, MODEL_PATH, RAW_DATA, TARGET
from .data import clean_data, load_data
from .model import get_models

RANDOM_STATE = 42

def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

def main():
    df = clean_data(load_data(RAW_DATA))
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("motorguard-insurance-churn")

    results = {}
    best_name = None
    best_model = None
    best_pr_auc = -1.0

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, probabilities)

            mlflow.log_param("model", name)
            mlflow.log_param("features", len(FEATURES))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

            results[name] = metrics

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_name = name
                best_model = model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    report = {
        "target": TARGET,
        "best_model": best_name,
        "selection_metric": "pr_auc",
        "metrics": results,
    }

    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
