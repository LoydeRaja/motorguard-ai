from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .features import build_preprocessor

def make_pipeline(estimator):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])

def get_models():
    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=4,
            )
        ),
    }
