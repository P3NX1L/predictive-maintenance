"""Training and evaluation for predictive-maintenance failure classification.

Trains and compares several models, evaluates with metrics appropriate for an
imbalanced classification problem (ROC-AUC, PR-AUC, F1, precision/recall), and
logs runs to MLflow when it is installed. MLflow is treated as OPTIONAL so the
pipeline runs in any environment; if it is missing, logging is skipped.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DataConfig, generate_run_to_failure, machine_split
from .features import add_features, feature_columns

try:  # MLflow is optional
    import mlflow

    _HAS_MLFLOW = True
except Exception:  # pragma: no cover - exercised only when mlflow absent
    _HAS_MLFLOW = False


MODELS = {
    "logistic_regression": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
}


@dataclass
class EvalResult:
    model_name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float

    def as_row(self) -> dict:
        return asdict(self)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, name: str, threshold: float = 0.5) -> EvalResult:
    y_pred = (y_prob >= threshold).astype(int)
    return EvalResult(
        model_name=name,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
    )


def train_all(
    data_config: DataConfig | None = None,
    artifacts_dir: str | Path = "artifacts",
    experiment_name: str = "predmaint",
) -> pd.DataFrame:
    """Train every model, evaluate on held-out machines, persist the best one.

    Returns a DataFrame of metrics sorted by PR-AUC (the right headline metric
    for an imbalanced failure-prediction task).
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    raw = generate_run_to_failure(data_config)
    train_df, test_df = machine_split(raw)

    train_feat = add_features(train_df)
    test_feat = add_features(test_df)
    cols = feature_columns(train_feat)

    X_train, y_train = train_feat[cols].values, train_feat["label"].values
    X_test, y_test = test_feat[cols].values, test_feat["label"].values

    if _HAS_MLFLOW:
        mlflow.set_experiment(experiment_name)

    results: list[EvalResult] = []
    fitted: dict[str, object] = {}
    for name, factory in MODELS.items():
        model = factory()
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                res = evaluate(y_test, prob, name)
                mlflow.log_params({"model": name, "n_features": len(cols)})
                mlflow.log_metrics(
                    {k: v for k, v in res.as_row().items() if k != "model_name"}
                )
        else:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            res = evaluate(y_test, prob, name)
        results.append(res)
        fitted[name] = model

    metrics = (
        pd.DataFrame([r.as_row() for r in results])
        .sort_values("pr_auc", ascending=False)
        .reset_index(drop=True)
    )

    best_name = metrics.iloc[0]["model_name"]
    best_model = fitted[best_name]

    with open(artifacts / "model.pkl", "wb") as fh:
        pickle.dump({"model": best_model, "feature_columns": cols, "name": best_name}, fh)
    metrics.to_csv(artifacts / "metrics.csv", index=False)
    metadata = {
        "best_model": best_name,
        "n_features": len(cols),
        "n_machines_test": int(test_df["machine_id"].nunique()),
    }
    with open(artifacts / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    return metrics


if __name__ == "__main__":
    table = train_all()
    print("\n=== Model comparison (held-out machines) ===")
    print(table.to_string(index=False))
    print(f"\nMLflow logging: {'enabled' if _HAS_MLFLOW else 'skipped (not installed)'}")
