"""Generate evaluation plots and SHAP feature-importance for the trained model.

Produces (into artifacts/):
  - feature_importance.png  : top features driving failure predictions
  - pr_curve.png            : precision-recall curve on held-out machines

SHAP is optional; if unavailable we fall back to the model's built-in importances.
Run: python -m predmaint.explain
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import PrecisionRecallDisplay  # noqa: E402

from .data import generate_run_to_failure, machine_split  # noqa: E402
from .features import add_features  # noqa: E402


def main(artifacts_dir: str | Path = "artifacts") -> None:
    artifacts = Path(artifacts_dir)
    with open(artifacts / "model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    model, cols = bundle["model"], bundle["feature_columns"]

    raw = generate_run_to_failure()
    _, test_df = machine_split(raw)
    test_feat = add_features(test_df)
    X_test, y_test = test_feat[cols].values, test_feat["label"].values

    # --- Precision-Recall curve ---
    prob = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, prob, ax=ax)
    ax.set_title("Precision-Recall (held-out machines)")
    fig.tight_layout()
    fig.savefig(artifacts / "pr_curve.png", dpi=120)
    plt.close(fig)

    # --- Feature importance (SHAP if available, else built-in) ---
    importances, title = _compute_importance(model, X_test, cols)
    order = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([cols[i] for i in order], importances[order])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(artifacts / "feature_importance.png", dpi=120)
    plt.close(fig)
    print("Saved pr_curve.png and feature_importance.png to", artifacts)


def _compute_importance(model, X, cols):
    try:
        import shap

        sample = X[np.random.default_rng(0).choice(len(X), size=min(500, len(X)), replace=False)]
        explainer = shap.Explainer(model.predict, sample)
        vals = explainer(sample).values
        return np.abs(vals).mean(axis=0), "Mean |SHAP| feature importance"
    except Exception:
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_, "Built-in feature importance"
        # linear pipeline fallback
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        return np.abs(getattr(clf, "coef_", np.zeros(len(cols))).ravel()), "Coefficient magnitude"


if __name__ == "__main__":
    main()
