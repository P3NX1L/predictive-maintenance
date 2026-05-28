"""Inference: load a trained model and score new sensor readings."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .features import add_features


class Predictor:
    """Loads a persisted model and produces failure-risk probabilities.

    Expects input as a sequence of recent cycles for a single machine so that
    rolling/trend features can be computed; returns the risk for the latest cycle.
    """

    def __init__(self, model_path: str | Path = "artifacts/model.pkl") -> None:
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.name = bundle["name"]

    def predict_latest(self, history: pd.DataFrame) -> float:
        """Return failure-within-horizon probability for the most recent cycle.

        `history` must contain machine_id, cycle and the raw sensor columns for a
        single machine, ordered (or orderable) by cycle.
        """
        feats = add_features(history)
        latest = feats.sort_values("cycle").iloc[[-1]]
        X = latest[self.feature_columns].values
        return float(self.model.predict_proba(X)[:, 1][0])


if __name__ == "__main__":
    from .data import DataConfig, generate_run_to_failure

    df = generate_run_to_failure(DataConfig(n_machines=1, seed=7))
    predictor = Predictor()
    risk = predictor.predict_latest(df)
    print(f"model={predictor.name}  latest failure risk={risk:.3f}")
