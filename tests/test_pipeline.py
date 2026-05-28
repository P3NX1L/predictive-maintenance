"""Unit tests for the predictive-maintenance pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predmaint.data import DataConfig, generate_run_to_failure, machine_split
from predmaint.features import add_features, feature_columns
from predmaint.train import evaluate, train_all


def test_data_shape_and_labels():
    df = generate_run_to_failure(DataConfig(n_machines=10, seed=1))
    assert {"machine_id", "cycle", "label", "rul"}.issubset(df.columns)
    assert df["label"].isin([0, 1]).all()
    # every machine should eventually fail, so each has at least one positive label
    assert (df.groupby("machine_id")["label"].max() == 1).all()


def test_rul_is_monotone_decreasing_per_machine():
    df = generate_run_to_failure(DataConfig(n_machines=5, seed=2))
    for _, g in df.groupby("machine_id"):
        rul = g.sort_values("cycle")["rul"].values
        assert np.all(np.diff(rul) <= 0)


def test_machine_split_has_no_leakage():
    df = generate_run_to_failure(DataConfig(n_machines=20, seed=3))
    train, test = machine_split(df, test_frac=0.25, seed=3)
    train_ids = set(train["machine_id"].unique())
    test_ids = set(test["machine_id"].unique())
    assert train_ids.isdisjoint(test_ids)  # critical: no machine in both splits


def test_features_no_nans_and_expected_growth():
    df = generate_run_to_failure(DataConfig(n_machines=5, seed=4))
    feats = add_features(df)
    cols = feature_columns(feats)
    assert len(cols) > df.shape[1]  # engineering added columns
    assert not feats[cols].isna().any().any()  # all NaNs filled


def test_features_are_causal():
    """A feature for cycle t must not change if future cycles are removed."""
    df = generate_run_to_failure(DataConfig(n_machines=1, seed=5))
    full = add_features(df)
    cols = feature_columns(full)
    cutoff = df["cycle"].max() // 2
    truncated = add_features(df[df["cycle"] <= cutoff])
    # row at the cutoff cycle should have identical features in both
    a = full[full["cycle"] == cutoff][cols].reset_index(drop=True)
    b = truncated[truncated["cycle"] == cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_evaluate_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    res = evaluate(y_true, y_prob, "perfect")
    assert res.roc_auc == 1.0
    assert res.recall == 1.0


def test_train_all_produces_sorted_metrics(tmp_path):
    metrics = train_all(DataConfig(n_machines=30, seed=6), artifacts_dir=tmp_path)
    assert len(metrics) == 3
    # sorted by pr_auc descending
    assert list(metrics["pr_auc"]) == sorted(metrics["pr_auc"], reverse=True)
    assert (tmp_path / "model.pkl").exists()
    # a competent model should beat random on this separable data
    assert metrics["roc_auc"].max() > 0.8
