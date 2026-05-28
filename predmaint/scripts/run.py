"""Command-line entrypoint: train, evaluate, explain, or score a sample.

Usage:
    python -m scripts.run train
    python -m scripts.run explain
    python -m scripts.run demo
"""
from __future__ import annotations

import argparse

from predmaint.data import DataConfig, generate_run_to_failure
from predmaint.predict import Predictor
from predmaint.train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive maintenance pipeline")
    parser.add_argument("command", choices=["train", "explain", "demo"])
    parser.add_argument("--machines", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "train":
        metrics = train_all(DataConfig(n_machines=args.machines, seed=args.seed))
        print(metrics.to_string(index=False))
    elif args.command == "explain":
        from predmaint.explain import main as explain_main

        explain_main()
    elif args.command == "demo":
        df = generate_run_to_failure(DataConfig(n_machines=1, seed=args.seed))
        risk = Predictor().predict_latest(df)
        print(f"Sample machine latest failure risk: {risk:.3f}")


if __name__ == "__main__":
    main()
