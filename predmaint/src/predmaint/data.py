"""Synthetic predictive-maintenance dataset generation.

Simulates run-to-failure sensor histories for a fleet of machines. Each machine
records multivariate time-series (temperature, vibration, spindle load, tool wear,
acoustic emission) that degrade as the machine approaches failure. The target is a
binary label: will this machine fail within the next `horizon` cycles?

This mirrors the structure of real industrial telemetry (e.g. CNC machining cells,
turbofan engines) without requiring a network download, so the pipeline is fully
reproducible offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "spindle_load",
    "tool_wear",
    "acoustic_emission",
]


@dataclass
class DataConfig:
    """Configuration for synthetic dataset generation."""

    n_machines: int = 100
    min_cycles: int = 80
    max_cycles: int = 300
    failure_horizon: int = 20  # label = 1 if failure occurs within this many cycles
    noise_scale: float = 0.08
    seed: int = 42


def _degradation_curve(n: int, rng: np.random.Generator) -> np.ndarray:
    """A monotone-ish degradation signal in [0, 1] over the machine lifetime.

    Combines a slow exponential ramp with a late-life acceleration so that the
    signal is hard to separate early but clearly degrades near failure.
    """
    t = np.linspace(0.0, 1.0, n)
    ramp = t ** 2.2
    accel = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.82)))  # sharp rise near end of life
    curve = 0.6 * ramp + 0.4 * accel
    jitter = rng.normal(0.0, 0.02, size=n).cumsum() * 0.05
    return np.clip(curve + jitter, 0.0, 1.5)


def generate_run_to_failure(config: DataConfig | None = None) -> pd.DataFrame:
    """Generate a long-format run-to-failure dataset.

    Returns a DataFrame with one row per (machine, cycle) and columns:
        machine_id, cycle, <sensors...>, rul (remaining useful life), label
    """
    config = config or DataConfig()
    rng = np.random.default_rng(config.seed)

    # Per-sensor baselines and how strongly each responds to degradation.
    baselines = {
        "temperature": 65.0,
        "vibration": 0.35,
        "spindle_load": 50.0,
        "tool_wear": 0.0,
        "acoustic_emission": 1.0,
    }
    sensitivities = {
        "temperature": 30.0,
        "vibration": 1.4,
        "spindle_load": 25.0,
        "tool_wear": 100.0,
        "acoustic_emission": 4.0,
    }

    frames = []
    for machine_id in range(config.n_machines):
        n_cycles = int(rng.integers(config.min_cycles, config.max_cycles + 1))
        deg = _degradation_curve(n_cycles, rng)

        data = {"machine_id": machine_id, "cycle": np.arange(n_cycles)}
        for sensor in SENSOR_COLUMNS:
            base = baselines[sensor]
            sens = sensitivities[sensor]
            # machine-to-machine offset so models can't just memorise baselines
            offset = rng.normal(0.0, 0.05 * max(abs(base), 1.0))
            signal = base + offset + sens * deg
            noise = rng.normal(0.0, config.noise_scale * max(abs(sens), 1.0), size=n_cycles)
            data[sensor] = signal + noise

        df = pd.DataFrame(data)
        df["rul"] = (n_cycles - 1) - df["cycle"]  # remaining useful life
        df["label"] = (df["rul"] <= config.failure_horizon).astype(int)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def machine_split(
    df: pd.DataFrame, test_frac: float = 0.25, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split BY MACHINE, never by row.

    Splitting on rows would leak future cycles of a machine into training and
    massively inflate scores. We hold out entire machines for test, which is the
    correct evaluation protocol for run-to-failure prediction.
    """
    rng = np.random.default_rng(seed)
    machines = df["machine_id"].unique()
    rng.shuffle(machines)
    n_test = max(1, int(len(machines) * test_frac))
    test_ids = set(machines[:n_test].tolist())

    test_mask = df["machine_id"].isin(test_ids)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


if __name__ == "__main__":
    frame = generate_run_to_failure()
    print(frame.head())
    print("\nshape:", frame.shape)
    print("failure rate:", round(frame["label"].mean(), 3))
