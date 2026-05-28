# Predictive Maintenance — Failure-Risk Forecasting from Sensor Time-Series

End-to-end machine learning system that predicts whether an industrial machine
will fail within the next *N* operating cycles, using multivariate sensor
telemetry (temperature, vibration, spindle load, tool wear, acoustic emission).
Built as a production-shaped project: reproducible data, leakage-safe evaluation,
model comparison with experiment tracking, interpretability, a containerized REST
API, and CI.

> **Domain:** industrial / manufacturing predictive maintenance (CNC machining,
> rotating equipment). The same pipeline architecture generalizes to any
> run-to-failure sensor problem.

## Results

Models are evaluated on **held-out machines** (never seen during training), which
is the correct protocol for run-to-failure prediction — splitting by row would
leak future cycles of a machine into training and inflate scores.

| Model               | ROC-AUC | PR-AUC | F1   | Precision | Recall |
|---------------------|:-------:|:------:|:----:|:---------:|:------:|
| Random Forest       | 0.997   | 0.979  | 0.907| 0.859     | 0.962  |
| Gradient Boosting   | 0.997   | 0.978  | 0.924| 0.902     | 0.947  |
| Logistic Regression | 0.995   | 0.963  | 0.824| 0.709     | 0.983  |

PR-AUC is the headline metric because failures are the minority class (~12% of
cycles). Plots (`pr_curve.png`, `feature_importance.png`) are written to
`artifacts/` by `python -m predmaint.explain`.

## Why this is set up the way it is

- **Leakage-safe split.** Train/test are partitioned *by machine*, not by row. A
  unit test (`test_machine_split_has_no_leakage`) enforces this.
- **Causal features.** Every rolling/trend feature uses only past and current
  cycles. A test (`test_features_are_causal`) verifies a cycle's features don't
  change when future cycles are removed — i.e. the pipeline would behave
  identically in production.
- **Imbalance-aware.** Class weighting + PR-AUC as the selection metric, rather
  than accuracy, which is misleading on imbalanced data.
- **Interpretability.** SHAP feature importance (with a built-in-importance
  fallback) explains *which* sensor behaviors drive a high-risk prediction.

## Feature engineering

From 5 raw sensors the pipeline derives **60 features** per cycle:

- Rolling **mean** and **std** over windows of 5/10/20 cycles (level + volatility)
- Rolling **slope** (least-squares trend) — captures *rate* of degradation
- **Lag-1 delta** — instantaneous change
- **Expanding max** — worst-seen-so-far, a monotone degradation proxy

## Quickstart

```bash
git clone https://github.com/P3NX1L/<repo>.git
cd <repo>
pip install -e ".[ml,dev]"      # core + mlflow/shap + dev tools

python -m scripts.run train     # generate data, train, evaluate, persist best model
python -m scripts.run explain   # write PR curve + SHAP importance to artifacts/
python -m scripts.run demo      # score one simulated machine
```

### Serve the model

```bash
uvicorn predmaint.api:app --reload --port 8000
# docs at http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "machine_id": 7,
        "history": [
          {"cycle": 0, "temperature": 66, "vibration": 0.35,
           "spindle_load": 50, "tool_wear": 0.0, "acoustic_emission": 1.0}
        ]
      }'
```

### Docker

```bash
docker build -t predmaint .
docker run -p 8000:8000 predmaint   # model is trained at build time
```

### Experiment tracking

When MLflow is installed, every training run logs params and metrics:

```bash
mlflow ui          # then open http://localhost:5000
```

MLflow is optional — the pipeline runs and skips logging gracefully if absent.

## Project layout

```
src/predmaint/
  data.py        # synthetic run-to-failure dataset (reproducible, offline)
  features.py    # causal rolling/trend feature engineering
  train.py       # model comparison, metrics, MLflow logging, persistence
  predict.py     # inference wrapper
  explain.py     # SHAP + PR-curve plots
  api.py         # FastAPI service with pydantic validation
tests/           # leakage, causality, metric, and end-to-end tests
scripts/run.py   # CLI: train | explain | demo
Dockerfile       # containerized API, model baked in at build
.github/workflows/ci.yml   # lint + test on py3.10–3.12
```

## Testing

```bash
pytest --cov=predmaint
```

The suite covers data integrity, the no-leakage split, feature causality, metric
correctness, and an end-to-end train run.

## Notes & limitations

- The dataset is **synthetic** but models realistic degradation dynamics so the
  full pipeline is reproducible without external downloads. Swapping in a real
  dataset (e.g. NASA C-MAPSS turbofan, or a CNC telemetry export) requires only a
  new loader returning the same long-format schema.
- Scores are high partly because synthetic degradation is cleaner than real
  shop-floor data; on real telemetry expect lower, noisier numbers — the value
  here is the *methodology and engineering*, not the headline AUC.

## License

MIT
