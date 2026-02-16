# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

iaq4j — ML platform for indoor air quality prediction. **Scope: any indoor air quality sensor, any indoor IAQ standard, ML-driven prediction.** Trains and serves MLP, KAN, LSTM, and CNN models. Python 3.9+, FastAPI, PyTorch, InfluxDB. Default sensor: BME680. Default standard: BSEC IAQ.

## Commands

```bash
# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train models (CLI — uses synthetic data by default)
python -m iaq4j train --model mlp --epochs 200
python -m iaq4j train --model all --epochs 50
python -m iaq4j train --model mlp --data-source influxdb  # real data
python -m iaq4j list

# Train from real InfluxDB data (standalone scripts)
python train_models.py          # MLP + KAN
python train_all_models.py      # MLP + CNN + KAN

# Create untrained dummy models for dev/testing
python training/create_dummy_models.py

# Integration test (sends simulated sensor readings to running server)
python test_client.py

# No pytest suite exists yet
```

## Architecture

**Two training paths** — this is the most important thing to understand:
1. `python -m iaq4j` CLI → `iaq4j/__main__.py` → `iaq4j/model_trainer.py` → `training/train.py:train_single_model()` → `training/pipeline.py:TrainingPipeline` — uses `SyntheticSource` by default or `InfluxDBSource` via `--data-source influxdb`
2. `train_models.py` / `train_all_models.py` (standalone scripts) → `training/utils.train_model()` — fetches **real data from InfluxDB**

Both paths save artifacts to `trained_models/{model_type}/` (model.pt, config.json, optional scaler.pkl files).

**FastAPI service** (`app/main.py`): loads all 4 models at startup into global `predictors` and `inference_engines` dicts. Active model switchable via `/model/select`. Predictions optionally written to InfluxDB.

**Inference flow**: `SensorReading` → `InferenceEngine` → `IAQPredictor.predict()` → profile feature engineering → scaler transform → torch forward pass → standard clamp/categorize → `IAQResponse`

**Sensor & standard abstractions** (`app/profiles.py`): `SensorProfile` ABC defines raw features, valid ranges, and feature engineering. `IAQStandard` ABC defines target scale and category breakpoints. Built-in implementations in `app/builtin_profiles.py` (BME680 + BSEC). Selected via `sensor.type` and `iaq_standard.type` in `model_config.yaml`.

**Config**: `model_config.yaml` is source of truth for model architecture, sensor profile, and IAQ standard. `database_config.yaml` for InfluxDB. Both loaded by `app/config.py:Settings` singleton (`settings`). YAML overrides hardcoded defaults.

## Key Technical Details

- **Features are profile-driven**: `SensorProfile.raw_features` + `SensorProfile.engineered_feature_names` determine `num_features`. BME680 default: 4 raw + 2 engineered = 6.
- **Feature engineering is code**: each `SensorProfile` subclass owns its `engineer_features()` method. No config DSL.
- **Sliding window**: all models buffer `window_size` readings before first prediction. `IAQPredictor.buffer` manages this.
- **Input dimensions**: MLP/KAN flatten to `window_size × num_features`; LSTM/CNN keep temporal shape `(batch, window_size, num_features)`.
- **Scaling**: StandardScaler for features, MinMaxScaler(0,1) for targets. Scalers saved as .pkl alongside models.
- **KAN**: `efficient-kan` vendored in `app/kan.py` — external package breaks on Python >3.9 on Apple Silicon.
- **InfluxDB**: Dual support for 1.x (`influxdb` client) and 2.x (`influxdb-client`). Disabled by default.
- **GPU**: Auto-detects MPS/CUDA/CPU via `training/utils.get_device()`.

## Conventions

- Config access: always use `from app.config import settings` — never pass DB/model params as function args
- Model types are lowercase strings: `mlp`, `kan`, `lstm`, `cnn`
- Model artifacts at `trained_models/{model_type}/`
- Absolute imports: `from app.models import IAQPredictor`
- All models inherit `torch.nn.Module`
