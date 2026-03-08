# AGENTS.md

Development guide for agentic coding agents working on the iaq4j repository.

## Project Overview

**iaq4j** — ML platform for indoor air quality prediction. Trains and serves MLP, KAN, LSTM, CNN, and BNN models. Default sensor: BME680. Default standard: BSEC IAQ. Scope: any indoor air quality sensor, any indoor IAQ standard.

**Technology Stack**: Python 3.9.x, FastAPI, PyTorch, InfluxDB, MLflow
**Python Requirement**: Python 3.9.x REQUIRED (KAN is incompatible with 3.10+)
**KAN**: `efficient-kan` is vendored in `app/kan.py` — do NOT install it via pip.

## System Assumptions

- **Timestamp latency is insignificant for this domain.** IAQ readings are sampled every 3–10 seconds. The worst-case end-to-end timestamp delta (MCU → Node-RED Pi aggregator → InfluxDB Cloud) is in the order of milliseconds — orders of magnitude below the sampling interval. No timestamp correction or source tracking is required. Reading ordering and gap detection are handled by `sequence_number` instead.
- **Hardware topology**: MCU sensors (LAN) → Node-RED on Pi aggregator (LAN) → Cloud. The Pi's role is sensor aggregation only. The IAQ service and InfluxDB reside in the cloud or on high-performance local hardware (e.g. Mac Mini M4) — the service is too resource-intensive for Pi hardware.
- **One sensor type, one IAQ standard per deployment** — selected via `model_config.yaml`.

## Design Principles

This project follows **Domain-Driven Design (DDD)**. When adding or modifying code, respect the domain model and its language.

### Ubiquitous Language
Use the domain's own terms consistently in code, comments, and docs:
- **SensorReading** — a single timestamped observation from a physical sensor (not "input", "sample", or "row")
- **IAQResponse** — the model's prediction output including score, category, and uncertainty (not "result" or "output")
- **SensorProfile** — the domain concept that defines what a sensor measures and how to engineer features from it (not "sensor config")
- **IAQStandard** — the scoring and categorisation scheme (e.g. BSEC) a deployment is calibrated against (not "target config")
- **InferenceEngine** — the domain service that transforms a SensorReading into an IAQResponse (not "predictor wrapper")
- **TrainingPipeline** — the bounded workflow that produces a trained model artifact from a data source

### Bounded Contexts
| Context | Responsibility | Key files |
|---|---|---|
| **Sensing** | Raw observation schema, field mapping, sensor/standard registry | `app/schemas.py`, `app/profiles.py`, `app/builtin_profiles.py`, `app/field_mapper.py` |
| **Inference** | Real-time prediction, uncertainty, Bayesian updates | `app/inference.py`, `app/models.py` |
| **Training** | Data ingestion, feature engineering, model fitting, provenance | `training/` |
| **Configuration** | Settings, quantity registry, YAML config | `app/config.py`, `app/quantities.py`, `quantities.yaml` |

### DDD Guidelines
- **Entities** (`SensorProfile`, `IAQStandard`) have identity — always access them via the registry, never instantiate ad-hoc
- **Value objects** (`SensorReading`, `IAQResponse`) are immutable — do not mutate fields after construction
- **Domain services** (`InferenceEngine`, `FieldMapper`, `TrainingPipeline`) encapsulate workflows that don't belong to a single entity
- **Repositories** (`DataSource` ABC and its implementations) abstract data access — never query InfluxDB or the filesystem directly outside of a `DataSource`
- **Domain events** (`StageResult`) communicate state transitions in the training pipeline — do not bypass them with direct side effects
- Keep domain logic out of the FastAPI layer (`app/main.py`) — routes should delegate immediately to domain services
- Do not leak infrastructure concerns (InfluxDB, file paths, HTTP) into domain objects

## Architecture

### Two Training Paths
1. **CLI path** (`python -m iaq4j train`) → `iaq4j/__main__.py` → `iaq4j/model_trainer.py` → `training/train.py:train_single_model()` → `training/pipeline.py:TrainingPipeline`. Uses `SyntheticSource` by default; switchable via `--data-source`.
   - **Use for**: development, CI, synthetic-data experiments, CSV or Label Studio data
2. **Standalone scripts** (`train_models.py`, `train_all_models.py`) → `training/utils.train_model()`. Fetch real data from InfluxDB.
   - **Use for**: production retraining against live sensor history in InfluxDB

Both paths save artifacts to `trained_models/{model_type}/`.

### Training Pipeline
FSM with stages: `SOURCE_ACCESS → INGESTION → FEATURE_ENGINEERING → WINDOWING → SPLITTING → SCALING → TRAINING → EVALUATION → SAVING`. Chronological train/val split (no shuffling) to prevent temporal data leakage.

### Data Sources
- `SyntheticSource` — default, generates physically plausible random readings
- `InfluxDBSource` — reads from `iaq_readings` measurement
- `CSVDataSource` — flat CSV files
- `LabelStudioDataSource` — exports annotated projects from Label Studio

### Sensor & Standard Abstractions
- `app/profiles.py` — `SensorProfile` ABC, `IAQStandard` ABC, registries
- `app/builtin_profiles.py` — `BME680Profile`, `BSECStandard` (registered at import via side effect)
- Selected via `sensor.type` and `iaq_standard.type` in `model_config.yaml`
- Feature engineering is code (per-profile `engineer_features()` method), not config-driven

### Physical Quantity Registry
- `quantities.yaml` — canonical table of all physical quantities (units, valid ranges, conversions)
- `app/quantities.py` — loads it lazily; exposes `get_quantity()`, `convert_to_canonical()`, `list_quantities()`

### Semantic Field Mapping
- `app/field_mapper.py` — `FieldMapper` with 3-tier strategy (exact → fuzzy → Ollama LLM)
- CLI: `python -m iaq4j map-fields --source <file>`
- REST: `POST /sensors/register`

### Model Artifacts & Semver
- Artifacts per model: `model.pt`, `config.json`, `feature_scaler.pkl`, `target_scaler.pkl`, `MANIFEST.json`, `data_manifest.json`
- Version format: `{model_type}-{MAJOR}.{MINOR}.{PATCH}` (e.g. `mlp-1.2.0`)
- `schema_fingerprint` = SHA256[:12] of (sensor_type, iaq_standard, window_size, num_features, model_type)
- MAJOR bumps on schema change — avoid touching fingerprint inputs without understanding the impact

### Merkle Tree Provenance
- `training/merkle.py` — 6-level chain: Sensor → RawData → CleansedData → PreprocessedData → SplitData → TrainedModel
- Root hash stored in `config.json`, `MANIFEST.json`, `data_manifest.json`
- Verify with `python -m iaq4j verify`

### Input Dimensions
The training pipeline always passes data as flattened `(batch, window_size * num_features)` to all models.
- **MLP / KAN / BNN**: consume this flattened shape directly
- **LSTM**: reshapes internally in `forward()` → `(batch, window_size, num_features)`
- **CNN**: reshapes internally in `forward()` → `(batch, window_size, num_features)`, then permutes → `(batch, num_features, window_size)` for Conv1d

## Development Commands

### Running the Application
```bash
# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

### Training Models
```bash
# Train via CLI (uses synthetic data by default)
python -m iaq4j train --model mlp --epochs 100
python -m iaq4j train --model lstm --epochs 200
python -m iaq4j train --model all --epochs 50

# Train with real data
python -m iaq4j train --model mlp --data-source influxdb
python -m iaq4j train --model mlp --data-source csv --csv-path data.csv

# Standalone scripts (InfluxDB real data)
python train_models.py        # MLP + KAN
python train_all_models.py    # MLP + CNN + KAN

# Create untrained dummy models for dev/testing
python training/create_dummy_models.py

# Model management
python -m iaq4j list
python -m iaq4j version               # show active semver + metrics
python -m iaq4j verify [--model mlp]  # verify Merkle tree provenance

# Semantic field mapping
python -m iaq4j map-fields --source data.csv [--backend fuzzy|ollama] [--save] [-y]
```

### Testing
```bash
# Integration test client (simulates sensor data against a running server)
python test_client.py

# No pytest suite exists yet — do not reference or run pytest
```

### MLflow Tracking
```bash
mlflow ui --port 5000
```

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Linting & Type Checking
```bash
ruff check app/ training/ --fix
mypy app/
black app/ training/
```

## Code Style

### Import Order
```python
# Standard library
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

# Third-party
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local imports (use absolute imports)
from app.models import IAQPredictor
from app.schemas import SensorReading, IAQResponse
from app.config import settings
```

### Naming Conventions
- **Classes**: PascalCase (`IAQPredictor`, `MLPRegressor`, `TrainingPipeline`)
- **Functions/Methods**: snake_case (`predict_iaq`, `load_model`, `train_single_model`)
- **Variables**: snake_case (`model_type`, `window_size`, `num_features`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `WINDOW_SIZE`, `MAX_RETRIES`)
- **Files**: snake_case (`main.py`, `models.py`, `data_sources.py`)
- **Model types**: Always lowercase strings (`mlp`, `kan`, `lstm`, `cnn`, `bnn`)

### Type Hints & Documentation
- Use type hints for ALL function signatures including return types
- Use Pydantic models (`BaseModel`) for API request/response validation
- Docstrings in triple quotes for classes and public methods
- Prefer `Optional[X]` over `X | None`, `Union[A, B]` over `A | B` for Python 3.9 compatibility

### Error Handling
- Use FastAPI's `HTTPException` for API errors with appropriate status codes
- Log errors with `logger.error()` or `logger.exception()`
- Never expose internal exception details to clients in production

```python
try:
    result = engine.predict_single(readings)
    return IAQResponse(**result)
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Prediction error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Prediction failed")
```

### Configuration
- Always use `from app.config import settings` — never pass config as function args
- Model config: `model_config.yaml` — architecture, sensor profile, IAQ standard, hyperparameters
- Database config: `database_config.yaml` — InfluxDB connection (supports 1.x and 2.x)
- App settings: `app/config.py` — Pydantic Settings with `.env` support

## File Structure
```
app/                    # FastAPI app
  main.py               # startup, routes, global predictors/inference_engines dicts
  config.py             # Settings singleton
  models.py             # IAQPredictor + all model classes (MLP, KAN, LSTM, CNN, BNN)
  schemas.py            # SensorReading, IAQResponse Pydantic models
  inference.py          # InferenceEngine wrapping IAQPredictor
  profiles.py           # SensorProfile ABC, IAQStandard ABC, registries
  builtin_profiles.py   # BME680Profile, BSECStandard (import as side effect to register)
  quantities.py         # Physical quantity registry loader
  field_mapper.py       # FieldMapper (exact/fuzzy/LLM field mapping)
  kan.py                # Vendored efficient-kan
training/
  train.py              # train_single_model() — used by CLI path
  pipeline.py           # TrainingPipeline FSM
  data_sources.py       # SyntheticSource, InfluxDBSource, CSVDataSource, LabelStudioDataSource
  utils.py              # train_model(), get_device(), compute_semver(), on_epoch callback
  merkle.py             # Merkle tree provenance
iaq4j/                  # CLI package
  __main__.py           # Entry point (train, list, version, verify, map-fields)
  model_trainer.py      # Bridges CLI to training/train.py
trained_models/         # Saved model artifacts (model.pt, config.json, scalers, MANIFEST.json)
quantities.yaml         # Central physical quantity table
model_config.yaml       # Source of truth for model + sensor + training config
database_config.yaml    # InfluxDB connection config
deploy/                 # deploy.sh, iaq4j.service (systemd), nginx-iaq4j.conf
```

## Deployment

Production server: `pi@<host>` → `/home/pi/iaq4j/`

```bash
# Full deploy (rsync + deps + systemd restart + nginx + verify)
bash deploy/deploy.sh

# Docker (alternative)
docker compose up -d

# View production logs
ssh pi@<host> 'journalctl -u iaq4j -f'
```

- Systemd service runs uvicorn on `127.0.0.1:8001`
- Nginx reverse-proxies `/iaq4j/` → `localhost:8001`
- `ROOT_PATH` env var must be set to `/iaq4j` in production for correct subpath routing

## Agent Boundaries

### Never do without explicit user confirmation
- Run `bash deploy/deploy.sh` or any `ssh` command to production
- `git push` to any remote
- Destructive git operations: `reset --hard`, force push, `checkout .`, `restore .`, `clean -f`
- `python -m iaq4j train --model all` — can run for a very long time; confirm first
- Modify or delete existing model artifacts in `trained_models/`

### Ask before doing
- Adding or removing entries in `requirements.txt`
- Changing `SensorReading` or `IAQResponse` schemas — backward compatibility implications
- Modifying `quantities.yaml` — affects all sensor profiles and unit conversions globally
- Changing values in `model_config.yaml` — affects training and inference globally
- Any change to schema fingerprint inputs (`sensor_type`,bnn `iaq_standard`, `window_size`, `num_features`, `model_type`) — triggers a MAJOR artifact version bump
- Modifying the training pipeline FSM stages in `training/pipeline

### Safe to do autonomously
- Lint (`ruff`), format (`black`), type-check (`mypy`)
- Train with synthetic data at low epochs (`--epochs 5`) for dev testing
- Run `python test_client.py` against a local server
- Read any file, edit local source files

## Secrets & Sensitive Data

**Never commit or push to any repository:**
- API keys or tokens (InfluxDB tokens, Anthropic API keys, Ollama API keys, any LLM credentials)
- Passwords or passphrases
- Production server IP addresses or hostnames
- SSH keys or deploy credentials
- `.env` files or any file containing real credentials
- `database_config.yaml` if it contains real InfluxDB credentials — use `.env` overrides instead

If you discover secrets already present in a file, flag it to the user immediately rather than committing.

## Critical Constraints
- Python 3.9.x required — KAN is incompatible with 3.10+
- KAN is vendored in `app/kan.py` — do not install or reference the external package
- No pytest suite exists — do not run or scaffold pytest
Make

## Quick Reference
```bash
python -m iaq4j train --model mlp --epochs 5    # Quick dev test (synthetic data)
python -m iaq4j train --model mlp --epochs 200  # Full training
python -m iaq4j version                         # Show active semver + metrics
python -m iaq4j verify                          # Verify Merkle tree provenance
python test_client.py                           # Integration test (server must be running)
```
