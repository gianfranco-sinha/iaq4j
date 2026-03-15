# AGENTS.md

Development guide for agentic coding agents working on the iaq4j repository.

## Project Overview

**iaq4j** — ML platform for indoor air quality prediction. Trains/serves MLP, KAN, LSTM, CNN, BNN models. Default sensor: BME680. Default standard: BSEC IAQ.

**Tech Stack**: Python 3.9.x (required), FastAPI, PyTorch, InfluxDB, MLflow
**KAN**: vendored in `app/kan.py` — do NOT pip install it

## Commands

### Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

### Training
```bash
python -m iaq4j train --model mlp --epochs 5      # quick dev test (synthetic)
python -m iaq4j train --model mlp --epochs 200    # full training
python -m iaq4j train --model all --epochs 50
python -m iaq4j train --model mlp --data-source influxdb   # real data
python -m iaq4j list
python -m iaq4j version
python -m iaq4j verify [--model mlp]
```

### Testing (pytest — 266+ tests)
```bash
python -m pytest                          # all tests
python -m pytest tests/unit/              # unit tests only
python -m pytest tests/integration/      # integration tests
python -m pytest tests/unit/test_models.py -v             # single file
python -m pytest tests/unit/test_models.py::TestBuildModel -v           # single class
python -m pytest tests/unit/test_models.py::TestBuildModel::test_build_mlp -v  # single test
python -m pytest --cov=app --cov=training tests/  # with coverage
python test_client.py                    # integration test client (server must run)
```

### Linting & Type Checking
```bash
ruff check app/ training/ --fix
mypy app/
black app/ training/
```

### MLflow & TensorBoard
```bash
mlflow ui --port 5000
tensorboard --logdir runs/
```

## Architecture

**Two training paths**:
1. CLI (`python -m iaq4j train`) → uses `SyntheticSource` by default
2. Standalone (`train_models.py`, `train_all_models.py`) → fetches real InfluxDB data

**Training pipeline FSM**: SOURCE_ACCESS → INGESTION → FEATURE_ENGINEERING → WINDOWING → SPLITTING → SCALING → TRAINING → EVALUATION → SAVING

**Bounded contexts**: Sensing (schemas, profiles), Inference (prediction), Training (pipeline), Configuration (settings)

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

# Local (absolute imports)
from app.models import IAQPredictor
from app.schemas import SensorReading, IAQResponse
from app.config import settings
```

### Naming
- **Classes**: PascalCase (`IAQPredictor`, `TrainingPipeline`)
- **Functions/Methods**: snake_case (`predict_iaq`, `train_single_model`)
- **Variables**: snake_case (`model_type`, `window_size`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `MAX_RETRIES`)
- **Model types**: lowercase strings (`mlp`, `kan`, `lstm`, `cnn`, `bnn`)

### Type Hints & Documentation
- Use type hints for ALL function signatures including returns
- Use Pydantic models for API request/response validation
- Prefer `Optional[X]` over `X | None`, `Union[A, B]` over `A | B` (Python 3.9 compat)
- Docstrings in triple quotes for classes and public methods

### Error Handling
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
- Model config: `model_config.yaml`
- Database config: `database_config.yaml`
- App settings: `app/config.py` with `.env` support

## Domain-Driven Design

### Ubiquitous Language
- **SensorReading** — single timestamped observation from physical sensor
- **IAQResponse** — model prediction output (score, category, uncertainty)
- **SensorProfile** — defines what a sensor measures and feature engineering
- **IAQStandard** — scoring/categorisation scheme (e.g. BSEC)
- **InferenceEngine** — transforms SensorReading to IAQResponse

### DDD Guidelines
- Access entities via registries, never instantiate ad-hoc
- Value objects are immutable — don't mutate after construction
- Keep domain logic out of FastAPI routes
- Never leak infrastructure concerns (InfluxDB, file paths) into domain objects

## Agent Boundaries

### Never do without confirmation
- Run `bash deploy/deploy.sh` or any `ssh` to production
- `git push` or destructive git ops (`reset --hard`, force push)
- `python -m iaq4j train --model all` (long running)
- Modify/delete model artifacts in `trained_models/`

### Safe to do autonomously
- Lint, format, type-check
- Quick training with `--epochs 5` for dev testing
- Run tests or test_client.py

## Secrets

**Never commit**: API keys, tokens, passwords, SSH keys, `.env` files, real credentials in config files. Use `.env` overrides instead.

## Critical Constraints
- Python 3.9.x required (KAN incompatible with 3.10+)
- KAN is vendored — do not pip install efficient-kan
