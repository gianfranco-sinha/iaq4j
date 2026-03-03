# AGENTS.md

Development guide for agentic coding agents working on the iaq4j repository.

## Project Overview

**iaq4j** — ML platform for indoor air quality prediction. Trains and serves MLP, KAN, LSTM, and CNN models. Default sensor: BME680. Default standard: BSEC IAQ.

**Technology Stack**: Python 3.9+, FastAPI, PyTorch, InfluxDB, MLflow
**Python Requirement**: Python 3.9.x REQUIRED (KAN compatibility is strict - cannot use 3.10+)

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
# Train via CLI (recommended)
python -m iaq4j train --model mlp --epochs 100
python -m iaq4j train --model lstm --epochs 200
python -m iaq4j train --model all --epochs 50

# Train with real data from InfluxDB
python -m iaq4j train --model mlp --data-source influxdb

# List available models
python -m iaq4j list
```

### Testing
```bash
# Integration test client (simulates sensor data)
python test_client.py

# Run all pytest tests
pytest tests/ -v

# Run a single test file/function
pytest tests/test_specific_file.py::test_function_name -v

# Run tests matching a pattern
pytest tests/ -k "test_name_pattern" -v
```

### MLflow Tracking
```bash
# View training experiments (if UI not already running)
mlflow ui --port 5000
```

### TensorBoard
```bash
# View training metrics
tensorboard --logdir runs/
```

### Linting & Type Checking
```bash
# Lint with ruff (auto-fix where possible)
ruff check app/ training/ --fix

# Lint specific file
ruff check app/main.py

# Type check with mypy
mypy app/

# Type check specific file
mypy app/models.py

# Format with black
black app/ training/
```

## Code Style

### Import Order
Imports must follow this three-section order with blank lines between sections:
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
- Prefer `Optional[X]` over `X | None`, `Union[A, B]` over `A | B` for Python 3.9

### Error Handling
- Use FastAPI's `HTTPException` for API errors with appropriate status codes
- Log errors with `logger.error()` or `logger.exception()`
- Return consistent error format in API responses
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
- Model config: `model_config.yaml` — architecture parameters for MLP, KAN, LSTM, CNN
- Database config: `database_config.yaml` — InfluxDB connection (supports 1.x and 2.x)
- App settings: `app/config.py` — Pydantic Settings with `.env` support

## File Structure
```
app/                   # FastAPI app: main.py, config.py, models.py, schemas.py, inference.py
training/              # Training: train.py, pipeline.py, data_sources.py, utils.py
iaq4j/                 # CLI: __main__.py, model_trainer.py
trained_models/        # Saved model artifacts
```

## Critical Constraints
- Python 3.9.x required (KAN incompatibility with 3.10+)
- KAN: `pip install git+https://github.com/Blealtan/efficient-kan.git`
- MLflow: params, metrics, models logged in `training/train.py`
- Model artifacts: `config.json` + `model.pt` + scalers in `trained_models/{model_type}/`

## Quick Reference
```bash
python -m iaq4j train --model mlp --epochs 50   # Full training
python -m iaq4j train --model mlp --epochs 5     # Quick dev test
python test_client.py                            # Integration test
```
