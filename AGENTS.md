# AGENTS.md

Development guide for agentic coding agents working on the iaq4j repository.

## Project Overview

**iaq4j** — ML platform for indoor air quality prediction. **Scope: any indoor air quality sensor, any indoor IAQ standard, ML-driven prediction.** Trains and serves MLP, KAN, LSTM, and CNN models. Default sensor: BME680. Default standard: BSEC IAQ.

Originally developed to reproduce BSEC IAQ indices from raw BME680 sensor data. Now sensor-agnostic and IAQ-standard-agnostic via `SensorProfile` and `IAQStandard` abstractions (`app/profiles.py`).
**Technology Stack**: Python 3.9+, FastAPI, PyTorch, InfluxDB
**ML Models**: MLP, KAN, LSTM, CNN for air quality prediction

## Development Environment Setup

### Prerequisites
- **Python 3.9.x REQUIRED** - KAN compatibility is strict (cannot use 3.10+)
- PyTorch 2.1.2 (compatible with CUDA 11.8/12.1)
- Git

### Installation Steps
```bash
# 1. Create virtual environment with Python 3.9
python3.9 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR venv\Scripts\activate  # Windows

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install KAN separately (CRITICAL)
pip install git+https://github.com/Blealtan/efficient-kan.git

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import efficient_kan; print('KAN installed')"
```

## Development Commands

### Build/Lint/Test Commands
```bash
# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# View API docs (auto-generated)
open http://localhost:8000/docs

# Run integration test client (simulates sensor data)
python test_client.py

# Train models via CLI (RECOMMENDED)
python3 -m iaq4j train --model mlp --epochs 100
python3 -m iaq4j train --model lstm --epochs 200
python3 -m iaq4j train --model all --epochs 50

# List available models
python3 -m iaq4j list

# Legacy training scripts (fetch from InfluxDB directly)
python train_models.py          # MLP + KAN
python train_all_models.py      # MLP + CNN + KAN
python training/create_dummy_models.py  # Dummy models for testing
```

### Single Test Execution
```bash
# Note: No formal pytest suite currently exists
# Use the integration client for testing:
python test_client.py

# If adding pytest tests in future:
pytest test_specific_file.py::test_function -v
pytest tests/ -k "test_name_pattern" -v
```

## Code Style and Conventions

### Import Style
- Use absolute imports: `from app.models import IAQPredictor`
- Group imports: standard library → third-party → local imports
- One import per line preferred in main files
- Use type hints consistently
- Keep imports at file top, after docstring header
- Avoid circular imports between app modules

```python
# Standard library
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from app.models import IAQPredictor
from app.schemas import SensorReading, IAQResponse
```

### Formatting and Code Quality
- No auto-formatter currently configured
- Consider adding black for consistent formatting: `black app/ training/`
- Use ruff for fast linting: `ruff check app/ training/`
- Use mypy for type checking: `mypy app/`
- Current project has minimal tooling setup

### Naming Conventions
- **Classes**: PascalCase (`IAQPredictor`, `MLPRegressor`, `InferenceEngine`)
- **Functions/Methods**: snake_case (`predict_iaq`, `load_model`, `reset_buffer`)
- **Variables**: snake_case (`model_type`, `window_size`, `prediction_history`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `WINDOW_SIZE`)
- **Files**: snake_case (`main.py`, `models.py`, `test_client.py`)

### Type Hints and Documentation
- Use type hints for all function signatures
- Docstrings in triple quotes for classes and key methods
- Use Pydantic models for API data validation

```python
def predict(
    self,
    readings: dict = None,
    **kwargs,
) -> dict:
    """
    Predict IAQ from sensor readings.

    Args:
        readings: Dict of sensor readings, e.g. {"temperature": 22, "rel_humidity": 55,
                  "pressure": 1013, "voc_resistance": 85000}

    Returns:
        Dictionary with prediction results
    """
```

### Error Handling
- Use structured error responses with HTTP status codes
- Log errors appropriately with `logger.error()`
- Return consistent error format in API responses

```python
try:
    result = engine.predict_single(...)
    return IAQResponse(**result)
except Exception as e:
    logger.error(f"Prediction error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Configuration Management
- Use `app/config.py` with Pydantic Settings for all configuration
- Environment variables supported via `.env` file
- Default values defined in Settings class
- **Model parameters configurable via `model_config.yaml`**
- **Database connection configurable via `database_config.yaml`**

### YAML Model Configuration
- **File**: `model_config.yaml` in project root
- **Purpose**: Configure model architecture parameters for all model types
- **Structure**: 
  - `global`: Shared settings across all models
  - `mlp`, `kan`, `lstm`, `cnn`: Model-specific configurations
  - `training`: Training parameters (for training scripts)

#### Usage Examples:
```yaml
# Global settings
global:
  window_size: 10
  num_features: 6   # 4 raw + 2 engineered (profile-driven)
  default_dropout: 0.2

# Sensor and IAQ standard selection
sensor:
  type: bme680
iaq_standard:
  type: bsec

# MLP configuration
mlp:
  hidden_dims: [64, 32, 16]
  dropout: 0.2
  input_dim: 6
  activation: "relu"
  use_batch_norm: true

# LSTM configuration
lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true
  fc_layers: [64, 32]
```

### YAML Database Configuration
- **File**: `database_config.yaml` in project root
- **Purpose**: Configure InfluxDB connection parameters and database operations
- **Version Support**: Supports both InfluxDB 1.x and 2.x with different client libraries
- **Structure**:
  - `influxdb`: Connection settings (version, host, port, database, auth)
  - `database`: Operational settings (batch size, retries, retention)
  - `logging`: Debug and performance logging options

#### Usage Examples:
```yaml
# InfluxDB 1.x (traditional)
influxdb:
  version: "1.x"
  host: "localhost"
  port: 8086
  database: "iaq4j_data"
  username: "admin"
  password: "secure_password"
  enabled: true
  timeout: 30

# InfluxDB 2.x (cloud)
influxdb:
  version: "2.x"
  host: "cloud.influxdata.com"
  org: "your_org"
  bucket: "your_bucket"
  token: "your_auth_token"
  enabled: true
  timeout: 30

database:
  batch_size: 500
  max_retries: 5
  data_retention_days: 90

logging:
  log_queries: true
  log_performance: true
```

#### Version-Specific Client Installation:
```bash
# For InfluxDB 1.x
pip install influxdb==5.3.1

# For InfluxDB 2.x
pip install influxdb-client==1.38.0
```

## Build/Lint/Test Commands
```bash
# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# View API docs (auto-generated)
open http://localhost:8000/docs

# Run integration test client (simulates sensor data)
python test_client.py

# Train models via CLI (RECOMMENDED)
python -m iaq4j train --model mlp --epochs 100
python -m iaq4j train --model lstm --epochs 200
python -m iaq4j train --model all --epochs 50
python -m iaq4j train --model mlp --data-source influxdb  # real data

# List available models
python -m iaq4j list
```

### Single Test Execution
```bash
# No formal pytest suite currently exists
# Integration testing:
python test_client.py
```

#### Loading Configuration:
```python
from app.config import settings

# Get model configuration
mlp_config = settings.get_model_config("mlp")
lstm_config = settings.get_model_config("lstm")

# Get database configuration
db_config = settings.get_database_config()
print(f"Database host: {db_config['host']}")
print(f"Database enabled: {db_config['enabled']}")
```

### Model Development Guidelines
- All models inherit from `torch.nn.Module`
- Use model registry pattern in `IAQPredictor`
- Support for temporal models with sliding window buffers
- Device management (CPU/GPU) handled in predictor
- Models saved with `config.json` metadata and `model.pt` weights

### CLI Integration
- Use `python3 -m iaq4j` for training commands
- Supports individual models: `--model mlp|kan|lstm|cnn`
- Supports batch training: `--model all`
- Configurable epochs and window size

## File Structure

```
app/                        # Main FastAPI application
├── main.py                # FastAPI routes and application entry
├── config.py              # Settings using Pydantic Settings + YAML config
├── models.py              # PyTorch model definitions (MLP, KAN, LSTM, CNN)
├── schemas.py             # Pydantic data models for API
├── inference.py           # Inference engine logic
├── database.py            # InfluxDB connection management
├── profiles.py            # SensorProfile ABC, IAQStandard ABC, registries
├── builtin_profiles.py    # BME680Profile, BSECStandard (registered at import)
└── kan.py                 # Vendored efficient-kan implementation

iaq4j/                     # CLI training module
├── __main__.py            # CLI entry point (python -m iaq4j)
└── model_trainer.py       # Model training orchestration

training/                  # Model training utilities
├── train.py               # train_single_model() entry point
├── pipeline.py            # TrainingPipeline FSM (ingest → engineer → train → save)
├── data_sources.py        # DataSource ABC, InfluxDBSource, SyntheticSource
├── utils.py               # save_trained_model(), get_device(), etc.
└── create_dummy_models.py # Create untrained models for dev/testing

trained_models/            # Saved model artifacts
├── mlp/                   # Multi-layer Perceptron models
├── kan/                   # KAN (Kolmogorov-Arnold Networks)
├── lstm/                  # LSTM models
└── cnn/                   # CNN models
```

## Critical Constraints

- **Python 3.9.x required** for KAN compatibility
- KAN must be installed from GitHub, not PyPI
- Temporal models (LSTM, CNN) require sliding window data
- Model loading expects specific directory structure
- YAML config takes precedence over saved model parameters