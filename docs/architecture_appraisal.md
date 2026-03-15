# Architectural Appraisal — iaq4j

**Date:** 2026-03-12
**Scope:** Full codebase audit across four criteria.
**Test baseline:** 358 tests (329 existing + 29 property-based).

---

## Grades

| Criterion | Grade | Key strength | Key weakness |
|---|---|---|---|
| **Extensibility** | B+ | YAML-driven standards, sensor registry | `build_model()` if/elif chain, no data source registry |
| **DDD Adherence** | B- | Clean bounded contexts, value objects, Merkle provenance | Domain→infrastructure dependency inversion, god object config |
| **Error Taxonomy** | C+ | `DomainErrorCode` enum + `StructuredResponse` exist | No exception hierarchy, 3 error signaling patterns |
| **State Management** | C | Lazy caches well-done, test fixtures cover gaps | 4 mutable globals in main.py, no thread safety |

---

## 1. Extensibility (B+)

### What works

| Extension point | Mechanism | Code change required? |
|---|---|---|
| New sensor | `SensorProfile` ABC + `register_sensor()` | New file only |
| New IAQ standard | Add entry to `iaq_standards.yaml` | **None** |
| New model | Subclass `nn.Module` + `MODEL_REGISTRY` | 2 lines in `models.py` + `build_model()` branch |
| New data source | Subclass `DataSource` ABC | New class + CLI wiring |

The YAML-driven standard registry (`app/standards.py`) requires zero code changes.
The sensor profile registry is nearly as clean.

### What limits it

- **`build_model()` (`app/models.py:305`)** — growing if/elif chain. Each model type
  needs a new branch. A factory dict mapping types to constructor kwargs would close this.
- **Data source selection** — hardcoded string matching in CLI (`iaq4j/__main__.py`).
  No registry analogous to sensors.
- **Pipeline stages** — fixed FSM sequence. Inserting a new stage (e.g. augmentation)
  requires editing the transition table + adding a `_do_*` method. The FSM makes the
  cost clear, but it's not pluggable.

---

## 2. DDD Adherence (B-)

### Bounded contexts (from AGENTS.md)

| Context | Files | Assessment |
|---|---|---|
| **Sensing** | `profiles.py`, `builtin_profiles.py`, `standards.py`, `quantities.py` | Clean |
| **Inference** | `inference.py`, `models.py` | Good |
| **Training** | `pipeline.py`, `data_sources.py`, `merkle.py`, `utils.py` | Excellent |
| **Configuration** | `config.py`, YAMLs | Problematic |
| **API** | `main.py`, `schemas.py` | Fair |

### Dependency inversion violations

Domain modules import infrastructure directly:

```
app/profiles.py:235     → from app.config import settings   ✗
app/schemas.py:49       → from app.config import settings   ✗
training/pipeline.py:14 → from app.config import settings   ✗
```

Domain factories (`get_sensor_profile()`, `get_iaq_standard()`) couple the domain
layer to the Settings singleton. Fix: accept `sensor_type: str` as an optional
parameter, defaulting to config lookup only when omitted.

### God object: `app/config.py` (282 lines)

`Settings` owns API config, model paths, database config, YAML paths, sensor
identity, and caching — all in one class. Should be split into `ModelSettings`,
`DatabaseSettings`, `SensorSettings`.

### Side-effect registration

`import app.builtin_profiles  # noqa: F401` required in `main.py`, `pipeline.py`,
and tests. Forgetting it is a silent failure. DDD would use an explicit composition
root / bootstrap function.

### What DDD gets right

- Value objects are immutable (`SensorReading`, `IAQResponse` via Pydantic)
- `InferenceEngine` is a proper aggregate wrapping `IAQPredictor`
- `StageResult` serves as a domain event in the pipeline FSM
- `DataSource` ABC is a clean repository pattern
- Merkle provenance is a first-class domain concept

---

## 3. Error Taxonomy (C+)

### What exists

- `DomainErrorCode` enum (`app/schemas.py`) — 8 typed codes
- `StructuredResponse` wrapper — `error_code`, `detail`, `warnings`, `next_steps`
- `FailureInfo` dataclass in pipeline — `error_code`, `suggestion`
- `_classify_error()` maps exception messages to codes

### Three competing patterns

| Module | Pattern | Example |
|---|---|---|
| `training/pipeline.py` | Structured exception (`PipelineError` + `FailureInfo`) | `raise PipelineError(str(e), failure_info)` |
| `app/models.py` | Error dict returned | `return {"status": "error", "message": ...}` |
| `app/database.py` | Boolean flag + `last_error` string | `self.connected = False` |

### What's missing

No custom exception hierarchy. The entire codebase uses `ValueError`, `RuntimeError`,
and bare `Exception`. The `DomainErrorCode` enum is defined but only populated in
API-level handlers — not used at the point where errors originate.

### Recommended hierarchy

```python
class IAQError(Exception):
    code: DomainErrorCode
    suggestion: str = None

class NoDataError(IAQError):          # code = NO_DATA
class SchemaMismatchError(IAQError):  # code = SCHEMA_MISMATCH
class TrainingDivergedError(IAQError): # code = TRAINING_DIVERGED
class InsufficientDataError(IAQError): # code = INSUFFICIENT_DATA
class InfluxUnreachableError(IAQError): # code = INFLUX_UNREACHABLE
```

Exceptions carry their own codes. `_classify_error()` becomes unnecessary.
API handlers catch `IAQError` subtypes and map to `StructuredResponse`.

---

## 4. State Management & Testability (C)

### Global mutable state in `app/main.py`

```python
predictors = {}           # Dict[str, IAQPredictor]
inference_engines = {}    # Dict[str, InferenceEngine]
active_model = "mlp"     # mutable global
_pending_mappings = {}    # transient field mapping proposals
```

Four module-level mutable variables. `active_model` mutated via `global active_model`
in a route handler.

**Problems:**

1. **Thread safety** — Uvicorn workers each have independent copies. Async workers
   can read stale `active_model` concurrently.
2. **Testability** — every prediction test must monkeypatch globals.
3. **Transient state loss** — `_pending_mappings` lost on restart. A mapping proposal
   disappears if the server bounces before confirmation.

### Module-level singletons

| Location | State | Risk |
|---|---|---|
| `app/config.py:282` | `settings = Settings()` | Low (immutable after init) |
| `app/database.py` | `influx_manager` singleton | Low (connection state) |
| `app/quantities.py:86` | `_registry = None` (lazy cache) | None |
| `app/standards.py:60` | `_definitions = None` (lazy cache) | None |
| `app/profiles.py:220-221` | `_SENSOR_REGISTRY`, `_STANDARD_REGISTRY` | None |

Lazy caches and registries are fine — effectively immutable after startup.

### Testing impact

Tests work because `monkeypatch` redirects globals extensively:

```python
monkeypatch.setattr(settings, "TRAINED_MODELS_BASE", str(tmp_path))
monkeypatch.setattr(settings, "_model_config_cache", test_cfg)
```

This is brittle. Fix: wrap mutable state in a service class, pass `Settings` as
constructor parameter.

---

## Property-Based Testing (added 2026-03-12)

29 Hypothesis tests in `tests/unit/test_property_based.py` covering:

- Feature engineering invariants (output length, no NaN/Inf, raw feature preservation)
- Cyclical encoding (bounded, unit circle, period wraparound)
- IAQ standard (clamp idempotent, categorize valid, distribution sums)
- Schema fingerprint (deterministic, model-type sensitive)
- Sliding window (shape, target alignment, empty on insufficient data)
- Absolute humidity (positive, finite, monotonic; documents `rh=0` singularity)
- Model forward pass shapes (MLP, CNN, LSTM, BNN)
- Scaler round-trip (StandardScaler, MinMaxScaler)

Shape assertions added to `CNNRegressor.forward()` and `LSTMRegressor.forward()`
to catch silent tensor reinterpretation at runtime.

---

## Prioritised Remediation Requirements

See roadmap items **#R1–R5** in `docs/roadmap.md` § Architectural Remediation.

| # | Item | Priority | Effort | Impact |
|---|---|---|---|---|
| R1 | Domain exception hierarchy | P1 | Small | Unifies 3 error patterns, eliminates `_classify_error()` |
| R2 | Config decomposition | P1 | Medium | Breaks god object, enables DI, improves testability |
| R3 | Service extraction (main.py) | P1 | Medium | Removes 4 mutable globals, enables thread safety |
| R4 | Model factory registry | P2 | Small | Closes `build_model()` if/elif chain |
| R5 | Data source registry | P2 | Small | Closes CLI string matching, enables plugin pattern |
