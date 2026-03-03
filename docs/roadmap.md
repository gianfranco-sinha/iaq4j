# iaq4j Roadmap

Requirements catalog for future implementation. Items are grouped by domain and
marked with priority: **P0** (next up), **P1** (soon), **P2** (later).

---

## Artifact Versioning

### Semantic versioning for models and datasets — P1

**Current state:** Per-model incrementing versions (`mlp-v1`, `mlp-v2`) with
SHA256 data fingerprints and git commit tracking in `MANIFEST.json`.

**Goal:** Layer proper semver (`MAJOR.MINOR.PATCH`) on top of the existing
lineage system so that artifact consumers can reason about compatibility.

**Proposed semantics:**

| Bump  | Trigger |
|-------|---------|
| MAJOR | Breaking change — input schema change (new/removed features), incompatible scaler format, sensor profile change |
| MINOR | Retrained model — same schema, new data or hyperparameters, improved metrics |
| PATCH | Metadata-only — config tweak, documentation, no weight changes |

**Requirements:**

- [x] Version string in `config.json` and `MANIFEST.json` follows `{model_type}-{MAJOR}.{MINOR}.{PATCH}` (e.g. `mlp-2.1.0`)
- [ ] Dataset artifacts get independent semver tied to data fingerprint and preprocessing pipeline version
- [x] Auto-detect bump level: compare current vs previous config.json schema, scaler shape, and data fingerprint
- [x] CLI support: `python -m iaq4j version` to show current active model versions
- [x] Warn (not reject) loading a model whose schema fingerprint doesn't match the serving code's expected schema
- [ ] Migration guide when MAJOR bump occurs (what changed, how to retrain)

---

## Training Run Entity

### Unified lifecycle container for training runs — P2

**Current state:** A single training run produces several disconnected artifacts:
`PipelineResult` (in-memory, discarded after CLI prints), `data_manifest.json`
(provenance), `config.json` (model config + version), `MANIFEST.json` entry
(central registry), `training_history.json` (per-epoch losses), and binary
artifacts (`model.pt`, scaler `.pkl` files). No single object ties these
together as "one run." Each model type shares one directory
(`trained_models/mlp/`) so each run overwrites the last.

**Goal:** A `TrainingRun` entity that captures the full lifecycle — from data
extraction through evaluation — in a single addressable container. Enables
run comparison, reproduction, and rollback.

**What it holds:**

| Field | Source today |
|-------|-------------|
| Run ID | Not assigned — would be `{model_type}-{semver}` or a UUID |
| Input config snapshot | `data_manifest.json → config` |
| Data source metadata | `data_manifest.json → data_source` |
| Data fingerprint | `MANIFEST.json` entry |
| Preprocessing report | `data_manifest.json → preprocessing_issues` |
| Stage timings | `data_manifest.json → stages` |
| Training history | `training_history.json` |
| Evaluation metrics | `config.json` + `MANIFEST.json` |
| Model artifacts | `model.pt`, `feature_scaler.pkl`, `target_scaler.pkl` |
| Version + schema fingerprint | `config.json` + `MANIFEST.json` |

**Key capabilities:**

- **Compare runs** — diff two runs by config, data, metrics without reading
  multiple JSON files
- **Reproduce a run** — config snapshot + data fingerprint + `random_state`
  (now seeded) is sufficient to replay
- **Roll back** — runs stored in versioned directories
  (`trained_models/mlp/1.2.0/`) instead of overwriting a single directory
- **Query** — `python -m iaq4j runs --model mlp` lists all runs with metrics

**Requirements:**

- [ ] `TrainingRun` dataclass that unifies all per-run state into one object
- [ ] Versioned artifact directories (`trained_models/mlp/1.2.0/`) — each run preserved, not overwritten
- [ ] `TrainingRun.save()` / `TrainingRun.load()` for serialization to/from a run directory
- [ ] `PipelineResult` replaced by or wraps `TrainingRun`
- [ ] `python -m iaq4j runs` CLI command to list, compare, and diff runs
- [ ] `IAQPredictor.load_model()` updated to load from versioned directories
- [ ] Rollback support: `python -m iaq4j activate --model mlp --version 1.1.0`

---

## MLflow Integration

### Experiment tracking and model registry — P1

**Current state:** Training runs produce disconnected artifacts (model weights,
scalers, config, manifest, training history) stitched together by custom code.
The Training Run Entity (P2 above) was designed to unify these into a single
addressable container. MLflow does all of this out of the box — and adds
experiment comparison, model registry, artifact versioning, and a web UI.

**Relationship to Training Run Entity:** MLflow would **subsume the Training
Run Entity entirely**. Every field in the proposed `TrainingRun` dataclass
(config snapshot, data fingerprint, metrics, artifacts, stage timings) maps
directly to MLflow's native concepts (params, metrics, artifacts, tags). Rather
than building a custom run container, we adopt the industry standard.

**What MLflow replaces:**

| Current custom component | MLflow equivalent |
|--------------------------|-------------------|
| `MANIFEST.json` central registry | MLflow experiment + run list |
| `training_history.json` per-epoch losses | `mlflow.log_metric()` per step |
| `config.json` metadata (version, fingerprint) | Run params + tags |
| Versioned artifact directories (proposed) | MLflow artifact store |
| `python -m iaq4j runs` (proposed) | MLflow UI + `mlflow.search_runs()` |
| Run comparison (proposed) | MLflow UI compare view |

**What stays custom (MLflow doesn't replace these):**

| Component | Why it stays |
|-----------|-------------|
| Merkle tree data provenance (`training/merkle.py`) | MLflow tracks artifacts but not content-addressable data lineage |
| Sensor profiles / IAQ standards (`app/profiles.py`) | Domain-specific ABCs — no MLflow equivalent |
| Schema fingerprint + semver (`training/utils.py`) | Compatibility detection is domain logic; logged as MLflow tags |
| `TrainingPipeline` FSM (`training/pipeline.py`) | Orchestration logic — MLflow tracks results, not execution |
| Feature engineering code | Per-profile methods — code, not config |

**Integration surface — what gets logged:**

```python
# In training/pipeline.py or training/train.py

with mlflow.start_run(run_name=f"{model_type}-{version}"):
    # Params (static per run)
    mlflow.log_params({
        "model_type": model_type,
        "sensor_type": sensor_profile.name,
        "iaq_standard": iaq_standard.name,
        "window_size": config["window_size"],
        "num_features": num_features,
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "schema_fingerprint": schema_fingerprint,
        "data_fingerprint": data_fingerprint,
        "git_commit": git_commit,
    })

    # Metrics (per epoch)
    for epoch in range(epochs):
        train_loss, val_loss = train_one_epoch(...)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    # Final evaluation metrics
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

    # Artifacts
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("feature_scaler.pkl")
    mlflow.log_artifact("target_scaler.pkl")
    mlflow.log_artifact("data_manifest.json")

    # Tags
    mlflow.set_tags({
        "version": version,
        "schema_fingerprint": schema_fingerprint,
        "merkle_root": merkle_root,
    })
```

**Open design questions (to decide before implementation):**

1. **Deployment mode:** Local file-based (`mlruns/` directory) vs MLflow Tracking
   Server (SQLite/Postgres backend, S3/local artifact store). Local is simplest
   to start; server enables remote access and the web UI.

2. **Layer vs replace:** Should MLflow run alongside `MANIFEST.json` (additive
   layer, low risk) or replace it entirely (cleaner, but migration needed)?
   Recommendation: layer first, then deprecate MANIFEST once stable.

3. **Model registry:** Use MLflow Model Registry for promotion
   (Staging → Production) or keep the current `trained_models/` directory
   as the serving source? Registry adds governance but also complexity.

4. **Artifact store:** Local filesystem vs S3-compatible (MinIO on the Pi)?
   Local is fine for single-node; MinIO enables backup and remote access.

5. **IAQPredictor loading:** Should `IAQPredictor.load_model()` load from
   MLflow artifact store or keep loading from `trained_models/`? Could
   support both with a config flag.

**Requirements:**

- [ ] Add `mlflow` to dependencies
- [ ] Wrap `train_single_model()` with `mlflow.start_run()` context manager
- [ ] Log params: model config, sensor type, IAQ standard, schema fingerprint, data fingerprint, git commit
- [ ] Log metrics per epoch: train_loss, val_loss
- [ ] Log final metrics: MAE, RMSE, R² (or whatever evaluation produces)
- [ ] Log artifacts: model.pt, scalers, data_manifest.json
- [ ] Log tags: version (semver), schema_fingerprint, merkle_root
- [ ] Decide deployment mode (local file vs tracking server)
- [ ] Decide layer vs replace strategy for MANIFEST.json
- [ ] Update `python -m iaq4j runs` to query MLflow instead of (or in addition to) MANIFEST
- [ ] Evaluate MLflow Model Registry for model promotion workflow
- [ ] Update standalone training scripts (`train_models.py`, `train_all_models.py`) to log to MLflow

---

## Sensor Onboarding

### LLM-driven semantic field mapping from firmware API to sensor profile — P1

**Current state:** `SensorProfile.raw_features` defines canonical internal names
(e.g. `temperature`, `voc_resistance`). Clients must send data using exactly
these names. Adding a new sensor or firmware variant requires manually writing
the field mapping.

**Goal:** Use a small LLM as a one-time onboarding tool that reads a firmware
API spec (OpenAPI doc, header file, example JSON payload) and produces a
semantic mapping from firmware field names to the `SensorProfile`'s internal
feature names. The mapping is persisted in config — no LLM in the hot path.

**Supported input sources:**

**Two primary input sources:**

**Source A: CSV file with sensor readings** — the most common path. User
uploads a CSV file containing column headers from their sensor. The mapper
reads the headers, samples values (to infer units and ranges), and maps each
column to a quantity in `quantities.yaml`.

```
Upload: my_sensor_data.csv
Headers: timestamp, comp_temp, comp_hum, comp_press, gas_res, static_iaq
Sample:  2026-01-15T10:00:00, 23.5, 55.2, 1013.0, 85000, 42

Mapper output:
  comp_temp    → temperature (°C, confidence 0.95, fuzzy match)
  comp_hum     → relative_humidity (%RH, confidence 0.92, fuzzy match)
  comp_press   → barometric_pressure (hPa, confidence 0.90, fuzzy match)
  gas_res      → voc_resistance (Ω, confidence 0.70, LLM match)
  static_iaq   → bsec_iaq (index, confidence 0.85, LLM match)
```

**Source B: Device profile on GitHub** — the firmware repo documents the
sensor's API fields in struct definitions, JSON schemas, README docs, or
example payloads. The mapper crawls the repo to extract field metadata.

| Source | Example | What the mapper extracts |
|--------|---------|--------------------------|
| CSV with headers | `--source data.csv` | Column names, sampled values for unit/range inference |
| GitHub repo | `--source https://github.com/org/firmware` | Struct defs, JSON schemas, API routes, README field docs |
| Example JSON payload | `--source payload.json` | Field names, value types, example values |
| OpenAPI / header file | `--source api.yaml` or `sensor.h` | Field names, types, descriptions |
| Raw URL | `--source https://docs.example.com/api` | Fetches and parses page content |

For CSV files, the mapper:
1. Reads column headers as candidate field names
2. Samples N rows to infer value ranges, types, and likely units
3. Cross-references against `quantities.yaml` (valid ranges, units)
4. Feeds results to the tiered matching pipeline

For GitHub repos, the mapper uses `gh` CLI or the GitHub API to:
1. List repo contents and identify relevant files (`.h`, `.c`, `.json`,
   `.yaml`, `README.md`, API route handlers)
2. Extract struct/typedef definitions and JSON serialization fields
3. Parse any example payloads or test fixtures
4. Feed all discovered field metadata to the matching tiers

**How it works (both sources):**

1. User provides a CSV file or device profile reference (GitHub repo, URL, file)
2. Mapper extracts field names and metadata from the source
3. Tier 1+2 (exact/fuzzy) resolve obvious matches against `quantities.yaml`
   entries — using field names, sampled value ranges, and unit inference
4. Remaining ambiguous fields go to Tier 3 (LLM) with full context from both
   the source and the quantity registry
5. LLM outputs candidate mapping with confidence scores and reasoning
6. User confirms or adjusts interactively
7. Mapping saved to `model_config.yaml` under `sensor.field_mapping`
8. At ingestion time, a lightweight translation layer applies the static mapping
   before the reading hits `SensorReading` validation

**Example mapping output:**

```yaml
sensor:
  type: bme680
  field_mapping:
    comp_temp: temperature
    comp_hum: rel_humidity
    comp_press: pressure
    gas_resistance_ohm: voc_resistance
```

**Impact on existing code:**

| Component | Change |
|-----------|--------|
| `SensorProfile` ABC | Add optional `field_descriptions` property (units, physical meaning) so the LLM has structured context to reason against |
| `SensorReading` schema | Apply `field_mapping` translation in `_build_readings()` validator before feature-name matching |
| `model_config.yaml` | New optional `sensor.field_mapping` section |
| Training data ingestion | Apply same mapping when reading from InfluxDB (column rename at fetch time) |
| Inference hot path | Zero cost — mapping resolved to a static dict at startup, simple key rename |

**Mapping backend — pluggable, tiered strategy:**

The mapper uses a tiered approach: deterministic matching first, LLM only for
ambiguous fields. The LLM backend itself is pluggable (local or cloud).

| Tier | Method | When it fires | Dependency |
|------|--------|---------------|------------|
| 1 | **Exact match** | Firmware field name equals internal name | None |
| 2 | **Fuzzy match** | `rapidfuzz` similarity > 0.85 + unit/range validation against `field_descriptions` | `rapidfuzz` (pip) |
| 3 | **LLM (local)** | Ambiguous fields — Ollama with Phi-3-mini or Llama 3.2 3B | Ollama running locally (optional) |
| 3 | **LLM (cloud)** | Same trigger — Claude Haiku or equivalent | API key (optional) |

- Tier 1+2 handle the common case (trivial renames, casing differences) with
  zero LLM cost
- Tier 3 only fires for genuinely different naming (`comp_gas` → `voc_resistance`)
- User picks backend via `--backend fuzzy|ollama|anthropic` flag (default: `fuzzy`,
  falls back to LLM if any field is unresolved)
- Ollama is called via HTTP (`localhost:11434/api/generate`) — no Python SDK needed,
  just `httpx` which is already a FastAPI dependency

**Done (implemented):**

- [x] `SensorProfile.field_descriptions` property on ABC with default empty dict
- [x] `BME680Profile.field_descriptions` with unit, description, example per feature

**Requirements:**

- [x] CLI command: `python -m iaq4j map-fields --source <file_or_url> [--backend fuzzy|ollama]`
- [ ] GitHub repo input: crawl repo via `gh` CLI for `.h`/`.c` structs, JSON schemas, example payloads, and README field docs
- [x] REST endpoint for sensor registration (see Sensor Registration API below)
- [x] Tier 1: exact name match (case-insensitive, strip underscores/hyphens)
- [x] Tier 2: fuzzy match via `rapidfuzz` + validate against `valid_ranges` and `field_descriptions` units
- [x] Tier 3 (Ollama): prompt template sends `field_descriptions` + firmware fields, expects JSON mapping back
- [ ] Tier 3 (Cloud): same prompt, via Anthropic Messages API with Haiku
- [x] Interactive confirmation: show mapping table with confidence, let user override
- [x] Confirmed mapping persisted in `model_config.yaml` under `sensor.field_mapping`
- [x] `SensorReading._build_readings()` applies `field_mapping` at validation time
- [x] Training data sources apply the same mapping at column-rename time
- [x] Fallback: if no mapping configured, current behavior (exact names) is unchanged
- [ ] `rapidfuzz` added as optional dependency (`pip install iaq4j[mapping]`)
- [ ] Ollama and Anthropic SDK are optional — graceful error if not available and selected

---

## Multi-Device Support

### Central physical quantity registry for units and denomination — P0

**Current state:** Units and valid ranges are defined per `SensorProfile`
subclass (`field_descriptions`, `valid_ranges`). With a single sensor (BME680)
this works, but with multiple devices it creates problems:

- Two sensors measuring the same quantity (e.g. temperature) may use different
  units (°C vs °F) — the model expects one canonical unit
- `valid_ranges` are unit-dependent and duplicated per profile
- Feature engineering formulas (e.g. absolute humidity) silently assume specific
  units
- The semantic field mapper has no way to know two fields represent the same
  physical quantity in different units

**Goal:** A central YAML-based registry of physical quantities that defines
canonical units, valid ranges, and accepted alternate units. Each sensor
profile maps its raw features to quantities in this table. The same table
is used by the semantic field mapper to match firmware API fields by physical
meaning, not just name.

**Design: YAML as the source of truth**

The registry lives in `quantities.yaml` alongside the other config files.
Python code (`app/quantities.py`) loads it at startup and exposes typed
lookup + unit conversion. No physical quantities are hardcoded in Python.

```yaml
# quantities.yaml — central physical quantity registry

quantities:
  temperature:
    canonical_unit: "°C"
    symbol: "T"
    description: "Ambient temperature"
    valid_range: [-40, 85]
    alternate_units:
      "°F":
        convert: "({value} - 32) * 5/9"
      "K":
        convert: "{value} - 273.15"

  relative_humidity:
    canonical_unit: "%RH"
    symbol: "RH"
    description: "Relative humidity"
    valid_range: [0, 100]

  barometric_pressure:
    canonical_unit: "hPa"
    symbol: "P"
    description: "Barometric pressure"
    valid_range: [300, 1100]
    alternate_units:
      "Pa":
        convert: "{value} / 100"
      "kPa":
        convert: "{value} * 10"
      "mbar":
        convert: "{value}"             # 1:1 with hPa
      "inHg":
        convert: "{value} * 33.8639"

  voc_resistance:
    canonical_unit: "Ω"
    symbol: "R_VOC"
    description: "MOX sensor resistance to volatile organic compounds"
    valid_range: [1000, 2000000]
    alternate_units:
      "kΩ":
        convert: "{value} * 1000"
      "MΩ":
        convert: "{value} * 1000000"

  co2:
    canonical_unit: "ppm"
    symbol: "CO2"
    description: "Carbon dioxide concentration"
    valid_range: [400, 5000]

  pm2_5:
    canonical_unit: "µg/m³"
    symbol: "PM2.5"
    description: "Particulate matter ≤ 2.5µm"
    valid_range: [0, 500]

  tvoc:
    canonical_unit: "ppb"
    symbol: "TVOC"
    description: "Total volatile organic compounds"
    valid_range: [0, 60000]

  # --- IAQ indices (computed by sensor firmware / external libraries) ---
  # These are both sensor outputs AND prediction targets.

  bsec_iaq:
    canonical_unit: "index"
    symbol: "IAQ"
    description: "Bosch BSEC Indoor Air Quality index"
    valid_range: [0, 500]
    kind: "iaq_standard"              # marks this as a prediction target, not just a sensor input

  epa_aqi:
    canonical_unit: "index"
    symbol: "AQI"
    description: "US EPA Air Quality Index"
    valid_range: [0, 500]
    kind: "iaq_standard"
```

**IAQ quantities serve dual roles:**

IAQ indices like `bsec_iaq` are both sensor outputs (the firmware computes
them) and prediction targets (the model learns to reproduce them). This means:

- They appear in `quantities.yaml` alongside sensor readings — same unit,
  range, and description metadata
- The `iaq_actual` field on `SensorReading` maps to a quantity in the table
  (e.g. `bsec_iaq`), enabling the mapper to match firmware fields like
  `static_iaq` or `iaq_score` to the right quantity
- `IAQStandard` subclasses derive their `scale_range` from the YAML table
  but still own behavioral logic in Python: category breakpoints, clamping,
  and `categorize()` — these are domain rules, not data
- Comparing predicted vs actual IAQ uses the same quantity definition, ensuring
  consistent scale and units across evaluation

**SensorProfile changes:**

```python
# Each profile maps its raw features to quantities from the YAML table:
class BME680Profile(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temperature":    "temperature",           # feature name → quantity name
            "rel_humidity":   "relative_humidity",
            "pressure":       "barometric_pressure",
            "voc_resistance": "voc_resistance",
        }

# A sensor reporting Fahrenheit — the quantity registry handles conversion:
class SomeOtherSensor(SensorProfile):
    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temp_f": "temperature",    # maps to same quantity
        }

    @property
    def feature_units(self) -> Dict[str, str]:
        return {
            "temp_f": "°F",             # not canonical — auto-converted at ingestion
        }
```

**How the mapper uses it:**

The semantic field mapper matches firmware fields against `quantities.yaml`
by description, unit, and valid range — not just field name. When a firmware
field has unit "°F" and range [32, 185], the mapper can confidently match it
to the `temperature` quantity even if the firmware calls it `comp_t`.

**Impact on existing code:**

| Component | Change |
|-----------|--------|
| New `quantities.yaml` | YAML table of all known physical quantities with canonical units, ranges, and conversions |
| New `app/quantities.py` | Loads YAML, exposes `get_quantity()`, `convert_to_canonical()` |
| `SensorProfile` ABC | Add `feature_quantities` property mapping features → quantity names; `valid_ranges` and `field_descriptions` become computed from the YAML registry |
| `BME680Profile` | Replace hardcoded `valid_ranges`/`field_descriptions` with `feature_quantities` mapping |
| Feature engineering | Conversion to canonical units applied before formulas |
| Semantic field mapper | Matches by physical quantity (description + unit + range), not just field name |
| `SensorReading` validation | Range checks derived from YAML registry in canonical units |
| Backward compat | `valid_ranges` and `field_descriptions` still work as computed properties |

**Requirements:**

- [ ] `quantities.yaml` with all known physical quantities, canonical units, valid ranges, and alternate unit conversions
- [ ] `app/quantities.py` loads YAML, exposes `get_quantity()`, `convert_to_canonical(value, from_unit, quantity_name)`
- [ ] Conversion expressions in YAML evaluated safely (no `eval` — use a simple expression parser or lookup table)
- [ ] `SensorProfile.feature_quantities` property maps feature names → quantity names from the YAML table
- [ ] `valid_ranges` and `field_descriptions` become computed from `feature_quantities` + YAML registry (backward compat)
- [ ] Conversion applied transparently at ingestion time (before feature engineering)
- [ ] Semantic mapper uses quantity identity + unit + range for cross-device field matching
- [ ] Registry extensible: add new quantities by editing YAML, no Python changes needed

---

## Sensor Registration API

### REST endpoint for sensor onboarding and field mapping — P1

**Depends on:** Semantic field mapping, Physical quantity registry

**Goal:** Expose the sensor onboarding workflow as REST endpoints so it can be
driven from a web UI, CI pipeline, or external tooling — not only the CLI.

**Proposed endpoints:**

```
POST   /sensors/register
       Accepts a firmware source and returns a proposed field mapping.

       Request body:
       {
         "source_type": "github_repo" | "api_spec_url" | "example_payload",
         "source": "https://github.com/org/firmware",   // or URL, or inline JSON
         "sensor_name": "my_bme680",                     // optional human label
         "backend": "fuzzy"                              // fuzzy | ollama | anthropic
       }

       Response:
       {
         "mapping_id": "abc123",
         "status": "proposed",
         "mapping": {
           "comp_temp":           {"internal": "temperature",    "quantity": "temperature",       "confidence": 0.98, "method": "fuzzy"},
           "comp_hum":            {"internal": "rel_humidity",   "quantity": "relative_humidity",  "confidence": 0.95, "method": "fuzzy"},
           "gas_resistance_ohm":  {"internal": "voc_resistance", "quantity": "voc_resistance",     "confidence": 0.72, "method": "llm"}
         },
         "unresolved": [],
         "source_fields_found": ["comp_temp", "comp_hum", "comp_press", "gas_resistance_ohm"]
       }

POST   /sensors/register/{mapping_id}/confirm
       User confirms or overrides the proposed mapping.

       Request body:
       {
         "overrides": {
           "gas_resistance_ohm": "voc_resistance"        // manual correction if needed
         }
       }

       Response:
       {
         "status": "confirmed",
         "sensor_name": "my_bme680",
         "field_mapping": { ... },                        // final mapping
         "persisted_to": "model_config.yaml"
       }

GET    /sensors
       List registered sensors and their active field mappings.

DELETE /sensors/{sensor_name}
       Remove a registered sensor and its mapping.
```

**Design notes:**

- `POST /sensors/register` is async-friendly: repo crawl + LLM call may take
  seconds, so return a `mapping_id` the client can poll or the response can
  be streamed
- The same tiered matching logic (exact → fuzzy → LLM) is shared between CLI
  and API — extracted into a reusable `FieldMapper` service class
- Confirmed mappings persist to `model_config.yaml` under `sensor.field_mapping`
  (same location the CLI uses)
- Multiple sensors can be registered simultaneously for multi-device deployments

**Requirements:**

- [x] `FieldMapper` service class extracted from CLI logic (shared by CLI + API)
- [x] `POST /sensors/register` endpoint — accepts source, runs tiered mapping, returns proposal
- [x] `POST /sensors/register/{mapping_id}/confirm` — persists confirmed mapping to config
- [x] `GET /sensors` — list active field mapping
- [x] `DELETE /sensors/mapping` — remove field mapping
- [x] Proposed mappings stored in memory until confirmed
- [ ] Background task support for slow operations (repo crawl, LLM inference)

---

## Security Hardening

### Production security improvements — P1

**Current state:** Service is behind nginx reverse proxy but has several gaps.

**Done:**

- [x] Docker port bound to localhost only (`127.0.0.1:8001:8000`) — external
  traffic must go through nginx

**Requirements:**

- [ ] HTTPS via Let's Encrypt / certbot on nginx for `enviro-sensors.uk`
- [ ] API key authentication — require `X-API-Key` header on `/predict` and mutation endpoints
- [ ] Rate limiting on nginx (`limit_req_zone`) to prevent abuse
- [ ] Tighten CORS — replace `allow_origins=["*"]` with explicit allowed origins
- [ ] Add `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security` headers in nginx
- [ ] Read-only filesystem in Docker container (`read_only: true` with tmpfs for `/tmp`)

---

## Telemetry Integrity

### Per-sensor sequence numbers for ordering and replay detection — P2

**Background:** Comparison against the IAQSignedTelemetry domain model
identified sequence numbers as a meaningful gap. The domain model requires
a monotonically increasing `sequence_number` per `sensor_id` as a first-class
invariant. iaq4j currently has no ordering or replay protection at the
ingestion boundary.

**What sequence numbers give us:**

| Problem | Without sequence number | With sequence number |
|---------|------------------------|---------------------|
| Dropped readings | Silent — gap in InfluxDB time series is indistinguishable from normal | Detectable — gap in sequence means missed readings |
| Reordered delivery | Silent — out-of-order write lands in InfluxDB at wrong timestamp | Detectable — sequence regresses |
| Replay attack | Identical duplicate is accepted and written | Rejected — sequence already seen for this sensor |
| Sensor restart | No signal | Sequence resets to 0, can be flagged as an event |

**Scope:** This is relevant today only if sensors are networked (MQTT, HTTP
push over the internet). For a local BME680 read over I²C on the same Pi, it
adds no value. Implement when remote or multi-sensor deployments are in scope.

**Proposed design:**

- Add optional `sequence_number: int` field to `SensorReading`
- `InferenceEngine` (or a new ingestion layer) tracks last-seen sequence per
  `sensor_id` in memory (and optionally persists to InfluxDB)
- On receipt: if `sequence_number <= last_seen`, log a warning and optionally
  reject (configurable — warn-only vs strict mode)
- Sequence resets (sensor restart) detected as large backward jump; logged
  as an event rather than an error
- No sequence number present → accepted without check (backward compat,
  local sensor mode)

**Requirements:**

- [ ] Add optional `sequence_number: int` to `SensorReading` schema
- [ ] `InferenceEngine` tracks last-seen sequence per `sensor_id`
- [ ] Configurable enforcement mode: `off` (default) | `warn` | `strict`
- [ ] Sequence gaps and regressions logged as structured warnings in `IAQResponse`
- [ ] Sequence state persisted across restarts (InfluxDB tag or local file)
- [ ] `GET /sensors/{sensor_id}/sequence` endpoint to inspect current state

---

## LLM Readiness

### Infrastructure prerequisites for agentic orchestration — P1

**Current state:** The codebase has strong machine-readable foundations — the
training pipeline emits structured `PipelineResult`/`StageResult`, sensor
profiles are self-describing ABCs, the field mapper returns confidence-scored
proposals, and model artifacts carry version + fingerprint metadata. However,
several gaps prevent an LLM agent (or any external orchestrator) from
reliably driving the system end-to-end.

**Goal:** Close the gap between "ML platform with a CLI" and "ML platform an
agent can operate." Every component that the agent needs must be accessible
via REST, return structured data, and fail with actionable error information.

**Codebase assessment — what's already agent-ready:**

| Component | File | Score | Why |
|-----------|------|-------|-----|
| TrainingPipeline | `training/pipeline.py` | 9/10 | Structured `PipelineResult`, callbacks, `StageResult` with timing |
| Sensor Profiles | `app/profiles.py` | 9/10 | Self-describing ABCs, `feature_quantities`, `valid_ranges` |
| Field Mapper | `app/field_mapper.py` | 9/10 | 3-tier strategy, confidence scores, method provenance |
| Quantities Registry | `quantities.yaml` | 9/10 | Central, aliases, valid_ranges, unit conversions |
| MANIFEST.json | `trained_models/` | 9/10 | Version, fingerprints, metrics, git_commit |
| IAQResponse schema | `app/schemas.py` | 8/10 | Bayesian structure: observation, predicted, prior, uncertainty |

**What needs fixing — 7 gap areas:**

---

#### Phase 1: Machine-Readable Foundation

Close internal gaps so that every component returns structured, actionable
information. No new REST endpoints yet — these are plumbing fixes.

**A. Config cache stale after writes**

`app/config.py` never invalidates `_model_config_cache` after endpoints write
to YAML. An agent (or any caller) that writes config then reads it back gets
stale values.

| Change | File | What |
|--------|------|------|
| Add `invalidate_config_cache()` | `app/config.py` | Clear `_model_config_cache` dict |
| Call after YAML writes | `app/main.py` | Every endpoint that writes to `model_config.yaml` calls invalidate |

**B. No InfluxDB read access**

`app/database.py` only has `write_prediction()`. An agent can't query
prediction history, compare predicted vs actual IAQ, or analyze trends.

| Change | File | What |
|--------|------|------|
| `query_predictions()` | `app/database.py` | Read back predictions by time range, model type |
| `query_raw_readings()` | `app/database.py` | Read raw sensor readings |
| `query_prediction_vs_actual()` | `app/database.py` | Join predicted and actual IAQ for evaluation |

**C. Training exceptions swallowed**

`training/train.py:train_single_model()` catches all exceptions and returns
`None`. An agent can't distinguish "failed because data was empty" from
"failed because GPU OOM" from "succeeded."

| Change | File | What |
|--------|------|------|
| Re-raise as `PipelineError` | `training/train.py` | Typed exception with stage, error code, traceback |
| `FailureInfo` schema | `training/pipeline.py` | Structured failure metadata (stage, cause, suggestion) |

**D. Standardized error model**

API endpoints return ad-hoc error dicts. An agent needs a consistent schema
to parse failures programmatically.

| Change | File | What |
|--------|------|------|
| `APIError` Pydantic model | `app/schemas.py` | `error_code`, `detail`, `timestamp`, optional `context` dict |
| Exception handlers | `app/main.py` | Map known exceptions to `APIError` responses |

**Requirements (Phase 1):**

- [ ] `app/config.py`: add `invalidate_config_cache()` method; call from all YAML-writing endpoints
- [ ] `app/database.py`: add `query_predictions(time_range, model_type)` returning list of dicts
- [ ] `app/database.py`: add `query_raw_readings(time_range)` returning list of dicts
- [ ] `app/database.py`: add `query_prediction_vs_actual(time_range, model_type)` for evaluation
- [ ] `training/train.py`: stop catching all exceptions; re-raise as typed `PipelineError`
- [ ] `training/pipeline.py`: add `FailureInfo` dataclass with stage, error_code, suggestion
- [ ] `app/schemas.py`: add `APIError` Pydantic model with `error_code`, `detail`, `timestamp`
- [ ] `app/main.py`: add exception handlers that return structured `APIError` responses

---

#### Phase 2: Agent Tool Surface

Expose every component the agent needs as a REST endpoint. Each endpoint
returns structured JSON that maps 1:1 to an agent tool.

**New endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `GET /profiles/sensors` | List registered sensor profiles with features, quantities, ranges |
| `GET /profiles/standards` | List IAQ standards with categories and scale ranges |
| `GET /quantities` | Physical quantity registry (names, units, aliases, ranges) |
| `GET /models/{type}/info` | Full artifact metadata: version, fingerprint, metrics, data lineage |
| `GET /models/manifest` | Central MANIFEST.json with all training runs |
| `GET /history/predictions` | Query historical predictions from InfluxDB (time range, model type) |
| `GET /history/readings` | Query historical raw sensor readings |
| `GET /history/accuracy` | Predicted vs actual IAQ comparison |
| `GET /config` | Active model + database config (secrets redacted) |
| `POST /training/start` | Trigger training as background task, return `job_id` |
| `GET /training/{job_id}` | Poll training job status and results |
| `GET /audit` | Queryable mutation audit trail (model switches, config writes) |

**Consistency fix:**

| Endpoint | Change |
|----------|--------|
| `GET /predict/compare` | Return structured `IAQResponse` per model instead of flat dicts |

**Requirements (Phase 2):**

- [ ] `GET /profiles/sensors` — serialize sensor profile ABCs (features, quantities, ranges, field descriptions)
- [ ] `GET /profiles/standards` — serialize IAQ standards (categories, scale range, breakpoints)
- [ ] `GET /quantities` — serve `quantities.yaml` as structured JSON
- [ ] `GET /models/{type}/info` — load `config.json` + `MANIFEST.json` entry for model type
- [ ] `GET /models/manifest` — serve full `MANIFEST.json`
- [ ] `GET /history/predictions` — query InfluxDB via new `query_predictions()` (Phase 1)
- [ ] `GET /history/readings` — query InfluxDB via new `query_raw_readings()` (Phase 1)
- [ ] `GET /history/accuracy` — query InfluxDB via new `query_prediction_vs_actual()` (Phase 1)
- [ ] `GET /config` — serialize active settings with secrets redacted
- [ ] `POST /training/start` — launch `train_single_model()` in background task, return job ID
- [ ] `GET /training/{job_id}` — return job status (pending/running/completed/failed) and results
- [ ] `GET /audit` — queryable log of mutations with timestamps
- [ ] Fix `/predict/compare` to return `Dict[str, IAQResponse]` instead of flat dicts
- [ ] OpenAPI schema includes all new endpoints with typed request/response models

---

#### Phase 3: Agent Loop

Phase 3 is the existing **LLM Agent** roadmap item below. It depends on
Phase 1 (structured internals) and Phase 2 (REST tool surface) being
complete. The agent's tool registry maps 1:1 to Phase 2 endpoints.

---

## LLM Agent

### Agentic orchestrator for data ingestion, training, and insight — P2

**Goal:** An embedded LLM agent that acts as the system's orchestration layer.
Users express intent in natural language; the agent coordinates data ingestion,
field mapping, training, evaluation, and reporting by calling existing
components as tools.

**Depends on:** LLM Readiness Phase 1 + Phase 2 (above), Semantic field mapping

**Prerequisites from LLM Readiness (must be complete before starting):**

- Phase 1: config cache invalidation, InfluxDB read access, training exception
  propagation, standardized `APIError` model
- Phase 2: all `GET /profiles/*`, `GET /quantities`, `GET /models/*`,
  `GET /history/*`, `GET /config`, `POST /training/start`,
  `GET /training/{job_id}`, `GET /audit` endpoints operational

**Core capabilities:**

**1. Universal data ingestion**
- Accept sensor data from any source: file upload (CSV, JSON, Excel),
  URL (raw file, API endpoint), or GitHub repo
- LLM inspects the data, identifies columns, infers physical quantities
  and units by cross-referencing `quantities.yaml`
- Proposes a field mapping, user confirms, data is normalized to canonical
  units and available as a `DataSource` for training

**2. Pipeline orchestration**
- User says "train a model on this data" — agent coordinates the full
  workflow: ingest → map → validate → train → evaluate → report
- Long-running operations (training) run async with progress updates
- Agent decides model type, hyperparameters, or asks user when ambiguous
- Compares new model against existing versions and recommends promotion

**3. Diagnostic reasoning**
- Interpret sensor drift, model divergence, preprocessing issues
- "Why did IAQ spike at 3am?" — agent queries InfluxDB history, correlates
  features, explains in natural language
- Compare `iaq_predicted` vs `iaq_actual` and diagnose systematic errors

**4. Conversational query interface**
- `POST /agent/ask` — natural language questions about air quality, model
  performance, data quality, sensor health
- Agent has read access to InfluxDB history, model manifests, training
  reports, and live predictions

**Tool registry — each tool maps 1:1 to a Phase 2 REST endpoint:**

| Tool | REST Endpoint | Purpose |
|------|---------------|---------|
| `inspect_data` | `POST /data/upload`, `POST /data/import` | Read file/URL, sample rows, infer schema and units |
| `map_fields` | `POST /sensors/register` | Semantic field mapper (exact → fuzzy → LLM) |
| `get_profiles` | `GET /profiles/sensors`, `GET /profiles/standards` | List sensor profiles and IAQ standards |
| `get_quantities` | `GET /quantities` | Physical quantity registry |
| `train_model` | `POST /training/start` | Launch training as background task |
| `poll_training` | `GET /training/{job_id}` | Check training job status and results |
| `get_model_info` | `GET /models/{type}/info`, `GET /models/manifest` | Artifact metadata, versions, metrics |
| `query_history` | `GET /history/predictions`, `GET /history/readings` | InfluxDB reads |
| `evaluate_accuracy` | `GET /history/accuracy` | Predicted vs actual IAQ comparison |
| `get_config` | `GET /config` | Active model + database config (redacted) |
| `get_audit` | `GET /audit` | Queryable mutation trail |
| `manage_models` | `POST /model/select`, existing endpoints | Switch active, compare versions |
| `manage_sensors` | `GET /sensors`, `DELETE /sensors/mapping` | List/remove field mappings |

**Proposed endpoints:**

```
POST   /agent/ask
       Natural language query or command.
       { "message": "Train an MLP on this CSV", "file": <upload> }
       Returns: streaming response with reasoning + tool calls + result

POST   /data/upload
       Upload sensor data file for LLM-assisted ingestion.
       Returns: proposed field mapping + data summary

POST   /data/import
       Import from URL (CSV link, API endpoint, GitHub repo).
       { "url": "https://github.com/user/sensor-data", "format": "auto" }
       Returns: proposed field mapping + data summary

GET    /agent/tasks
       List running/completed agent tasks (training jobs, data imports).
```

**Architecture:**

- LLM backend: pluggable (Ollama local / Anthropic cloud), same as field
  mapper — configured via `model_config.yaml`
- Agent runs in a background task (FastAPI `BackgroundTasks` or Celery)
  for long operations
- Tool calls are structured function calls — the LLM decides which tools
  to invoke and in what order based on user intent
- All tool results are logged for audit trail
- Prediction hot path (`/predict`) is unchanged — zero LLM latency on
  real-time inference

**Requirements:**

- [ ] Tool registry: wrap existing components as callable tools with typed input/output schemas
- [ ] Agent loop: LLM receives user message + tool definitions, emits tool calls, receives results, iterates
- [ ] `POST /agent/ask` endpoint with streaming response
- [ ] `POST /data/upload` — file upload + LLM-assisted schema inspection and mapping
- [ ] `POST /data/import` — URL/repo import with auto-detection
- [ ] `ExternalDataSource` class implementing `DataSource` ABC for user-provided data
- [ ] Background task management for long-running agent operations
- [ ] Conversation history (per-session) for multi-turn interactions
- [ ] Ollama / Anthropic backend selection via config
- [ ] Guardrails: agent can read data and trigger training but cannot modify production model without user confirmation

---
