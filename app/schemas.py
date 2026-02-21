# ============================================================================
# File: app/schemas.py
# ============================================================================
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, model_validator


class SensorReading(BaseModel):
    """Sensor reading input.

    Accepts either a generic ``readings`` dict or the legacy BME680 keyword
    fields (temperature, rel_humidity, pressure, voc_resistance).  Both
    formats are supported for backward compatibility.
    """
    readings: Optional[Dict[str, float]] = Field(
        None, description="Sensor readings keyed by feature name"
    )
    prior_variables: Optional[Dict[str, float]] = Field(
        None, description="External prior variables (e.g. {\"presence\": 1.0})"
    )
    iaq_actual: Optional[float] = Field(
        None, description="Actual IAQ score from sensor (e.g. BSEC IAQ)"
    )
    timestamp: Optional[str] = Field(None, description="ISO timestamp")

    # Legacy BME680 fields — populated into ``readings`` if present
    temperature: Optional[float] = Field(None, exclude=True)
    rel_humidity: Optional[float] = Field(None, exclude=True)
    pressure: Optional[float] = Field(None, exclude=True)
    voc_resistance: Optional[float] = Field(None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _build_readings(cls, values):
        """Apply field_mapping then merge legacy fields into readings dict."""
        if isinstance(values, dict):
            # Apply field mapping if configured
            from app.config import settings
            cfg = settings.load_model_config()
            field_mapping = cfg.get("sensor", {}).get("field_mapping", {})
            if field_mapping:
                readings = values.get("readings")
                if readings and isinstance(readings, dict):
                    values["readings"] = {
                        field_mapping.get(k, k): v for k, v in readings.items()
                    }
                else:
                    for ext, internal in field_mapping.items():
                        if ext in values and values[ext] is not None:
                            values[internal] = values.pop(ext)

            # Merge legacy fields into readings when readings is absent
            if values.get("readings") is None:
                legacy_keys = ["temperature", "rel_humidity", "pressure", "voc_resistance"]
                legacy = {k: values[k] for k in legacy_keys if k in values and values[k] is not None}
                if legacy:
                    values["readings"] = legacy
        return values

    def get_readings(self) -> Dict[str, float]:
        """Return the sensor readings dict (always populated after validation)."""
        return self.readings or {}


# ---------------------------------------------------------------------------
# Bayesian inference response structure
# ---------------------------------------------------------------------------

class PriorVariableEffect(BaseModel):
    """Effect of a single prior variable on the predictive distribution."""
    variable: str
    value: float
    state: str = Field(description="Resolved state (e.g. 'true' or 'false')")
    target_shift: float
    prior_std: float
    description: Optional[str] = None


class BayesianUpdate(BaseModel):
    """Result of applying Gaussian conjugate update from prior variables."""
    pre_mean: float
    pre_std: float
    post_mean: float
    post_std: float
    variables_applied: List[PriorVariableEffect]


class Observation(BaseModel):
    """Direct sensor measurements — the evidence conditioning our inference."""
    sensor_type: str
    readings: Dict[str, float]
    engineered_features: Optional[Dict[str, float]] = None
    timestamp: Optional[str] = None


class UncertaintyEstimate(BaseModel):
    """Quantified uncertainty around the predicted value."""
    std: float
    ci_lower: float = Field(description="Lower bound of 95% credible interval")
    ci_upper: float = Field(description="Upper bound of 95% credible interval")
    method: str = Field(description="mc_dropout | weight_sampling | history_std | deterministic")


class Predicted(BaseModel):
    """The model's predicted value for the latent IAQ variable given the evidence."""
    mean: float
    category: str
    uncertainty: Optional[UncertaintyEstimate] = None
    iaq_standard: str = "bsec"


class Prior(BaseModel):
    """Belief about IAQ before this observation — from recent history or training distribution."""
    mean: float
    std: float
    source: str = Field(description="history_window | training_distribution")
    n_observations: int


class InferenceMetadata(BaseModel):
    """How the inference was performed."""
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    window_size: int
    buffer_size: int
    uncertainty_method: Optional[str] = None
    mc_samples: Optional[int] = None


class IAQResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Backward-compatible top-level fields
    iaq: Optional[float] = Field(None, description="Predicted IAQ index")
    category: Optional[str] = Field(None, description="Air quality category")
    status: str = Field(..., description="Prediction status")
    model_type: Optional[str] = None
    raw_inputs: Optional[Dict[str, float]] = None
    buffer_size: Optional[int] = None
    required: Optional[int] = None
    message: Optional[str] = None

    # Structured inference fields
    observation: Optional[Observation] = None
    predicted: Optional[Predicted] = None
    prior: Optional[Prior] = None
    inference: Optional[InferenceMetadata] = None
    bayesian_update: Optional[BayesianUpdate] = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    window_size: int
    loaded: bool
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, bool]
    active_model: str


class ModelSelection(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str = Field(..., description="Model type to activate")


# ---------------------------------------------------------------------------
# Sensor registration (field mapping API)
# ---------------------------------------------------------------------------

class SensorRegisterRequest(BaseModel):
    """Request body for POST /sensors/register."""
    source_type: str = Field(
        "csv_headers",
        description="Source type: 'csv_headers' or 'example_payload'",
    )
    fields: List[str] = Field(..., description="List of source field names to map")
    sample_values: Optional[Dict[str, List[float]]] = Field(
        None, description="Sample values per field for range validation",
    )
    backend: str = Field("fuzzy", description="Mapping backend: 'fuzzy' or 'ollama'")


class FieldMatchResponse(BaseModel):
    """A single field mapping in the registration response."""
    source_field: str
    target_feature: str
    target_quantity: str
    confidence: float
    method: str


class SensorRegisterResponse(BaseModel):
    """Response from POST /sensors/register."""
    mapping_id: str
    status: str = "proposed"
    mapping: List[FieldMatchResponse]
    unresolved: List[str]


class SensorConfirmRequest(BaseModel):
    """Request body for POST /sensors/register/{mapping_id}/confirm."""
    overrides: Optional[Dict[str, str]] = Field(
        None, description="Manual overrides: {source_field: target_feature}",
    )


class SensorConfirmResponse(BaseModel):
    """Response from POST /sensors/register/{mapping_id}/confirm."""
    status: str = "confirmed"
    field_mapping: Dict[str, str]
