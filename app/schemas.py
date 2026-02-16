# ============================================================================
# File: app/schemas.py
# ============================================================================
from typing import Optional, Dict, Any
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
    timestamp: Optional[str] = Field(None, description="ISO timestamp")

    # Legacy BME680 fields — populated into ``readings`` if present
    temperature: Optional[float] = Field(None, exclude=True)
    rel_humidity: Optional[float] = Field(None, exclude=True)
    pressure: Optional[float] = Field(None, exclude=True)
    voc_resistance: Optional[float] = Field(None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _build_readings(cls, values):
        """Merge legacy fields into readings dict when readings is absent."""
        if isinstance(values, dict) and values.get("readings") is None:
            legacy_keys = ["temperature", "rel_humidity", "pressure", "voc_resistance"]
            legacy = {k: values[k] for k in legacy_keys if k in values and values[k] is not None}
            if legacy:
                values["readings"] = legacy
        return values

    def get_readings(self) -> Dict[str, float]:
        """Return the sensor readings dict (always populated after validation)."""
        return self.readings or {}


class IAQResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    iaq: Optional[float] = Field(None, description="Predicted IAQ index")
    category: Optional[str] = Field(None, description="Air quality category")
    status: str = Field(..., description="Prediction status")
    model_type: Optional[str] = None
    raw_inputs: Optional[Dict[str, float]] = None
    buffer_size: Optional[int] = None
    required: Optional[int] = None
    message: Optional[str] = None


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
