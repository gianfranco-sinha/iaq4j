"""Sensor drift correction functions.

Applies age-based drift correction to raw sensor readings using
coefficients from drift analysis (results/drift_3yr/drift_summary.json).
Sensor age is estimated per-reading from the timestamp relative to the
drift summary's date_start (when the sensor began collecting data).

Two strategies: linear subtraction and VOC-compensated (log-space for VOC).
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger("training.drift_correction")

DEFAULT_DRIFT_SUMMARY = "results/drift_3yr/drift_summary.json"


@dataclass
class DriftCoefficient:
    """Drift coefficient for a single feature."""

    feature: str
    trend_slope_per_day: float
    trend_r2: float
    status: str


@dataclass
class DriftSummary:
    """Drift coefficients plus the sensor start date."""

    coefficients: Dict[str, DriftCoefficient]
    date_start: pd.Timestamp


def load_drift_summary(
    summary_path: Optional[str] = None,
) -> DriftSummary:
    """Load drift summary from JSON, including date_start and per-feature coefficients.

    Args:
        summary_path: Path to drift_summary.json. Defaults to
            results/drift_3yr/drift_summary.json.

    Returns:
        DriftSummary with coefficients and sensor start date.
    """
    path = Path(summary_path or DEFAULT_DRIFT_SUMMARY)
    if not path.exists():
        raise FileNotFoundError(f"Drift summary not found: {path}")

    with open(path) as f:
        data = json.load(f)

    coefficients = {}
    for feature_name, metrics in data.get("features", {}).items():
        coefficients[feature_name] = DriftCoefficient(
            feature=feature_name,
            trend_slope_per_day=metrics["trend_slope_per_day"],
            trend_r2=metrics["trend_r2"],
            status=metrics.get("status", "OK"),
        )

    date_start = pd.Timestamp(data["date_start"])

    return DriftSummary(coefficients=coefficients, date_start=date_start)


def compute_sensor_age_days(sensor_start: pd.Timestamp, reading_time: pd.Timestamp) -> float:
    """Compute fractional days between sensor start and a reading timestamp."""
    delta = reading_time - sensor_start
    return delta.total_seconds() / 86400.0


def apply_linear_correction(
    readings: Dict[str, float],
    age_days: float,
    coefficients: Dict[str, DriftCoefficient],
) -> Dict[str, float]:
    """Apply linear drift correction: corrected = raw - slope * age_days.

    Only corrects features present in both readings and coefficients.
    Features without drift coefficients pass through unchanged.
    """
    corrected = dict(readings)
    for feature, value in readings.items():
        if feature in coefficients:
            slope = coefficients[feature].trend_slope_per_day
            corrected[feature] = value - slope * age_days
    return corrected


def apply_voc_compensated_correction(
    readings: Dict[str, float],
    age_days: float,
    coefficients: Dict[str, DriftCoefficient],
) -> Dict[str, float]:
    """Apply VOC-compensated drift correction.

    For non-VOC features: same as linear (subtract slope * age).
    For voc_resistance: works in log-space using the compensated slope,
        corrected_voc = raw_voc * exp(-compensated_slope * age_days)
    """
    corrected = dict(readings)

    for feature, value in readings.items():
        if feature == "voc_resistance":
            # Use compensated slope in log-space
            comp_coeff = coefficients.get("voc_resistance_compensated")
            if comp_coeff is not None:
                corrected[feature] = value * math.exp(
                    -comp_coeff.trend_slope_per_day * age_days
                )
        elif feature in coefficients:
            slope = coefficients[feature].trend_slope_per_day
            corrected[feature] = value - slope * age_days

    return corrected
