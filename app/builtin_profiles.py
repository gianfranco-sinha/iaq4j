# ============================================================================
# File: app/builtin_profiles.py
# Built-in sensor profiles and IAQ standards.
# ============================================================================
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.profiles import (
    SensorProfile,
    register_sensor,
)
from app.standards import register_yaml_standards


class BME680Profile(SensorProfile):
    """Bosch BME680/BME688 environmental sensor."""

    @property
    def name(self) -> str:
        return "bme680"

    @property
    def raw_features(self) -> List[str]:
        return ["temperature", "rel_humidity", "pressure", "voc_resistance"]

    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "temperature": "temperature",
            "rel_humidity": "relative_humidity",
            "pressure": "barometric_pressure",
            "voc_resistance": "voc_resistance",
        }

    @property
    def valid_ranges(self) -> Dict[str, Tuple[float, float]]:
        ranges = super().valid_ranges  # computed from registry
        ranges["iaq_accuracy"] = (2, 3)  # quality column, not a quantity
        return ranges

    @property
    def quality_column(self) -> Optional[str]:
        return "iaq_accuracy"

    @property
    def quality_min(self) -> Optional[float]:
        return 2

    @property
    def expected_interval_seconds(self) -> Optional[float]:
        return 3.0  # BME680 LP mode

    @property
    def engineered_feature_names(self) -> List[str]:
        return ["voc_ratio", "abs_humidity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    def compute_baselines(self, raw: np.ndarray) -> Dict[str, float]:
        voc_idx = self.raw_features.index("voc_resistance")
        return {"voc_resistance": float(np.median(raw[:, voc_idx]))}

    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        from training.utils import calculate_absolute_humidity

        voc_idx = self.raw_features.index("voc_resistance")
        temp_idx = self.raw_features.index("temperature")
        hum_idx = self.raw_features.index("rel_humidity")

        baseline_voc = (
            baselines.get("voc_resistance", float(np.median(raw[:, voc_idx])))
            if baselines
            else float(np.median(raw[:, voc_idx]))
        )
        voc_ratio = raw[:, voc_idx] / baseline_voc
        abs_humidity = calculate_absolute_humidity(raw[:, temp_idx], raw[:, hum_idx])

        # Temporal cyclical features
        if timestamps is not None:
            ts_index = pd.DatetimeIndex(timestamps)
            hours = ts_index.hour.values.astype(float)
            dows = ts_index.dayofweek.values.astype(float)
        else:
            hours = np.zeros(len(raw))
            dows = np.zeros(len(raw))

        hour_sin, hour_cos = self._cyclical_encode(hours, 24.0)
        dow_sin, dow_cos = self._cyclical_encode(dows, 7.0)

        return np.column_stack([
            raw,
            voc_ratio.reshape(-1, 1),
            abs_humidity.reshape(-1, 1),
            hour_sin.reshape(-1, 1),
            hour_cos.reshape(-1, 1),
            dow_sin.reshape(-1, 1),
            dow_cos.reshape(-1, 1),
        ])

    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        from training.utils import calculate_absolute_humidity

        raw_vals = [reading[f] for f in self.raw_features]

        voc_r = reading["voc_resistance"]
        baseline_voc = (
            baselines.get("voc_resistance", voc_r) if baselines else voc_r
        )
        voc_ratio = voc_r / baseline_voc

        abs_hum = calculate_absolute_humidity(
            np.array([reading["temperature"]]),
            np.array([reading["rel_humidity"]]),
        )[0]

        hour = float(timestamp.hour) if timestamp is not None else 0.0
        dow = float(timestamp.weekday()) if timestamp is not None else 0.0

        hour_sin, hour_cos = self._cyclical_encode(np.array([hour]), 24.0)
        dow_sin, dow_cos = self._cyclical_encode(np.array([dow]), 7.0)

        return np.array(raw_vals + [voc_ratio, abs_hum,
                                    hour_sin[0], hour_cos[0],
                                    dow_sin[0], dow_cos[0]])


class SPS30Profile(SensorProfile):
    """Sensirion SPS30 particulate matter sensor."""

    @property
    def name(self) -> str:
        return "sps30"

    @property
    def raw_features(self) -> List[str]:
        return ["pm1_0", "pm2_5", "pm4_0", "pm10"]

    @property
    def feature_quantities(self) -> Dict[str, str]:
        return {
            "pm1_0": "pm1_0",
            "pm2_5": "pm2_5",
            "pm4_0": "pm4_0",
            "pm10": "pm10",
        }

    @property
    def engineered_feature_names(self) -> List[str]:
        return ["pm25_pm10_ratio", "pm1_pm25_ratio"]

    def engineer_features(
        self,
        raw: np.ndarray,
        baselines: Optional[Dict[str, float]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        pm1_idx = self.raw_features.index("pm1_0")
        pm25_idx = self.raw_features.index("pm2_5")
        pm10_idx = self.raw_features.index("pm10")

        pm25_pm10_ratio = raw[:, pm25_idx] / np.maximum(raw[:, pm10_idx], 0.1)
        pm1_pm25_ratio = raw[:, pm1_idx] / np.maximum(raw[:, pm25_idx], 0.1)

        return np.column_stack(
            [raw, pm25_pm10_ratio.reshape(-1, 1), pm1_pm25_ratio.reshape(-1, 1)]
        )

    def engineer_features_single(
        self,
        reading: Dict[str, float],
        baselines: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        raw_vals = [reading[f] for f in self.raw_features]

        pm25_pm10_ratio = reading["pm2_5"] / max(reading["pm10"], 0.1)
        pm1_pm25_ratio = reading["pm1_0"] / max(reading["pm2_5"], 0.1)

        return np.array(raw_vals + [pm25_pm10_ratio, pm1_pm25_ratio])


# ---------------------------------------------------------------------------
# Register built-in profiles and YAML-driven standards
# ---------------------------------------------------------------------------
register_sensor("bme680", BME680Profile)
register_sensor("sps30", SPS30Profile)
register_yaml_standards()
