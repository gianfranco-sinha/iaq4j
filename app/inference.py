# ============================================================================
# File: app/inference.py
# ============================================================================
"""
Inference logic for IAQ prediction.
Handles batch processing, streaming, and real-time predictions.
"""

import logging
import math
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_prior_variable(
    var_name: str, value: float, var_config: dict
) -> Optional[dict]:
    """Resolve a prior variable value to its configured shift and std.

    For boolean variables, value >= 0.5 maps to "true", else "false".
    Returns dict with target_shift, prior_std, state, description or None.
    """
    var_type = var_config.get("type", "boolean")
    values_map = var_config.get("values", {})

    if var_type == "boolean":
        state = "true" if value >= 0.5 else "false"
    else:
        state = str(value)

    state_config = values_map.get(state)
    if state_config is None:
        logger.warning(
            "Prior variable '%s' has no config for state '%s' — skipped",
            var_name,
            state,
        )
        return None

    return {
        "variable": var_name,
        "value": value,
        "state": state,
        "target_shift": float(state_config.get("target_shift", 0.0)),
        "prior_std": float(state_config.get("prior_std", 100.0)),
        "description": var_config.get("description"),
    }


def _apply_bayesian_update(
    model_mean: float,
    model_std: float,
    prior_variables: Dict[str, float],
) -> Optional[dict]:
    """Apply sequential Gaussian conjugate update from prior variables.

    Each prior variable contributes N(model_mean + shift, prior_std^2).
    The conjugate formula combines model likelihood with each prior:
        precision_post = 1/sigma_model^2 + 1/sigma_prior^2
        mu_post = (mu_model/sigma_model^2 + mu_prior/sigma_prior^2) / precision_post

    Returns dict with pre/post mean+std and list of applied effects, or None.
    """
    prior_config = settings.get_prior_variables_config()
    if not prior_config or not prior_variables:
        return None

    pre_mean = model_mean
    pre_std = model_std

    mu = model_mean
    sigma = model_std
    applied = []

    for var_name, value in prior_variables.items():
        var_cfg = prior_config.get(var_name)
        if var_cfg is None:
            logger.warning("Unknown prior variable '%s' — skipped", var_name)
            continue

        resolved = _resolve_prior_variable(var_name, value, var_cfg)
        if resolved is None:
            continue

        shift = resolved["target_shift"]
        prior_std = resolved["prior_std"]

        if prior_std <= 0 or sigma <= 0:
            continue

        # Gaussian conjugate update
        mu_prior = mu + shift
        prec_model = 1.0 / (sigma**2)
        prec_prior = 1.0 / (prior_std**2)
        prec_post = prec_model + prec_prior
        mu = (mu * prec_model + mu_prior * prec_prior) / prec_post
        sigma = math.sqrt(1.0 / prec_post)

        applied.append(resolved)

    if not applied:
        return None

    return {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "post_mean": mu,
        "post_std": sigma,
        "variables_applied": applied,
    }


class InferenceEngine:
    """High-level inference engine for IAQ prediction."""

    def __init__(self, predictor):
        """
        Args:
            predictor: IAQPredictor instance
        """
        self.predictor = predictor
        self.prediction_history = []
        self.max_history = 1000
        self.last_sequences: Dict[str, int] = {}
        self.last_timestamps: Dict[str, float] = {}
        self._last_reading_ts: Optional[datetime] = None
        # Readings arriving more than this many seconds after the previous one
        # are flagged as stale (gap > 2× typical 30 s BME680 LP interval).
        self._staleness_threshold_seconds: float = 60.0

    def _validate_sequence(
        self,
        sensor_id: Optional[str],
        sequence_number: Optional[int],
        timestamp: Optional[str] = None,
    ) -> Optional[Dict]:
        """Validate sequence number for replay detection and ordering.

        Also validates that timestamp and sequence_number are monotonically aligned
        (i.e., if t2 > t1 then s2 > s1).

        Args:
            sensor_id: Unique sensor identifier
            sequence_number: Monotonically increasing sequence number
            timestamp: ISO timestamp string

        Returns:
            Dict with validation result (None if valid), or None if no check needed
        """
        if sensor_id is None:
            if sequence_number is not None:
                logger.warning(
                    "sequence_number provided (%d) but no sensor_id — cannot track per-sensor sequence",
                    sequence_number,
                )
            return None

        if sequence_number is None:
            logger.debug(
                "No sequence_number provided for sensor_id=%s — skipping sequence validation",
                sensor_id,
            )
            return None

        last_seq = self.last_sequences.get(sensor_id)
        last_ts = self.last_timestamps.get(sensor_id)

        if last_seq is None:
            logger.info(
                "First reading from sensor_id=%s with sequence_number=%d",
                sensor_id,
                sequence_number,
            )
        elif sequence_number < last_seq:
            logger.warning(
                "Sequence regression for sensor_id=%s: got %d, expected >= %d",
                sensor_id,
                sequence_number,
                last_seq,
            )
            return {
                "status": "regression",
                "message": f"Sequence number {sequence_number} is less than last seen {last_seq}",
                "last_sequence": last_seq,
                "received_sequence": sequence_number,
            }
        elif sequence_number == last_seq:
            logger.warning(
                "Duplicate sequence for sensor_id=%s: sequence_number=%d already seen",
                sensor_id,
                sequence_number,
            )
            return {
                "status": "duplicate",
                "message": f"Sequence number {sequence_number} already processed",
                "last_sequence": last_seq,
            }

        if timestamp is not None and last_ts is not None:
            try:
                from datetime import datetime

                current_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                last_ts_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))

                ts_advanced = current_ts > last_ts_dt
                seq_advanced = sequence_number > last_seq

                if ts_advanced != seq_advanced:
                    violation_type = (
                        "timestamp_advanced_seq_not"
                        if ts_advanced
                        else "seq_advanced_timestamp_not"
                    )
                    logger.warning(
                        "Monotonicity violation for sensor_id=%s: timestamp and sequence_number not aligned. "
                        "timestamp: %s -> %s (%s), sequence: %d -> %d (%s)",
                        sensor_id,
                        last_ts,
                        timestamp,
                        "advanced" if ts_advanced else "regressed",
                        last_seq,
                        sequence_number,
                        "advanced" if seq_advanced else "regressed",
                    )
                    return {
                        "status": "monotonicity_violation",
                        "message": f"Timestamp and sequence number monotonicity misaligned: "
                        f"timestamp {('advanced' if ts_advanced else 'regressed')}, "
                        f"sequence {('advanced' if seq_advanced else 'regressed')}",
                        "last_timestamp": last_ts,
                        "received_timestamp": timestamp,
                        "last_sequence": last_seq,
                        "received_sequence": sequence_number,
                    }
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Failed to parse timestamp for monotonicity check: %s — %s",
                    timestamp,
                    e,
                )

        return None

        return None

    def get_sequence_state(self, sensor_id: str) -> Dict:
        """Get the current sequence state for a sensor.

        Args:
            sensor_id: Unique sensor identifier

        Returns:
            Dict with last_sequence and timestamp info
        """
        last_seq = self.last_sequences.get(sensor_id)
        last_ts = self.last_timestamps.get(sensor_id)
        if last_seq is None:
            return {
                "sensor_id": sensor_id,
                "last_sequence": None,
                "last_timestamp": None,
                "message": "No readings received from this sensor",
            }
        return {
            "sensor_id": sensor_id,
            "last_sequence": last_seq,
            "last_timestamp": last_ts,
        }

    def _compute_prior(self) -> Optional[dict]:
        """Compute a prior belief about IAQ from recent history or training distribution."""
        if len(self.prediction_history) >= 5:
            window = min(20, len(self.prediction_history))
            recent_iaqs = [h["iaq"] for h in self.prediction_history[-window:]]
            return {
                "mean": float(np.mean(recent_iaqs)),
                "std": float(np.std(recent_iaqs)),
                "source": "history_window",
                "n_observations": window,
            }

        if self.predictor.target_scaler is not None:
            scaler = self.predictor.target_scaler
            if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                lo = float(scaler.data_min_[0])
                hi = float(scaler.data_max_[0])
                return {
                    "mean": (lo + hi) / 2.0,
                    "std": (hi - lo) / 4.0,
                    "source": "training_distribution",
                    "n_observations": 0,
                }

        return None

    def predict_single(
        self,
        readings: dict = None,
        include_uncertainty: bool = True,
        n_mc_samples: int = 10,
        prior_variables: Dict[str, float] = None,
        sensor_id: Optional[str] = None,
        sequence_number: Optional[int] = None,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Single prediction with prior, uncertainty, and structured output.

        Args:
            readings: sensor readings dict.
            include_uncertainty: whether to run MC dropout for uncertainty.
            n_mc_samples: number of MC dropout forward passes.
            prior_variables: external signals for Bayesian conjugate update.
            sensor_id: Unique sensor identifier for sequence tracking.
            sequence_number: Monotonically increasing sequence number for replay detection.
            timestamp: ISO timestamp string for temporal ordering validation.
        """
        if readings is None:
            readings = kwargs

        # Parse ISO timestamp to datetime for temporal features and staleness detection
        parsed_ts: Optional[datetime] = None
        if timestamp:
            try:
                parsed_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning("Could not parse timestamp '%s' — temporal features will be zero-valued", timestamp)

        # Staleness detection: flag if gap since last reading exceeds threshold
        stale = False
        stale_gap: Optional[float] = None
        if parsed_ts is not None and self._last_reading_ts is not None:
            gap = (parsed_ts - self._last_reading_ts).total_seconds()
            if gap > self._staleness_threshold_seconds:
                stale = True
                stale_gap = gap
                logger.warning(
                    "Stale reading detected: %.1fs gap (threshold %.0fs) — "
                    "buffer may contain outdated context",
                    gap,
                    self._staleness_threshold_seconds,
                )

        seq_validation = self._validate_sequence(sensor_id, sequence_number, timestamp)

        if seq_validation is not None:
            result = {
                "status": "sequence_error",
                "sequence_error": seq_validation,
                "readings": readings,
            }
            if sensor_id is not None and sequence_number is not None:
                if seq_validation["status"] == "regression":
                    pass
                else:
                    return result
            return result

        # Compute prior BEFORE the new prediction
        prior = self._compute_prior()

        mc = n_mc_samples if include_uncertainty else 1
        result = self.predictor.predict(readings, n_mc_samples=mc, timestamp=parsed_ts)

        # Attach prior
        if prior is not None:
            result["prior"] = prior

        # Apply Bayesian conjugate update from prior variables
        if (
            prior_variables
            and result.get("status") == "ready"
            and result.get("iaq") is not None
        ):
            predicted = result.get("predicted", {})
            unc = predicted.get("uncertainty")
            if unc and unc.get("std") and unc["std"] > 0:
                update = _apply_bayesian_update(
                    predicted["mean"],
                    unc["std"],
                    prior_variables,
                )
                if update is not None:
                    from app.profiles import get_iaq_standard

                    standard = get_iaq_standard()

                    new_iaq = standard.clamp(update["post_mean"])
                    new_category = standard.categorize(new_iaq)
                    new_std = update["post_std"]

                    result["iaq"] = new_iaq
                    result["category"] = new_category
                    result["predicted"]["mean"] = new_iaq
                    result["predicted"]["category"] = new_category
                    result["predicted"]["uncertainty"]["std"] = new_std
                    result["predicted"]["uncertainty"]["ci_lower"] = standard.clamp(
                        new_iaq - 1.96 * new_std
                    )
                    result["predicted"]["uncertainty"]["ci_upper"] = standard.clamp(
                        new_iaq + 1.96 * new_std
                    )
                    result["bayesian_update"] = update

        # Attach staleness flag if detected
        if stale:
            result["stale"] = True
            result["stale_gap_seconds"] = round(stale_gap, 1)

        # Add to history if prediction was successful
        if result.get("status") == "ready" and result.get("iaq") is not None:
            history_entry = dict(readings)
            history_entry["iaq"] = result["iaq"]
            self.prediction_history.append(history_entry)

            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)

            # Update sequence and timestamp tracking
            if sensor_id is not None and sequence_number is not None:
                self.last_sequences[sensor_id] = sequence_number
                if timestamp is not None:
                    self.last_timestamps[sensor_id] = timestamp
            if parsed_ts is not None:
                self._last_reading_ts = parsed_ts

            # Supplement with history-based uncertainty for models without dropout
            predicted = result.get("predicted", {})
            unc = predicted.get("uncertainty", {})
            if (
                unc.get("method") == "deterministic"
                and len(self.prediction_history) >= 10
            ):
                recent_iaqs = [h["iaq"] for h in self.prediction_history[-20:]]
                result["predicted"]["uncertainty"] = {
                    "std": float(np.std(recent_iaqs)),
                    "ci_lower": float(np.percentile(recent_iaqs, 2.5)),
                    "ci_upper": float(np.percentile(recent_iaqs, 97.5)),
                    "method": "history_std",
                }
                result["inference"]["uncertainty_method"] = "history_std"

            # Add statistics if we have history
            if len(self.prediction_history) >= 10:
                recent_iaqs = [h["iaq"] for h in self.prediction_history[-10:]]
                result["statistics"] = {
                    "recent_avg": np.mean(recent_iaqs),
                    "recent_std": np.std(recent_iaqs),
                    "recent_min": np.min(recent_iaqs),
                    "recent_max": np.max(recent_iaqs),
                    "trend": "improving"
                    if recent_iaqs[-1] < recent_iaqs[0]
                    else "worsening",
                }

        return result

    def predict_batch(self, readings: List[Dict]) -> List[Dict]:
        """Batch prediction for multiple readings."""
        return [self.predict_single(reading) for reading in readings]

    def predict_with_uncertainty(
        self,
        readings: dict = None,
        n_samples: int = 10,
        prior_variables: Dict[str, float] = None,
        **kwargs,
    ) -> Dict:
        """Prediction with uncertainty estimation.

        Delegates to predict_single with MC dropout enabled.
        Works for all model types — models without dropout get history-based
        uncertainty instead.
        """
        return self.predict_single(
            readings=readings or kwargs,
            include_uncertainty=True,
            n_mc_samples=n_samples,
            prior_variables=prior_variables,
        )

    def get_statistics(self) -> Dict:
        """Get statistics from prediction history."""
        if not self.prediction_history:
            return {"count": 0, "message": "No predictions yet"}

        iaqs = [h["iaq"] for h in self.prediction_history]

        from app.profiles import get_iaq_standard

        standard = get_iaq_standard()

        return {
            "count": len(self.prediction_history),
            "iaq_mean": float(np.mean(iaqs)),
            "iaq_std": float(np.std(iaqs)),
            "iaq_min": float(np.min(iaqs)),
            "iaq_max": float(np.max(iaqs)),
            "percentiles": {
                "p25": float(np.percentile(iaqs, 25)),
                "p50": float(np.percentile(iaqs, 50)),
                "p75": float(np.percentile(iaqs, 75)),
                "p95": float(np.percentile(iaqs, 95)),
            },
            "distribution": standard.category_distribution(iaqs),
        }

    def reset_history(self):
        """Clear prediction history."""
        self.prediction_history = []
        logger.info("Prediction history cleared")

    def analyze_sensor_drift(self) -> Optional[Dict]:
        """Analyze if sensor might be drifting based on prediction history."""
        if len(self.prediction_history) < 50:
            return None

        from app.profiles import get_sensor_profile

        profile = get_sensor_profile()

        recent = self.prediction_history[-50:]

        # Check stability of each raw feature
        feature_stability = {}
        warnings = []

        for feat in profile.raw_features:
            values = [h.get(feat) for h in recent if feat in h]
            if not values:
                continue
            feat_mean = np.mean(values)
            feat_std = np.std(values)
            cv = float(feat_std / feat_mean) if feat_mean != 0 else 0.0
            feature_stability[feat] = {
                "mean": float(feat_mean),
                "std": float(feat_std),
                "cv": cv,
            }
            if cv > 0.3:
                warnings.append(f"High {feat} variability (CV={cv:.2f})")
            elif cv < 0.01 and feat_mean != 0:
                warnings.append(f"{feat} appears stuck (CV={cv:.4f})")

        # Check IAQ trend
        iaqs = [h["iaq"] for h in recent]
        iaq_trend = np.polyfit(range(len(iaqs)), iaqs, 1)[0]

        if abs(iaq_trend) > 1.0:
            warnings.append("Rapid IAQ trend detected")

        return {
            "feature_stability": feature_stability,
            "iaq_trend": float(iaq_trend),
            "warnings": warnings,
            "health": "good" if not warnings else "warning",
        }


class StreamingInference:
    """Handle streaming/continuous inference with rate limiting."""

    def __init__(self, predictor, max_rate_hz: float = 1.0):
        """
        Args:
            predictor: IAQPredictor instance
            max_rate_hz: Maximum prediction rate in Hz
        """
        self.predictor = predictor
        self.min_interval = 1.0 / max_rate_hz
        self.last_prediction_time = 0
        self.dropped_readings = 0

    def predict(
        self, readings: dict = None, force: bool = False, **kwargs
    ) -> Optional[Dict]:
        """Predict with rate limiting.

        Accepts a readings dict or keyword args for backward compatibility.
        """
        if readings is None:
            readings = kwargs

        import time

        current_time = time.time()

        if not force and (current_time - self.last_prediction_time) < self.min_interval:
            self.dropped_readings += 1
            return None

        result = self.predictor.predict(readings)
        self.last_prediction_time = current_time

        return result

    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            "dropped_readings": self.dropped_readings,
            "max_rate_hz": 1.0 / self.min_interval,
            "buffer_status": {
                "current_size": len(self.predictor.buffer),
                "required_size": self.predictor.window_size,
                "ready": len(self.predictor.buffer) >= self.predictor.window_size,
            },
        }
