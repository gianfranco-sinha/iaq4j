"""Tests for per-sensor drift report generation and persistence."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app.builtin_profiles  # noqa: F401
from app.inference import _SensorDriftState, InferenceEngine


@pytest.fixture
def drift_summary_data():
    """Minimal drift summary matching the real format."""
    return {
        "date_start": "2023-02-14 11:42:30+00:00",
        "date_end": "2026-02-27 16:35:46+00:00",
        "features": {
            "temperature": {
                "trend_slope_per_day": 0.000193,
                "trend_r2": 0.000285,
                "status": "OK",
            },
            "rel_humidity": {
                "trend_slope_per_day": -0.00199,
                "trend_r2": 0.01164,
                "status": "OK",
            },
            "pressure": {
                "trend_slope_per_day": -0.00137,
                "trend_r2": 0.00151,
                "status": "OK",
            },
            "voc_resistance": {
                "trend_slope_per_day": 307.2,
                "trend_r2": 0.214,
                "status": "DRIFT",
            },
            "voc_resistance_compensated": {
                "trend_slope_per_day": 0.000786,
                "trend_r2": 0.424,
                "status": "OK",
            },
        },
    }


@pytest.fixture
def mock_drift_summary(drift_summary_data, tmp_path):
    """Write drift summary to a temp file and patch load_drift_summary to use it."""
    path = tmp_path / "drift_summary.json"
    path.write_text(json.dumps(drift_summary_data))
    return str(path)


def _make_reading(temp=25.0, rh=50.0, pressure=1013.25, voc=100000.0):
    return {
        "temperature": temp,
        "rel_humidity": rh,
        "pressure": pressure,
        "voc_resistance": voc,
    }


def _make_timestamps(n, start_date="2024-06-01T12:00:00+00:00", interval_seconds=3):
    """Generate n timestamps starting from start_date."""
    start = datetime.fromisoformat(start_date)
    return [start + timedelta(seconds=i * interval_seconds) for i in range(n)]


class TestSensorDriftState:
    def test_add_reading(self):
        state = _SensorDriftState(sensor_id="test-sensor")
        state.add_reading(_make_reading())
        assert state.readings_count == 1
        assert len(state.readings) == 1

    def test_rolling_window_cap(self):
        state = _SensorDriftState(sensor_id="test-sensor", max_readings=5)
        for i in range(10):
            state.add_reading(_make_reading(temp=20 + i))
        assert state.readings_count == 10
        assert len(state.readings) == 5
        # Oldest readings dropped
        assert state.readings[0]["temperature"] == 25.0

    def test_generate_report_insufficient_data(self):
        state = _SensorDriftState(sensor_id="test-sensor")
        for i in range(5):
            state.add_reading(_make_reading())
        assert state.generate_report() is None

    def test_generate_report_with_enough_data(self, mock_drift_summary):
        state = _SensorDriftState(sensor_id="test-sensor")
        timestamps = _make_timestamps(20, start_date="2024-06-01T12:00:00+00:00")

        for i, ts in enumerate(timestamps):
            state.add_reading(_make_reading(temp=25.0 + i * 0.1), ts)

        with patch(
            "training.drift_correction.DEFAULT_DRIFT_SUMMARY",
            mock_drift_summary,
        ):
            report = state.generate_report()

        assert report is not None
        assert report["sensor_id"] == "test-sensor"
        assert report["readings_count"] == 20
        assert report["readings_in_window"] == 20
        assert "temperature" in report["features"]
        assert "voc_resistance" in report["features"]
        assert report["health"] in ("good", "warning", "drift")

    def test_report_detects_voc_drift(self, mock_drift_summary):
        """VOC resistance should be flagged as DRIFT."""
        state = _SensorDriftState(sensor_id="test-sensor")
        # Timestamps 2 years after sensor start → large estimated drift
        timestamps = _make_timestamps(20, start_date="2025-02-14T12:00:00+00:00")

        for ts in timestamps:
            state.add_reading(_make_reading(voc=200000.0), ts)

        with patch(
            "training.drift_correction.DEFAULT_DRIFT_SUMMARY",
            mock_drift_summary,
        ):
            report = state.generate_report()

        voc_stats = report["features"]["voc_resistance"]
        assert voc_stats["drift_status"] == "DRIFT"
        assert voc_stats["estimated_drift"] is not None
        # 2 years * 307.2/day = ~224k drift — significant % of 200k reading
        assert abs(voc_stats["estimated_drift"]) > 100000

    def test_serialization_roundtrip(self):
        state = _SensorDriftState(sensor_id="test-sensor")
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        state.add_reading(_make_reading(), ts)
        state.add_reading(_make_reading(temp=26.0), ts + timedelta(seconds=3))

        data = state.to_dict()
        restored = _SensorDriftState.from_dict(data)

        assert restored.sensor_id == "test-sensor"
        assert restored.readings_count == 2
        assert len(restored.readings) == 2
        assert len(restored.timestamps) == 2
        assert restored.readings[0]["temperature"] == 25.0
        assert restored.readings[1]["temperature"] == 26.0


class TestInferenceEngineDrift:
    @pytest.fixture
    def mock_predictor(self):
        predictor = MagicMock()
        predictor.predict.return_value = {
            "status": "ready",
            "iaq": 50.0,
            "category": "good",
            "predicted": {
                "mean": 50.0,
                "category": "good",
                "uncertainty": {"std": 5.0, "ci_lower": 40.0, "ci_upper": 60.0, "method": "mc_dropout"},
            },
            "inference": {"model_type": "mlp", "window_size": 10, "buffer_size": 10},
        }
        return predictor

    def test_drift_accumulates_on_predict(self, mock_predictor):
        engine = InferenceEngine(mock_predictor)
        sensor_id = "bme680-test-001"

        engine.predict_single(
            readings=_make_reading(),
            sensor_id=sensor_id,
            timestamp="2024-06-01T12:00:00Z",
        )

        assert sensor_id in engine._sensor_drift
        assert engine._sensor_drift[sensor_id].readings_count == 1

    def test_no_drift_without_sensor_id(self, mock_predictor):
        engine = InferenceEngine(mock_predictor)
        engine.predict_single(
            readings=_make_reading(),
            timestamp="2024-06-01T12:00:00Z",
        )
        assert len(engine._sensor_drift) == 0

    def test_report_none_with_few_readings(self, mock_predictor):
        engine = InferenceEngine(mock_predictor)
        engine.update_sensor_drift("s1", _make_reading())
        assert engine.get_sensor_drift_report("s1") is None

    def test_report_none_for_unknown_sensor(self, mock_predictor):
        engine = InferenceEngine(mock_predictor)
        assert engine.get_sensor_drift_report("nonexistent") is None

    def test_save_and_load_drift_state(self, mock_predictor, tmp_path, drift_summary_data):
        """Drift state persists to disk and loads back."""
        # Write drift summary for report generation during save
        drift_path = tmp_path / "drift_summary.json"
        drift_path.write_text(json.dumps(drift_summary_data))

        with patch("app.inference.DRIFT_REPORTS_DIR", tmp_path), \
             patch("training.drift_correction.DEFAULT_DRIFT_SUMMARY", str(drift_path)):
            engine = InferenceEngine(mock_predictor)
            sensor_id = "bme680-persist-test"

            timestamps = _make_timestamps(15)
            for ts in timestamps:
                engine.update_sensor_drift(sensor_id, _make_reading(), ts)

            # Save
            path = engine.save_sensor_drift(sensor_id)
            assert path is not None
            assert path.exists()

            # Verify saved JSON contains both state and report
            saved = json.loads(path.read_text())
            assert saved["sensor_id"] == sensor_id
            assert "report" in saved

            # Load in a fresh engine
            engine2 = InferenceEngine(mock_predictor)
            state = engine2._load_drift_state(sensor_id)
            assert state is not None
            assert state.readings_count == 15
            assert len(state.readings) == 15

    def test_list_sensor_drift_reports(self, mock_predictor, tmp_path):
        with patch("app.inference.DRIFT_REPORTS_DIR", tmp_path):
            engine = InferenceEngine(mock_predictor)
            engine.update_sensor_drift("sensor-a", _make_reading())
            engine.update_sensor_drift("sensor-b", _make_reading())

            # Also persist one to disk only
            (tmp_path / "sensor-c.json").write_text("{}")

            ids = engine.list_sensor_drift_reports()
            assert "sensor-a" in ids
            assert "sensor-b" in ids
            assert "sensor-c" in ids
