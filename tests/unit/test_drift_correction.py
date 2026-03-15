"""Tests for training.drift_correction module."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from training.drift_correction import (
    DriftCoefficient,
    apply_linear_correction,
    apply_voc_compensated_correction,
    compute_sensor_age_days,
    load_drift_summary,
)


@pytest.fixture
def sample_coefficients():
    """Drift coefficients matching drift_summary.json structure."""
    return {
        "temperature": DriftCoefficient(
            feature="temperature",
            trend_slope_per_day=0.000193,
            trend_r2=0.000285,
            status="OK",
        ),
        "rel_humidity": DriftCoefficient(
            feature="rel_humidity",
            trend_slope_per_day=-0.00199,
            trend_r2=0.01164,
            status="OK",
        ),
        "pressure": DriftCoefficient(
            feature="pressure",
            trend_slope_per_day=-0.00137,
            trend_r2=0.00151,
            status="OK",
        ),
        "voc_resistance": DriftCoefficient(
            feature="voc_resistance",
            trend_slope_per_day=307.2,
            trend_r2=0.214,
            status="DRIFT",
        ),
        "voc_resistance_compensated": DriftCoefficient(
            feature="voc_resistance_compensated",
            trend_slope_per_day=0.000786,
            trend_r2=0.424,
            status="OK",
        ),
    }


@pytest.fixture
def sample_readings():
    return {
        "temperature": 25.0,
        "rel_humidity": 50.0,
        "pressure": 1013.25,
        "voc_resistance": 100000.0,
    }


class TestComputeSensorAgeDays:
    def test_same_time_returns_zero(self):
        t = pd.Timestamp("2023-02-14 12:00:00+00:00")
        assert compute_sensor_age_days(t, t) == 0.0

    def test_one_day(self):
        start = pd.Timestamp("2023-02-14 00:00:00+00:00")
        reading = pd.Timestamp("2023-02-15 00:00:00+00:00")
        assert compute_sensor_age_days(start, reading) == pytest.approx(1.0)

    def test_fractional_days(self):
        start = pd.Timestamp("2023-02-14 00:00:00+00:00")
        reading = pd.Timestamp("2023-02-14 12:00:00+00:00")
        assert compute_sensor_age_days(start, reading) == pytest.approx(0.5)

    def test_three_years(self):
        start = pd.Timestamp("2023-02-14 11:42:30+00:00")
        reading = pd.Timestamp("2026-02-27 16:35:46+00:00")
        age = compute_sensor_age_days(start, reading)
        assert age == pytest.approx(1109.204, abs=0.01)


class TestLinearCorrection:
    def test_subtracts_slope(self, sample_readings, sample_coefficients):
        age_days = 365.0
        corrected = apply_linear_correction(sample_readings, age_days, sample_coefficients)

        # temperature: 25.0 - 0.000193 * 365 = 24.929555
        assert corrected["temperature"] == pytest.approx(
            25.0 - 0.000193 * 365, abs=1e-4
        )
        # voc_resistance: 100000 - 307.2 * 365 = -12128 (large correction)
        assert corrected["voc_resistance"] == pytest.approx(
            100000.0 - 307.2 * 365, abs=1
        )

    def test_zero_age_returns_unchanged(self, sample_readings, sample_coefficients):
        corrected = apply_linear_correction(sample_readings, 0.0, sample_coefficients)
        for f in sample_readings:
            assert corrected[f] == sample_readings[f]

    def test_missing_coefficient_skips_feature(self, sample_readings):
        """Features without coefficients pass through unchanged."""
        partial_coeffs = {
            "temperature": DriftCoefficient("temperature", 0.1, 0.5, "DRIFT"),
        }
        corrected = apply_linear_correction(sample_readings, 100.0, partial_coeffs)

        # temperature corrected
        assert corrected["temperature"] == pytest.approx(25.0 - 0.1 * 100)
        # others unchanged
        assert corrected["rel_humidity"] == 50.0
        assert corrected["pressure"] == 1013.25
        assert corrected["voc_resistance"] == 100000.0

    def test_correction_preserves_non_drifting_features(self, sample_readings, sample_coefficients):
        """T/H/P corrections are negligible at 30 days."""
        corrected = apply_linear_correction(sample_readings, 30.0, sample_coefficients)

        # Temperature drift at 30 days: 0.000193 * 30 = 0.00579 C — negligible
        assert abs(corrected["temperature"] - sample_readings["temperature"]) < 0.01
        # Humidity: 0.00199 * 30 = 0.0597 — negligible
        assert abs(corrected["rel_humidity"] - sample_readings["rel_humidity"]) < 0.1
        # Pressure: 0.00137 * 30 = 0.041 — negligible
        assert abs(corrected["pressure"] - sample_readings["pressure"]) < 0.1


class TestVocCompensatedCorrection:
    def test_uses_log_space_for_voc(self, sample_readings, sample_coefficients):
        age_days = 365.0
        corrected = apply_voc_compensated_correction(
            sample_readings, age_days, sample_coefficients
        )

        # VOC: corrected = 100000 * exp(-0.000786 * 365)
        expected_voc = 100000.0 * math.exp(-0.000786 * 365)
        assert corrected["voc_resistance"] == pytest.approx(expected_voc, rel=1e-4)

    def test_non_voc_same_as_linear(self, sample_readings, sample_coefficients):
        age_days = 365.0
        linear = apply_linear_correction(sample_readings, age_days, sample_coefficients)
        compensated = apply_voc_compensated_correction(
            sample_readings, age_days, sample_coefficients
        )

        # T, H, P should be identical to linear
        assert compensated["temperature"] == pytest.approx(linear["temperature"])
        assert compensated["rel_humidity"] == pytest.approx(linear["rel_humidity"])
        assert compensated["pressure"] == pytest.approx(linear["pressure"])
        # VOC should differ
        assert compensated["voc_resistance"] != pytest.approx(linear["voc_resistance"], rel=0.01)

    def test_zero_age_returns_unchanged(self, sample_readings, sample_coefficients):
        corrected = apply_voc_compensated_correction(
            sample_readings, 0.0, sample_coefficients
        )
        for f in sample_readings:
            assert corrected[f] == pytest.approx(sample_readings[f])

    def test_voc_correction_smaller_than_linear(self, sample_readings, sample_coefficients):
        """Compensated correction should be much smaller than raw linear for VOC."""
        age_days = 365.0
        linear = apply_linear_correction(sample_readings, age_days, sample_coefficients)
        compensated = apply_voc_compensated_correction(
            sample_readings, age_days, sample_coefficients
        )

        linear_delta = abs(linear["voc_resistance"] - sample_readings["voc_resistance"])
        comp_delta = abs(compensated["voc_resistance"] - sample_readings["voc_resistance"])

        # Compensated delta should be much smaller than linear delta
        assert comp_delta < linear_delta


class TestLoadDriftSummary:
    def test_load_from_results(self):
        """Load from the actual drift_summary.json if available."""
        summary_path = Path("results/drift_3yr/drift_summary.json")
        if not summary_path.exists():
            pytest.skip("drift_summary.json not available")

        summary = load_drift_summary(str(summary_path))
        assert "temperature" in summary.coefficients
        assert "voc_resistance" in summary.coefficients
        assert "voc_resistance_compensated" in summary.coefficients
        assert summary.coefficients["voc_resistance"].status == "DRIFT"
        # date_start from actual data
        assert summary.date_start.year == 2023
        assert summary.date_start.month == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_drift_summary(str(tmp_path / "nonexistent.json"))

    def test_load_from_custom_json(self, tmp_path):
        data = {
            "date_start": "2024-01-01 00:00:00+00:00",
            "features": {
                "temperature": {
                    "trend_slope_per_day": 0.5,
                    "trend_r2": 0.9,
                    "status": "DRIFT",
                }
            },
        }
        path = tmp_path / "drift.json"
        path.write_text(json.dumps(data))

        summary = load_drift_summary(str(path))
        assert len(summary.coefficients) == 1
        assert summary.coefficients["temperature"].trend_slope_per_day == 0.5
        assert summary.date_start == pd.Timestamp("2024-01-01 00:00:00+00:00")

    def test_date_start_extracted(self):
        """Sensor start date comes from drift summary, not config."""
        summary_path = Path("results/drift_3yr/drift_summary.json")
        if not summary_path.exists():
            pytest.skip("drift_summary.json not available")

        summary = load_drift_summary(str(summary_path))
        assert summary.date_start == pd.Timestamp("2023-02-14 11:42:30.967556848+00:00")
