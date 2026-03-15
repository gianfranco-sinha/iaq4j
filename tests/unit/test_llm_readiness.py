"""Tests for LLM Readiness Phase 1: config cache, StructuredResponse, FailureInfo, InfluxDB reads."""
import pytest

from app.config import settings


# ── A. Config cache invalidation ──────────────────────────────────────────


class TestInvalidateConfigCache:
    def test_clears_model_config(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {"test": True})
        settings.invalidate_config_cache()
        assert settings._model_config_cache is None

    def test_clears_database_config(self, monkeypatch):
        monkeypatch.setattr(settings, "_database_config_cache", {"test": True})
        settings.invalidate_config_cache()
        assert settings._database_config_cache is None

    def test_clears_both(self, monkeypatch):
        monkeypatch.setattr(settings, "_model_config_cache", {"m": 1})
        monkeypatch.setattr(settings, "_database_config_cache", {"d": 1})
        settings.invalidate_config_cache()
        assert settings._model_config_cache is None
        assert settings._database_config_cache is None

    def test_reload_after_invalidate(self, monkeypatch):
        """After invalidation, next load_model_config() re-reads from disk."""
        monkeypatch.setattr(settings, "_model_config_cache", {"stale": True})
        settings.invalidate_config_cache()
        result = settings.load_model_config()
        assert "stale" not in result


# ── B. InfluxDB read queries (unit — no real connection) ──────────────────


class TestInfluxDBReadQueries:
    def test_query_predictions_disabled(self):
        """Returns empty list when InfluxDB is disabled."""
        from app.database import InfluxDBManager
        mgr = InfluxDBManager()
        assert mgr.query_predictions("2026-01-01", "2026-01-02") == []

    def test_query_raw_readings_disabled(self):
        from app.database import InfluxDBManager
        mgr = InfluxDBManager()
        assert mgr.query_raw_readings("2026-01-01", "2026-01-02") == []

    def test_query_prediction_vs_actual_disabled(self):
        from app.database import InfluxDBManager
        mgr = InfluxDBManager()
        assert mgr.query_prediction_vs_actual("2026-01-01", "2026-01-02") == []

    def test_query_predictions_with_model_filter(self):
        """Disabled manager still returns empty regardless of model_type."""
        from app.database import InfluxDBManager
        mgr = InfluxDBManager()
        assert mgr.query_predictions("2026-01-01", "2026-01-02", model_type="mlp") == []


# ── C. FailureInfo + PipelineResult extensions ────────────────────────────


class TestFailureInfoExtensions:
    def test_error_code_field(self):
        from training.pipeline import FailureInfo, PipelineState
        info = FailureInfo(
            failed_state=PipelineState.INGESTION,
            error=ValueError("no data returned"),
            stage_results=[],
            error_code="NO_DATA",
            suggestion="Widen time range",
        )
        assert info.error_code == "NO_DATA"
        assert info.suggestion == "Widen time range"

    def test_defaults_none(self):
        from training.pipeline import FailureInfo, PipelineState
        info = FailureInfo(
            failed_state=PipelineState.TRAINING,
            error=RuntimeError("boom"),
            stage_results=[],
        )
        assert info.error_code is None
        assert info.suggestion is None


class TestPipelineResultWarnings:
    def test_empty_report(self):
        from pathlib import Path
        from training.pipeline import PipelineResult
        result = PipelineResult(
            metrics={}, training_history={}, model_dir=Path("."), stage_results=[]
        )
        assert result.warnings == []

    def test_warnings_from_report(self):
        from pathlib import Path
        from training.pipeline import (
            PipelineResult, PreprocessingReport, IssueSeverity,
        )
        report = PreprocessingReport()
        report.add(IssueSeverity.WARNING, "ingestion", "3 NaN rows dropped", rows_affected=3)
        report.add(IssueSeverity.ERROR, "ingestion", "fatal problem")
        result = PipelineResult(
            metrics={}, training_history={}, model_dir=Path("."),
            stage_results=[], preprocessing_report=report,
        )
        # Only warnings, not errors
        assert len(result.warnings) == 1
        assert "3 NaN rows dropped" in result.warnings[0]
        assert "(3 rows)" in result.warnings[0]


class TestClassifyError:
    def _classify(self, state_name, msg):
        from training.pipeline import TrainingPipeline, PipelineState
        state = PipelineState[state_name]
        return TrainingPipeline._classify_error(state, ValueError(msg))

    def test_no_data(self):
        code, suggestion = self._classify("INGESTION", "no data returned from source")
        assert code == "NO_DATA"

    def test_empty(self):
        code, _ = self._classify("INGESTION", "DataFrame is empty")
        assert code == "NO_DATA"

    def test_insufficient(self):
        code, _ = self._classify("WINDOWING", "too few samples for window")
        assert code == "INSUFFICIENT_DATA"

    def test_diverged_nan(self):
        code, _ = self._classify("TRAINING", "loss is nan at epoch 5")
        assert code == "TRAINING_DIVERGED"

    def test_checkpoint_not_found(self):
        code, _ = self._classify("TRAINING", "checkpoint not found at path")
        assert code == "CHECKPOINT_NOT_FOUND"

    def test_connection_refused(self):
        code, _ = self._classify("SOURCE_ACCESS", "connection refused")
        assert code == "INFLUX_UNREACHABLE"

    def test_unknown_error(self):
        code, suggestion = self._classify("TRAINING", "something unexpected")
        assert code is None
        assert suggestion is None


# ── D. StructuredResponse + DomainErrorCode ───────────────────────────────


class TestDomainErrorCode:
    def test_all_codes(self):
        from app.schemas import DomainErrorCode
        expected = {
            "NO_DATA", "INSUFFICIENT_DATA", "SCHEMA_MISMATCH",
            "INFLUX_UNREACHABLE", "TRAINING_DIVERGED", "NEGATIVE_R2",
            "STALE_CONFIG", "CHECKPOINT_NOT_FOUND",
        }
        actual = {e.value for e in DomainErrorCode}
        assert actual == expected

    def test_is_string_enum(self):
        from app.schemas import DomainErrorCode
        assert DomainErrorCode.NO_DATA == "NO_DATA"
        assert isinstance(DomainErrorCode.NO_DATA, str)


class TestStructuredResponse:
    def test_success(self):
        from app.schemas import StructuredResponse
        r = StructuredResponse(status="success", result={"iaq": 75.0})
        assert r.status == "success"
        assert r.result == {"iaq": 75.0}
        assert r.warnings == []
        assert r.next_steps == []

    def test_error_with_code(self):
        from app.schemas import StructuredResponse, DomainErrorCode
        r = StructuredResponse(
            status="error",
            error_code=DomainErrorCode.NO_DATA,
            detail="InfluxDB returned empty result set",
            next_steps=["Widen time range", "Check measurement name"],
        )
        assert r.error_code == "NO_DATA"
        assert len(r.next_steps) == 2

    def test_warning_with_list(self):
        from app.schemas import StructuredResponse
        r = StructuredResponse(
            status="warning",
            result={"trained": True},
            warnings=["3 NaN rows dropped", "Sampling interval irregular"],
        )
        assert len(r.warnings) == 2

    def test_context_dict(self):
        from app.schemas import StructuredResponse
        r = StructuredResponse(
            status="error",
            detail="fail",
            context={"model_type": "mlp", "stage": "TRAINING"},
        )
        assert r.context["model_type"] == "mlp"

    def test_serialization_excludes_none(self):
        from app.schemas import StructuredResponse
        r = StructuredResponse(status="success", result=42)
        d = r.model_dump(exclude_none=True)
        assert "error_code" not in d
        assert "detail" not in d
        assert d["status"] == "success"
        assert d["result"] == 42


# ── E. Feature name integrity validation ───────────────────────────────


class TestFeatureNameIntegrity:
    def test_feature_names_in_saved_config(self, tmp_path):
        """save_trained_model() persists feature_names list in config.json."""
        import json
        import torch
        from app.models import MLPRegressor
        from training.utils import save_trained_model
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        model = MLPRegressor(input_dim=30)
        fs = StandardScaler()
        fs.fit([[0.0] * 30])
        ts = MinMaxScaler()
        ts.fit([[0.0]])

        names = ["temperature", "humidity", "pressure", "voc_resistance",
                 "abs_humidity", "voc_ratio", "hour_sin", "hour_cos",
                 "dow_sin", "dow_cos"]

        save_trained_model(
            model=model, feature_scaler=fs, target_scaler=ts,
            model_type="mlp", window_size=3, model_dir=str(tmp_path),
            metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.9},
            feature_names=names,
        )

        with open(tmp_path / "config.json") as f:
            config = json.load(f)
        assert config["feature_names"] == names

    def test_fingerprint_changes_on_reorder(self):
        """Different feature orderings produce different schema fingerprints."""
        from training.utils import compute_schema_fingerprint

        names_a = ["temperature", "humidity", "pressure"]
        names_b = ["humidity", "temperature", "pressure"]

        fp_a = compute_schema_fingerprint(
            "bme680", "bsec", 10, 3, "mlp", feature_names=names_a
        )
        fp_b = compute_schema_fingerprint(
            "bme680", "bsec", 10, 3, "mlp", feature_names=names_b
        )
        assert fp_a != fp_b

    def test_fingerprint_unchanged_without_feature_names(self):
        """Omitting feature_names produces the same hash as before (backward compat)."""
        from training.utils import compute_schema_fingerprint

        fp_legacy = compute_schema_fingerprint("bme680", "bsec", 10, 6, "mlp")
        fp_none = compute_schema_fingerprint(
            "bme680", "bsec", 10, 6, "mlp", feature_names=None
        )
        assert fp_legacy == fp_none

    def test_load_warns_on_mismatch(self, tmp_path, caplog):
        """Loading a model with mismatched feature_names logs a warning."""
        import json
        import torch
        import pickle
        from app.models import IAQPredictor

        # Create minimal model artifacts with wrong feature names
        config = {
            "window_size": 3,
            "feature_names": ["a", "b", "c"],
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        from app.models import MLPRegressor
        from app.profiles import get_sensor_profile
        profile = get_sensor_profile()
        model = MLPRegressor(input_dim=3 * profile.total_features)
        torch.save({"state_dict": model.state_dict(), "model_type": "mlp",
                     "window_size": 3, "input_dim": 3 * profile.total_features},
                    tmp_path / "model.pt")
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        with open(tmp_path / "feature_scaler.pkl", "wb") as f:
            pickle.dump(StandardScaler(), f)
        with open(tmp_path / "target_scaler.pkl", "wb") as f:
            pickle.dump(MinMaxScaler(), f)

        predictor = IAQPredictor(model_type="mlp", window_size=3)
        with caplog.at_level("WARNING"):
            predictor.load_model(str(tmp_path))
        assert "Feature name mismatch" in caplog.text

    def test_load_succeeds_without_feature_names(self, tmp_path):
        """Old config.json without feature_names field loads without error."""
        import json
        import torch
        import pickle
        from app.models import IAQPredictor

        config = {"window_size": 3}
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)

        from app.models import MLPRegressor
        from app.profiles import get_sensor_profile
        profile = get_sensor_profile()
        model = MLPRegressor(input_dim=3 * profile.total_features)
        torch.save({"state_dict": model.state_dict(), "model_type": "mlp",
                     "window_size": 3, "input_dim": 3 * profile.total_features},
                    tmp_path / "model.pt")
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        with open(tmp_path / "feature_scaler.pkl", "wb") as f:
            pickle.dump(StandardScaler(), f)
        with open(tmp_path / "target_scaler.pkl", "wb") as f:
            pickle.dump(MinMaxScaler(), f)

        predictor = IAQPredictor(model_type="mlp", window_size=3)
        result = predictor.load_model(str(tmp_path))
        assert result is True
