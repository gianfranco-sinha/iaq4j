# training/train.py
import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch

from training.data_sources import DataSource, SyntheticSource
from training.pipeline import PipelineError, PipelineResult, TrainingPipeline

logger = logging.getLogger(__name__)


def train_single_model(
    model_type: str,
    epochs: int = 200,
    window_size: Optional[int] = None,
    num_records: int = None,
    data_source: DataSource = None,
    experiment_name: str = "iaq4j",
) -> PipelineResult:
    """Train a single model using the TrainingPipeline.

    Args:
        model_type: One of "mlp", "kan", "lstm", "cnn", "bnn".
        epochs: Number of training epochs.
        window_size: Sliding window size. If None, reads from model_config.yaml.
        num_records: Number of synthetic samples (ignored when data_source is provided).
        data_source: Data source to use. Defaults to SyntheticSource.
        experiment_name: MLflow experiment name.

    Returns:
        PipelineResult on success.

    Raises:
        PipelineError: if any pipeline stage fails.
    """
    if window_size is None:
        from app.config import settings
        window_size = settings.get_model_config(model_type).get("window_size", 10)
    mlflow.set_experiment(experiment_name)

    if data_source is None:
        num_samples = num_records if num_records else 1000
        data_source = SyntheticSource(num_samples=num_samples)

    mlflow.start_run(run_name=model_type)
    try:
        def on_epoch(epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
                step=epoch,
            )

        pipeline = TrainingPipeline(
            source=data_source,
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
            on_epoch=on_epoch,
        )

        result = pipeline.orchestrate()

        # Fix run name now that we have the semver (version is set during SAVING).
        # result.version already includes the model type prefix, e.g. "mlp-1.2.0"
        mlflow.set_tag("mlflow.runName", result.version)

        # Full param set — sensor type, iaq standard, schema fingerprint, data fingerprint, etc.
        mlflow.log_params(pipeline.collect_run_params())

        # Final evaluation metrics
        mlflow.log_metrics({
            "best_val_loss": result.training_history.get("best_val_loss", 0),
            "mae": result.metrics.get("mae", 0),
            "rmse": result.metrics.get("rmse", 0),
            "r2": result.metrics.get("r2", 0),
        })

        # Provenance tags
        mlflow.set_tags({
            "version": result.version,
            "merkle_root": result.merkle_root_hash,
        })

        # Artifacts: scalers and data manifest alongside the pytorch model
        model_dir = Path(result.model_dir)
        for name in ("feature_scaler.pkl", "target_scaler.pkl", "data_manifest.json"):
            path = model_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))

        mlflow.pytorch.log_model(pipeline._model, "model")

        mlflow.end_run()
        return result

    except PipelineError as e:
        if e.failure_info:
            mlflow.set_tag("failure_stage", e.failure_info.failed_state.value)
        mlflow.end_run(status="FAILED")
        raise
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
