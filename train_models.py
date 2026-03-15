"""
Train all IAQ models (MLP, KAN, BNN, CNN, LSTM) from collected BSEC data.
"""
import logging

from app.config import settings
from training.data_sources import InfluxDBSource
from training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)


def log_progress(state, result):
    if result and result.extra:
        print(f"  [{state.value}] {result.extra}")


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING MLP, CNN, BNN, LSTM, KAN MODELS FROM BSEC DATA")
    print("=" * 70)

    results = {}

    for model_type in ["mlp", "kan", "bnn", "cnn", "lstm"]:
        model_cfg = settings.get_model_config(model_type)
        window_size = model_cfg.get("window_size", 10)

        print(f"\n{'─' * 70}")
        print(f"Training {model_type.upper()} (window_size={window_size})...")
        print(f"{'─' * 70}")

        try:
            source = InfluxDBSource(hours_back=168 * 8, database="home_study_room_iaq")
            pipeline = TrainingPipeline(
                source, model_type=model_type, epochs=200, window_size=window_size,
                resume=True,
            )
            pipeline.on_stage_complete(log_progress)

            result = pipeline.orchestrate()
            if result.interrupted:
                print(f"⏸️  {model_type.upper()} interrupted — checkpoint saved")
                results[model_type] = None
            else:
                results[model_type] = result.metrics
            source.close()
        except Exception as e:
            print(f"❌ {model_type.upper()} training failed: {e}")
            results[model_type] = None

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name.upper()}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")
        else:
            print(f"  {name.upper()}: FAILED")
    print("\nRestart your service to use the new models:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
