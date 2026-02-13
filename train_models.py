"""
Train MLP and KAN models from collected BSEC data.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from app.models import MLPRegressor, KANRegressor
from training.utils import (
    fetch_training_data,
    prepare_features,
    train_model,
    evaluate_model,
    save_trained_model,
)

WINDOW_SIZE = 10


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING MLP AND KAN MODELS FROM BSEC DATA")
    print("=" * 70)

    # Fetch data
    print("\n[1/5] Fetching training data from InfluxDB...")
    df, client = fetch_training_data()

    if len(df) < 1000:
        print("\nWARNING: Less than 1000 samples!")
        print("   Models may not perform well. Consider collecting more data.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Exiting...")
            exit()

    # Prepare features
    print("\n[2/5] Preparing features...")
    X, y, baseline_gas_resistance = prepare_features(df, WINDOW_SIZE)

    # Normalize
    print("\n[3/5] Normalizing data...")
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Train models
    print("\n[4/5] Training models...")
    print("-" * 70)

    input_dim = WINDOW_SIZE * 6  # 4 raw + 2 engineered features

    mlp = MLPRegressor(input_dim=input_dim, hidden_dims=[64, 32, 16])
    mlp_history = train_model(mlp, X_train, y_train, X_val, y_val, "MLP", epochs=200)
    mlp_metrics = evaluate_model(mlp, X_val, y_val, target_scaler)

    print("-" * 70)
    kan = KANRegressor(input_dim=input_dim, hidden_dims=[32, 16])
    kan_history = train_model(kan, X_train, y_train, X_val, y_val, "KAN", epochs=200)
    kan_metrics = evaluate_model(kan, X_val, y_val, target_scaler)

    # Save models
    print("\n[5/5] Saving models...")
    print("-" * 70)
    save_trained_model(
        mlp, feature_scaler, target_scaler, "mlp",
        WINDOW_SIZE, baseline_gas_resistance, "trained_models/mlp", mlp_metrics,
        training_history=mlp_history,
    )
    save_trained_model(
        kan, feature_scaler, target_scaler, "kan",
        WINDOW_SIZE, baseline_gas_resistance, "trained_models/kan", kan_metrics,
        training_history=kan_history,
    )

    client.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    print(f"  MLP: MAE={mlp_metrics['mae']:.2f}, R2={mlp_metrics['r2']:.4f}")
    print(f"  KAN: MAE={kan_metrics['mae']:.2f}, R2={kan_metrics['r2']:.4f}")
    print("\nRestart your service to use the new models:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
