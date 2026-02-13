"""
Train all models (MLP, CNN, KAN) from collected BSEC data.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from app.models import MLPRegressor, CNNRegressor, KANRegressor
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
    print("TRAINING ALL MODELS: MLP (Baseline), CNN, KAN")
    print("=" * 70)

    # Fetch data
    print("\n[1/5] Fetching data...")
    df, client = fetch_training_data()

    # Prepare
    print("\n[2/5] Preparing features...")
    X, y, baseline_gas_resistance = prepare_features(df, WINDOW_SIZE)

    # Normalize
    print("\n[3/5] Normalizing...")
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")

    # Train all models
    input_dim = WINDOW_SIZE * 6  # 4 raw + 2 engineered features

    print("\n[4/5] Training models...")
    print("-" * 70)

    # MLP (Baseline)
    mlp = MLPRegressor(input_dim=input_dim, hidden_dims=[64, 32, 16])
    mlp_history = train_model(mlp, X_train, y_train, X_val, y_val, "MLP (Baseline)", epochs=200)
    mlp_metrics = evaluate_model(mlp, X_val, y_val, target_scaler)

    # CNN
    cnn = CNNRegressor(window_size=WINDOW_SIZE, num_features=6)
    cnn_history = train_model(cnn, X_train, y_train, X_val, y_val, "CNN", epochs=200)
    cnn_metrics = evaluate_model(cnn, X_val, y_val, target_scaler)

    # KAN
    kan = KANRegressor(input_dim=input_dim, hidden_dims=[32, 16])
    kan_history = train_model(kan, X_train, y_train, X_val, y_val, "KAN", epochs=200)
    kan_metrics = evaluate_model(kan, X_val, y_val, target_scaler)

    # Save models
    print("\n[5/5] Saving models...")
    print("-" * 70)
    for model, name, metrics, history in [
        (mlp, "mlp", mlp_metrics, mlp_history),
        (cnn, "cnn", cnn_metrics, cnn_history),
        (kan, "kan", kan_metrics, kan_history),
    ]:
        save_trained_model(
            model, feature_scaler, target_scaler, name,
            WINDOW_SIZE, baseline_gas_resistance, f"trained_models/{name}", metrics,
            training_history=history,
        )

    client.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    print(f"  MLP (Baseline): MAE={mlp_metrics['mae']:.2f}, R2={mlp_metrics['r2']:.4f}")
    print(f"  CNN:            MAE={cnn_metrics['mae']:.2f}, R2={cnn_metrics['r2']:.4f}")
    print(f"  KAN:            MAE={kan_metrics['mae']:.2f}, R2={kan_metrics['r2']:.4f}")
    print("\nRestart service:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
