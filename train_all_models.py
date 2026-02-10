from influxdb import DataFrameClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import os
from tqdm import tqdm

from app.models import MLPRegressor, CNNRegressor, KANRegressor

# Import centralized configuration
from app.config import settings

# Use InfluxDB settings from config
INFLUX_HOST = settings.INFLUX_HOST
INFLUX_PORT = settings.INFLUX_PORT
INFLUX_DATABASE = settings.INFLUX_DATABASE
INFLUX_USERNAME = settings.INFLUX_USERNAME
INFLUX_PASSWORD = settings.INFLUX_PASSWORD

# Training settings
MEASUREMENT = "iaq"
HOURS_BACK = 168 * 8  # 8 weeks
WINDOW_SIZE = 10
MIN_IAQ_ACCURACY = 2

# Check for GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


def fetch_training_data():
    """Fetch training data from InfluxDB."""
    print(f"Connecting to InfluxDB at {INFLUX_HOST}:{INFLUX_PORT}...")

    client = DataFrameClient(
        host=INFLUX_HOST,
        port=INFLUX_PORT,
        username=INFLUX_USERNAME,
        password=INFLUX_PASSWORD,
        database=INFLUX_DATABASE,
    )

    query = f"""
    SELECT temperature, humidity, pressure, gas_resistance, iaq, iaq_accuracy
    FROM {MEASUREMENT}
    WHERE time > now() - {HOURS_BACK}h
    """

    print(f"Fetching data from last {HOURS_BACK / 168:.1f} weeks...")
    result = client.query(query)

    if MEASUREMENT not in result:
        raise ValueError(f"No data found in measurement '{MEASUREMENT}'")

    df = result[MEASUREMENT]
    print(f"Fetched {len(df)} raw data points")

    df = df[df["iaq_accuracy"] >= MIN_IAQ_ACCURACY]
    df = df.dropna()

    print(f"After filtering: {len(df)} calibrated samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"IAQ range: {df['iaq'].min():.1f} - {df['iaq'].max():.1f}")

    return df, client


def create_sliding_windows(features, targets, window_size=10):
    """Create sliding window sequences."""
    windows_X, windows_y = [], []

    for i in range(len(features) - window_size + 1):
        window_X = features[i : i + window_size].flatten()
        window_y = targets[i + window_size - 1]
        windows_X.append(window_X)
        windows_y.append(window_y)

    return np.array(windows_X), np.array(windows_y)


def calculate_absolute_humidity(temperature, relative_humidity):
    """Calculate absolute humidity."""
    a, b = 17.27, 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(relative_humidity / 100.0)
    return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)


def prepare_features(df, window_size=10):
    """Prepare features with engineering."""
    print("\nEngineering features...")

    features = df[["temperature", "humidity", "pressure", "gas_resistance"]].values

    baseline_resistance = np.median(features[:, 3])
    print(f"Baseline gas resistance: {baseline_resistance:.0f} Ω")

    gas_ratio = features[:, 3] / baseline_resistance
    abs_humidity = calculate_absolute_humidity(features[:, 0], features[:, 1])

    features_enhanced = np.column_stack(
        [features, gas_ratio.reshape(-1, 1), abs_humidity.reshape(-1, 1)]
    )

    targets = df["iaq"].values

    print(f"Creating sliding windows (size={window_size})...")
    X, y = create_sliding_windows(features_enhanced, targets, window_size)

    print(f"Created {len(X)} windows")

    return X, y, baseline_resistance


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=200):
    """Train a model."""
    print(f"\nTraining {model_name}...")

    model = model.to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"  Best validation loss: {best_val_loss:.6f}")
    return best_val_loss


def evaluate_model(model, X_val, y_val, target_scaler):
    """Evaluate model."""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val).to(device)
        predictions = model(X_tensor).cpu().numpy()

    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(predictions).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(
    model,
    feature_scaler,
    target_scaler,
    model_type,
    window_size,
    baseline_resistance,
    model_dir,
    metrics,
):
    """Save model."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    config = {
        "baseline_resistance": float(baseline_resistance),
        "trained_date": pd.Timestamp.now().isoformat(),
        "window_size": window_size,
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "notes": f"Trained on real BSEC data with {model_type.upper()}",
    }

    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "model_type": model_type,
        "window_size": window_size,
        "input_dim": 60,
    }

    if model_type == "mlp":
        checkpoint["hidden_dims"] = [64, 32, 16]
    elif model_type == "cnn":
        checkpoint["num_filters"] = [64, 128, 256]
        checkpoint["kernel_sizes"] = [3, 3, 3]
    else:
        checkpoint["hidden_dims"] = [32, 16]

    torch.save(checkpoint, f"{model_dir}/model.pt")

    print(f"\n✓ {model_type.upper()}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING ALL MODELS: MLP (Baseline), CNN, KAN")
    print("=" * 70)

    # Fetch data
    print("\n[1/5] Fetching data...")
    df, client = fetch_training_data()

    # Prepare
    print("\n[2/5] Preparing features...")
    X, y, baseline_resistance = prepare_features(df, WINDOW_SIZE)

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
    print("\n[4/5] Training models...")
    print("-" * 70)

    # MLP (Baseline)
    mlp = MLPRegressor(input_dim=60, hidden_dims=[64, 32, 16])
    train_model(mlp, X_train, y_train, X_val, y_val, "MLP (Baseline)", epochs=200)
    mlp_metrics = evaluate_model(mlp, X_val, y_val, target_scaler)

    # CNN
    cnn = CNNRegressor(window_size=10, num_features=6)
    train_model(cnn, X_train, y_train, X_val, y_val, "CNN", epochs=200)
    cnn_metrics = evaluate_model(cnn, X_val, y_val, target_scaler)

    # KAN
    kan = KANRegressor(input_dim=60, hidden_dims=[32, 16])
    train_model(kan, X_train, y_train, X_val, y_val, "KAN", epochs=200)
    kan_metrics = evaluate_model(kan, X_val, y_val, target_scaler)

    # Save models
    print("\n[5/5] Saving models...")
    print("-" * 70)
    save_model(
        mlp,
        feature_scaler,
        target_scaler,
        "mlp",
        WINDOW_SIZE,
        baseline_resistance,
        "trained_models/mlp",
        mlp_metrics,
    )
    save_model(
        cnn,
        feature_scaler,
        target_scaler,
        "cnn",
        WINDOW_SIZE,
        baseline_resistance,
        "trained_models/cnn",
        cnn_metrics,
    )
    save_model(
        kan,
        feature_scaler,
        target_scaler,
        "kan",
        WINDOW_SIZE,
        baseline_resistance,
        "trained_models/kan",
        kan_metrics,
    )

    client.close()

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel Comparison:")
    print(f"  MLP (Baseline): MAE={mlp_metrics['mae']:.2f}, R²={mlp_metrics['r2']:.4f}")
    print(f"  CNN:            MAE={cnn_metrics['mae']:.2f}, R²={cnn_metrics['r2']:.4f}")
    print(f"  KAN:            MAE={kan_metrics['mae']:.2f}, R²={kan_metrics['r2']:.4f}")
    print("\nRestart service:")
    print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
