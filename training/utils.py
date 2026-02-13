from pathlib import Path
import json
import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.config import settings


def get_device():
    """Detect best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fetch_training_data(
    measurement="bme688",
    hours_back=168 * 8,
    min_iaq_accuracy=2,
):
    """Fetch training data from InfluxDB using centralized config."""
    from influxdb import DataFrameClient

    host = settings.INFLUX_HOST
    port = settings.INFLUX_PORT

    print(f"Connecting to InfluxDB at {host}:{port}...")

    client = DataFrameClient(
        host=host,
        port=port,
        username=settings.INFLUX_USERNAME,
        password=settings.INFLUX_PASSWORD,
        database=settings.INFLUX_DATABASE,
    )

    query = f"""
    SELECT temperature, rel_humidity, pressure, gas_resistance, iaq, iaq_accuracy
    FROM {measurement}
    WHERE time > now() - {hours_back}h
    """

    print(f"Fetching data from last {hours_back / 168:.1f} weeks...")
    result = client.query(query)

    if measurement not in result:
        raise ValueError(f"No data found in measurement '{measurement}'")

    df = result[measurement]
    print(f"Fetched {len(df)} raw data points")

    df = df[df["iaq_accuracy"] >= min_iaq_accuracy]
    df = df.dropna()

    print(f"After filtering (iaq_accuracy >= {min_iaq_accuracy}): {len(df)} samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"IAQ range: {df['iaq'].min():.1f} - {df['iaq'].max():.1f}")

    return df, client


def create_sliding_windows(features, targets, window_size=10):
    """Create sliding window sequences from feature array and target array."""
    windows_X = []
    windows_y = []

    for i in range(len(features) - window_size + 1):
        windows_X.append(features[i : i + window_size].flatten())
        windows_y.append(targets[i + window_size - 1])

    return np.array(windows_X), np.array(windows_y)


def calculate_absolute_humidity(temperature, rel_humidity):
    """Calculate absolute humidity (g/m^3) from temperature (C) and relative humidity (%)."""
    a, b = 17.27, 237.7
    alpha = ((a * temperature) / (b + temperature)) + np.log(rel_humidity / 100.0)
    return (6.112 * np.exp(alpha) * 2.1674) / (273.15 + temperature)


def prepare_features(df, window_size=10):
    """Engineer features and create sliding windows.

    Returns (X, y, baseline_gas_resistance) where X has 6 features per timestep:
    temperature, rel_humidity, pressure, gas_resistance, gas_ratio, abs_humidity.
    """
    print("\nEngineering features...")

    features = df[["temperature", "rel_humidity", "pressure", "gas_resistance"]].values

    baseline_gas_resistance = np.median(features[:, 3])
    print(f"Baseline gas resistance: {baseline_gas_resistance:.0f} Ohm")

    gas_ratio = features[:, 3] / baseline_gas_resistance
    abs_humidity = calculate_absolute_humidity(features[:, 0], features[:, 1])

    features_enhanced = np.column_stack(
        [features, gas_ratio.reshape(-1, 1), abs_humidity.reshape(-1, 1)]
    )

    targets = df["iaq"].values

    print(f"Creating sliding windows (size={window_size})...")
    X, y = create_sliding_windows(features_enhanced, targets, window_size)
    print(f"Created {len(X)} windows with {X.shape[1]} features each")

    return X, y, baseline_gas_resistance


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=200, device=None):
    """Train a model with DataLoader, LR scheduler, and validation tracking."""
    if device is None:
        device = get_device()

    print(f"\nTraining {model_name} on {device}...")
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
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
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
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}")

    print(f"  Best validation loss: {best_val_loss:.6f}")
    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def evaluate_model(model, X_val, y_val, target_scaler, device=None):
    """Evaluate model and return MAE, RMSE, R2 in original IAQ scale."""
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()

    y_true = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(predictions).flatten()

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def save_training_history(model_type, epochs, train_losses, val_losses, metrics, output_dir):
    """Save epoch-level training history and final metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "model_type": model_type,
        "trained_date": pd.Timestamp.now().isoformat(),
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


def save_trained_model(
    model, feature_scaler, target_scaler, model_type,
    window_size, baseline_gas_resistance, model_dir, metrics,
    training_history=None,
):
    """Save a fully trained model with scalers, config, and checkpoint."""
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(f"{model_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    config = {
        "baseline_gas_resistance": float(baseline_gas_resistance),
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
        "input_dim": window_size * 6,
    }

    torch.save(checkpoint, f"{model_dir}/model.pt")

    if training_history is not None:
        save_training_history(
            model_type=model_type,
            epochs=len(training_history["train_losses"]),
            train_losses=training_history["train_losses"],
            val_losses=training_history["val_losses"],
            metrics=metrics,
            output_dir=model_dir,
        )

    print(f"\n  Saved {model_type.upper()}: MAE={metrics['mae']:.2f}, R2={metrics['r2']:.4f}")


# --- Functions used by the iaqforge CLI path (training/train.py) ---

def save_model(
    model,
    model_name: str,
    output_dir: Union[str, Path],
    config: dict,
    scaler=None,
):
    """Save model weights, config, and optional scaler for iaqforge CLI."""
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if scaler is not None:
        joblib.dump(scaler, output_dir / "scaler.pkl")


def load_model(
    model_cls,
    model_name: str,
    model_dir: Union[str, Path],
    device: str = "cpu",
):
    """Load model, config, and optional scaler."""
    model_dir = Path(model_dir) / model_name

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = model_cls(**config["model_params"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return model, config, scaler


def load_dataset(config):
    """Placeholder dataset for iaqforge CLI training (generates dummy data)."""
    model_type = config.get("model_type", "mlp")
    num_features = 6  # 4 raw + 2 engineered

    if model_type in ["lstm", "cnn"]:
        window_size = config.get("model_params", {}).get("window_size", 10)
        feat = config.get("model_params", {}).get("num_features", num_features)
        X_train = np.random.randn(100, window_size, feat).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)
        X_val = np.random.randn(20, window_size, feat).astype(np.float32)
        y_val = np.random.randn(20, 1).astype(np.float32)
    else:
        input_dim = config.get("model_params", {}).get("input_dim", num_features)
        X_train = np.random.randn(100, input_dim).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)
        X_val = np.random.randn(20, input_dim).astype(np.float32)
        y_val = np.random.randn(20, 1).astype(np.float32)

    scaler = None
    return X_train, y_train, X_val, y_val, scaler


def save_artifacts(model, scaler, config, model_name, training_history=None):
    """Save model artifacts using save_model (iaqforge CLI path)."""
    output_dir = Path("trained_models")
    save_model(model, model_name, output_dir, config, scaler)

    if training_history is not None:
        save_training_history(
            model_type=model_name,
            epochs=len(training_history["train_losses"]),
            train_losses=training_history["train_losses"],
            val_losses=training_history["val_losses"],
            metrics=training_history.get("metrics", {}),
            output_dir=output_dir / model_name,
        )
