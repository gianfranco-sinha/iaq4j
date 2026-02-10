from pathlib import Path
import json
from typing import Union

import torch
import joblib


def save_model(
    model,
    model_name: str,
    output_dir: Union[str, Path],
    config: dict,
    scaler=None,
):
    """
    Save model weights, config, and optional scaler.

    Structure:
    output_dir/
        model_name/
            model.pt
            config.json
            scaler.pkl (optional)
    """
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights
    torch.save(model.state_dict(), output_dir / "model.pt")

    # 2. Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 3. Save scaler (if any)
    if scaler is not None:
        joblib.dump(scaler, output_dir / "scaler.pkl")


def load_model(
    model_cls,
    model_name: str,
    model_dir: Union[str, Path],
    device: str = "cpu",
):
    """
    Load model, config, and optional scaler.

    Returns:
        model, config, scaler
    """
    model_dir = Path(model_dir) / model_name

    # 1. Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # 2. Instantiate model
    model = model_cls(**config["model_params"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    # 3. Load scaler if present
    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return model, config, scaler


def load_dataset(config):
    """Placeholder for loading dataset - to be implemented."""
    import numpy as np
    # This is a placeholder - in real implementation, you'd load from InfluxDB
    # Return dummy data for now

    model_type = config.get("model_type", "mlp")

    if model_type in ["lstm", "cnn"]:
        # Temporal models need sequence data
        window_size = config.get("model_params", {}).get("window_size", 10)
        num_features = config.get("model_params", {}).get("num_features", 4)

        # Create sequential data (batch_size, window_size, num_features)
        X_train = np.random.randn(100, window_size, num_features).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)
        X_val = np.random.randn(20, window_size, num_features).astype(np.float32)
        y_val = np.random.randn(20, 1).astype(np.float32)
    else:
        # MLP and KAN use flattened features
        X_train = np.random.randn(100, 4).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)
        X_val = np.random.randn(20, 4).astype(np.float32)
        y_val = np.random.randn(20, 1).astype(np.float32)

    scaler = None
    return X_train, y_train, X_val, y_val, scaler


def save_artifacts(model, scaler, config, model_name):
    """Save model artifacts using the save_model function."""
    output_dir = Path("trained_models")
    save_model(model, model_name, output_dir, config, scaler)
