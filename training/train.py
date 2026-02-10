# training/train.py
import argparse
import json
from pathlib import Path
import yaml
import torch
from app.models import (
    MLPRegressor,
    LSTMRegressor,
    KANRegressor,
    CNNRegressor,
)
from training.utils import load_dataset, save_artifacts

MODEL_REGISTRY = {
    "mlp": MLPRegressor,
    "lstm": LSTMRegressor,
    "kan": KANRegressor,
    "cnn": CNNRegressor,
}


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(base_path: Path, model_path: Path) -> dict:
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(model_path) as f:
        model_cfg = yaml.safe_load(f)
    return deep_update(base_cfg, model_cfg)


def train(model_name: str, config: dict):
    ModelCls = MODEL_REGISTRY[model_name]

    X_train, y_train, X_val, y_val, scaler = load_dataset(config)

    model = ModelCls(**config["model_params"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = torch.nn.MSELoss()

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    save_artifacts(model, scaler, config, model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.base_config), Path(args.model_config))
    train(args.model, config)


def train_single_model(
    model_type: str,
    epochs: int = 200,
    window_size: int = 10,
    num_records: int = None,
    influx_host: str = None,
    influx_port: int = None,
    influx_database: str = None,
    influx_username: str = None,
    influx_password: str = None,
) -> bool:
    """Train a single model using the new YAML configuration system."""
    try:
        from app.config import settings

        # Get model configuration from YAML
        model_config = settings.get_model_config(model_type)

        # Create model configuration dict
        config = {
            "model_type": model_type,
            "model_params": model_config,
            "training": {
                "epochs": epochs,
                "lr": 0.001,
                "batch_size": 32,
            },
        }

        # Load dataset (placeholder for now)
        X_train, y_train, X_val, y_val, scaler = load_dataset(config)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Create model
        ModelCls = MODEL_REGISTRY[model_type]
        model = ModelCls(config=model_config)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = loss_fn(preds, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Save model
        save_artifacts(model, scaler, config, model_type)

        return True

    except Exception as e:
        print(f"Error training {model_type}: {e}")
        return False


if __name__ == "__main__":
    main()
