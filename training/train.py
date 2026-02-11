# training/train.py
from pathlib import Path
import torch
from app.models import (
    MLPRegressor,
    LSTMRegressor,
    KANRegressor,
    CNNRegressor,
)
from app.config import settings
from training.utils import load_dataset, save_artifacts

MODEL_REGISTRY = {
    "mlp": MLPRegressor,
    "lstm": LSTMRegressor,
    "kan": KANRegressor,
    "cnn": CNNRegressor,
}


def train(model_name: str, config: dict):
    """Train a specific model type."""
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


def train_single_model(
    model_type: str,
    epochs: int = 200,
    window_size: int = 10,
    num_records: int = None,
) -> bool:
    """Train a single model using the YAML configuration system."""
    try:
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

        # Load dataset
        X_train, y_train, X_val, y_val, scaler = load_dataset(config)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        # Create model
        ModelCls = MODEL_REGISTRY[model_type]

        # Handle different model constructors
        if model_type == "kan":
            input_dim = model_config.get("input_dim", 6)
            hidden_dims = model_config.get("hidden_dims", [32, 16])
            model = ModelCls(input_dim=input_dim, hidden_dims=hidden_dims)
        elif model_type == "mlp":
            input_dim = model_config.get("input_dim", 6)
            hidden_dims = model_config.get("hidden_dims", [64, 32, 16])
            dropout = model_config.get("dropout", 0.2)
            model = ModelCls(
                input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
            )
        else:
            # LSTM and CNN expect full config dict
            model = ModelCls(**model_config)

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
