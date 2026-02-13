# training/train.py
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    model = ModelCls(**config["model_params"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = torch.nn.MSELoss()

    epochs = config["training"]["epochs"]
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = loss_fn(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = loss_fn(val_preds, y_val_tensor)
        val_losses.append(val_loss.item())

    # Compute metrics on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).numpy()
    y_val_np = y_val_tensor.numpy()

    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "mae": float(mean_absolute_error(y_val_np, val_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_val_np, val_preds))),
        "r2": float(r2_score(y_val_np, val_preds)),
    }
    config["metrics"] = metrics

    training_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metrics": metrics,
    }
    save_artifacts(model, scaler, config, model_name, training_history=training_history)


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
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

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
        elif model_type == "lstm":
            model = ModelCls(
                window_size=model_config.get("window_size", 10),
                num_features=model_config.get("num_features", 6),
                hidden_size=model_config.get("hidden_size", 128),
                num_layers=model_config.get("num_layers", 2),
                dropout=model_config.get("dropout", 0.3),
                bidirectional=model_config.get("bidirectional", True),
            )
        elif model_type == "cnn":
            model = ModelCls(
                window_size=model_config.get("window_size", 10),
                num_features=model_config.get("num_features", 6),
                num_filters=model_config.get("num_filters", [64, 128, 256]),
                kernel_sizes=model_config.get("kernel_sizes", [3, 3, 3]),
                dropout=model_config.get("dropout", 0.3),
            )
        else:
            model = ModelCls(**model_config)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = loss_fn(preds, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = loss_fn(val_preds, y_val_tensor)
            val_losses.append(val_loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train: {loss.item():.4f}, Val: {val_loss.item():.4f}")

        # Compute metrics on validation set
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).numpy()
        y_val_np = y_val_tensor.numpy()

        metrics = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "mae": float(mean_absolute_error(y_val_np, val_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val_np, val_preds))),
            "r2": float(r2_score(y_val_np, val_preds)),
        }
        config["metrics"] = metrics

        training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "metrics": metrics,
        }

        # Save model
        save_artifacts(model, scaler, config, model_type, training_history=training_history)

        return True

    except Exception as e:
        print(f"Error training {model_type}: {e}")
        return False
