# ============================================================================
# File: app/models.py - Complete Version with All Models
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
from efficient_kan import KAN


class MLPRegressor(nn.Module):
    """MLP for IAQ prediction (Baseline Model)."""

    def __init__(self, input_dim=None, hidden_dims=None, dropout=None, config=None):
        super(MLPRegressor, self).__init__()

        # Load from config if provided, otherwise use defaults
        if config:
            input_dim = config.get("input_dim", 4)
            hidden_dims = config.get("hidden_dims", [64, 32, 16])
            dropout = config.get("dropout", 0.2)
            use_batch_norm = config.get("use_batch_norm", True)
            activation = config.get("activation", "relu")
        else:
            input_dim = input_dim or 4
            hidden_dims = hidden_dims or [64, 32, 16]
            dropout = dropout or 0.2
            use_batch_norm = True
            activation = "relu"

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Add activation function
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())  # Default to ReLU

            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNNRegressor(nn.Module):
    """CNN for IAQ prediction from temporal sequences."""

    def __init__(
        self,
        window_size=None,
        num_features=None,
        num_filters=None,
        kernel_sizes=None,
        dropout=None,
        config=None,
    ):
        super(CNNRegressor, self).__init__()

        # Load from config if provided, otherwise use defaults
        if config:
            window_size = config.get("window_size", 10)
            num_features = config.get("num_features", 4)
            num_filters = config.get("num_filters", [64, 128, 256])
            kernel_sizes = config.get("kernel_sizes", [3, 3, 3])
            dropout = config.get("dropout", 0.3)
            fc_layers = config.get("fc_layers", [128, 64])
        else:
            window_size = window_size or 10
            num_features = num_features or 4
            num_filters = num_filters or [64, 128, 256]
            kernel_sizes = kernel_sizes or [3, 3, 3]
            dropout = dropout or 0.3
            fc_layers = [128, 64]

        self.window_size = window_size
        self.num_features = num_features

        # Build convolutional layers
        conv_layers = []
        in_channels = num_features

        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.BatchNorm1d(num_filter))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = num_filter

        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build fully connected layers dynamically
        fc_layer_list = []
        prev_size = num_filters[-1]

        for fc_size in fc_layers:
            fc_layer_list.append(nn.Linear(prev_size, fc_size))
            fc_layer_list.append(nn.ReLU())
            fc_layer_list.append(nn.Dropout(dropout))
            prev_size = fc_size

        # Final output layer
        fc_layer_list.append(nn.Linear(prev_size, 1))

        self.fc = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        # Reshape from (batch, window_size * num_features) to (batch, num_features, window_size)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.window_size, self.num_features)
        x = x.permute(0, 2, 1)  # (batch, num_features, window_size)

        # Convolutional layers
        x = self.conv(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Fully connected
        x = self.fc(x)

        return x


class LSTMRegressor(nn.Module):
    """LSTM for IAQ prediction from temporal sequences."""

    def __init__(
        self,
        window_size=None,
        num_features=None,
        hidden_size=None,
        num_layers=None,
        dropout=None,
        bidirectional=None,
        config=None,
    ):
        super(LSTMRegressor, self).__init__()

        # Load from config if provided, otherwise use defaults
        if config:
            window_size = config.get("window_size", 10)
            num_features = config.get("num_features", 4)
            hidden_size = config.get("hidden_size", 128)
            num_layers = config.get("num_layers", 2)
            dropout = config.get("dropout", 0.3)
            bidirectional = config.get("bidirectional", True)
            fc_layers = config.get("fc_layers", [64, 32])
        else:
            window_size = window_size or 10
            num_features = num_features or 4
            hidden_size = hidden_size or 128
            num_layers = num_layers or 2
            dropout = dropout or 0.3
            bidirectional = bidirectional if bidirectional is not None else True
            fc_layers = [64, 32]

        self.window_size = window_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Build fully connected layers dynamically
        fc_layer_list = []
        prev_size = lstm_output_size

        for fc_size in fc_layers:
            fc_layer_list.append(nn.Linear(prev_size, fc_size))
            fc_layer_list.append(nn.ReLU())
            fc_layer_list.append(nn.Dropout(dropout))
            prev_size = fc_size

        # Final output layer
        fc_layer_list.append(nn.Linear(prev_size, 1))

        self.fc = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        # Reshape from (batch, window_size * num_features) to (batch, window_size, num_features)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.window_size, self.num_features)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            last_hidden = hidden[-1, :, :]

        # Fully connected layers
        output = self.fc(last_hidden)

        return output


class KANRegressor(nn.Module):
    """KAN for IAQ prediction."""

    def __init__(self, input_dim=None, hidden_dims=None, config=None):
        super(KANRegressor, self).__init__()

        # Load from config if provided, otherwise use defaults
        if config:
            input_dim = config.get("input_dim", 4)
            hidden_dims = config.get("hidden_dims", [32, 16])
            grid_size = config.get("grid_size", 5)
            spline_order = config.get("spline_order", 3)
        else:
            input_dim = input_dim or 4
            hidden_dims = hidden_dims or [32, 16]
            grid_size = 5
            spline_order = 3

        layers = [input_dim] + hidden_dims + [1]
        self.kan = KAN(layers, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        return self.kan(x)


class IAQPredictor:
    """Main predictor class that handles model loading and inference with sliding window."""

    def __init__(
        self, model_type: str = "mlp", window_size: int = 10, model_path: str = None
    ):
        """
        Initialize IAQ predictor.

        Args:
            model_type: Type of model ('mlp', 'lstm', 'cnn', 'kan')
            window_size: Size of sliding window for temporal models
            model_path: Path to saved model directory
        """
        self.model_type: str = model_type
        self.window_size: int = window_size
        self.model_path: str = model_path
        self.model = None  # Type: torch.nn.Module
        self.scaler = None  # Type: sklearn.base.BaseEstimator
        self.config = None  # Type: dict
        self.device: str = "cpu"

        # Sliding window buffer for temporal data
        self.buffer: list = []

        # Model registry for creating instances
        self._model_registry: dict = {
            "mlp": MLPRegressor,
            "lstm": LSTMRegressor,
            "cnn": CNNRegressor,
            "kan": KANRegressor,
        }

    def load_model(self, model_path: str) -> bool:
        """Load model from saved artifacts."""
        import json
        import joblib
        from pathlib import Path
        from app.config import settings

        model_dir = Path(model_path)

        try:
            # Load config from JSON file
            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                self.config = json.load(f)

            # Get model class
            if self.model_type not in self._model_registry:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            ModelClass = self._model_registry[self.model_type]

            # Load model weights
            weights_path = model_dir / "model.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weights_path}")

            model_data = torch.load(weights_path, map_location=self.device)

            # Get YAML configuration for this model type
            yaml_config = settings.get_model_config(self.model_type)

            # Extract model parameters from saved data
            saved_params = self.config.get("model_params", {})

            # Merge saved parameters with YAML config (YAML takes precedence)
            model_params = {}
            model_params.update(saved_params)  # Start with saved params

            # Override with YAML configuration
            model_params.update(yaml_config)

            # For backward compatibility, handle saved model data structure
            if isinstance(model_data, dict):
                if self.model_type in ["mlp", "kan"]:
                    # MLP and KAN specific parameters from saved data
                    if "input_dim" in model_data and "input_dim" not in saved_params:
                        model_params["input_dim"] = model_data["input_dim"]
                    if (
                        "hidden_dims" in model_data
                        and "hidden_dims" not in saved_params
                    ):
                        model_params["hidden_dims"] = model_data["hidden_dims"]
                    if "dropout" in model_data and "dropout" not in saved_params:
                        model_params["dropout"] = model_data["dropout"]
                elif self.model_type in ["lstm", "cnn"]:
                    # LSTM and CNN specific parameters from saved data
                    if (
                        "window_size" in model_data
                        and "window_size" not in saved_params
                    ):
                        model_params["window_size"] = model_data["window_size"]
                    if (
                        "num_features" in model_data
                        and "num_features" not in saved_params
                    ):
                        model_params["num_features"] = model_data["num_features"]

            # Set required defaults if not present
            if self.model_type in ["mlp", "kan"]:
                if "input_dim" not in model_params:
                    model_params["input_dim"] = 4  # Default to 4 features
            elif self.model_type in ["lstm", "cnn"]:
                model_params.setdefault("window_size", self.window_size)
                model_params.setdefault(
                    "num_features", 4
                )  # temp, humidity, pressure, gas

            # Create model with configuration
            self.model = ModelClass(config=model_params)

            # Load model weights
            if isinstance(model_data, dict) and "state_dict" in model_data:
                self.model.load_state_dict(model_data["state_dict"])
            else:
                self.model.load_state_dict(model_data)
            self.model.to(self.device)
            self.model.eval()

            # Load scaler if available
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(
        self,
        temperature: float,
        rel_humidity: float,
        pressure: float,
        gas_resistance: float,
    ) -> dict:
        """
        Predict IAQ from sensor readings.

        Args:
            temperature: Temperature in Celsius
            rel_humidity: Relative humidity in %
            pressure: Pressure in hPa
            gas_resistance: Gas resistance in Ohms

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"iaq": None, "status": "error", "message": "Model not loaded"}

        try:
            # Create feature vector
            features = np.array([[temperature, rel_humidity, pressure, gas_resistance]])

            # Apply scaler if available
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)

            # Handle temporal models with sliding window
            if self.model_type in ["lstm", "cnn"]:
                # Add to buffer
                self.buffer.append(features[0])

                # Maintain window size
                if len(self.buffer) > self.window_size:
                    self.buffer.pop(0)

                # Check if we have enough data
                if len(self.buffer) < self.window_size:
                    return {
                        "iaq": None,
                        "status": "buffering",
                        "buffer_size": len(self.buffer),
                        "required": self.window_size,
                        "message": f"Collecting data... {len(self.buffer)}/{self.window_size}",
                    }

                # Create windowed input
                window_data = np.array(self.buffer)
                features_tensor = (
                    torch.FloatTensor(window_data.flatten())
                    .unsqueeze(0)
                    .to(self.device)
                )

            # Make prediction
            with torch.no_grad():
                prediction = self.model(features_tensor)
                iaq_value = float(prediction.cpu().numpy()[0, 0])

            # Clamp to valid IAQ range
            iaq_value = max(0, min(500, iaq_value))

            # Determine air quality category
            if iaq_value <= 50:
                category = "Excellent"
            elif iaq_value <= 100:
                category = "Good"
            elif iaq_value <= 200:
                category = "Moderate"
            elif iaq_value <= 300:
                category = "Poor"
            else:
                category = "Very Poor"

            return {
                "iaq": iaq_value,
                "category": category,
                "status": "ready",
                "model_type": self.model_type,
                "raw_inputs": {
                    "temperature": temperature,
                    "rel_humidity": rel_humidity,
                    "pressure": pressure,
                    "gas_resistance": gas_resistance,
                },
                "buffer_size": len(self.buffer)
                if self.model_type in ["lstm", "cnn"]
                else None,
                "required": self.window_size
                if self.model_type in ["lstm", "cnn"]
                else None,
            }

        except Exception as e:
            return {
                "iaq": None,
                "status": "error",
                "message": f"Prediction failed: {str(e)}",
            }

    def create_model_from_config(self) -> bool:
        """Create a new model instance from YAML configuration (for training or fresh starts)."""
        from app.config import settings

        try:
            # Get model class
            if self.model_type not in self._model_registry:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            ModelClass = self._model_registry[self.model_type]

            # Get YAML configuration for this model type
            model_config = settings.get_model_config(self.model_type)

            # Create model with configuration
            self.model = ModelClass(config=model_config)
            self.model.to(self.device)

            return True

        except Exception as e:
            print(f"Failed to create model from config: {e}")
            return False

    def reset_buffer(self) -> None:
        """Reset the sliding window buffer."""
        self.buffer = []
