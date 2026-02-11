#!/usr/bin/env python3
"""
Model Trainer CLI Module

Handles training of specific models from the registry.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_single_model
from app.config import settings


class ModelTrainer:
    """Handles training of specific models from the registry."""

    def __init__(self):
        # Load database configuration from YAML
        db_config = settings.get_database_config()

        self.influx_host = db_config.get("host")
        self.influx_port = db_config.get("port")
        self.influx_database = db_config.get("database")
        self.influx_username = db_config.get("username")
        self.influx_password = db_config.get("password")
        self.db_enabled = db_config.get("enabled", False)

        print(f"Database configuration loaded:")
        print(f"  - Host: {self.influx_host}")
        print(f"  - Port: {self.influx_port}")
        print(f"  - Database: {self.influx_database}")
        print(f"  - Enabled: {self.db_enabled}")

    def train_model(
        self,
        model_type: str,
        epochs: int = 200,
        window_size: int = 10,
        num_records: int = None,
    ):
        """Train a specific model type.

        Args:
            model_type: Type of model to train ('mlp', 'kan', 'lstm', 'cnn')
            epochs: Number of training epochs
            window_size: Sliding window size for temporal models
            num_records: Number of records to fetch from database (optional)
        """
        if model_type not in ["mlp", "kan", "lstm", "cnn"]:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"Starting training for {model_type.upper()} model...")
        print(f"Configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Window Size: {window_size}")
        print(f"  - Data Records: {num_records if num_records else 'All available'}")
        print(f"  - Database: {self.influx_database}")

        # Train the model using the existing training infrastructure
        success = train_single_model(
            model_type=model_type,
            epochs=epochs,
            window_size=window_size,
            num_records=num_records,
            influx_host=self.influx_host,
            influx_port=self.influx_port,
            influx_database=self.influx_database,
            influx_username=self.influx_username,
            influx_password=self.influx_password,
        )

        if success:
            model_path = f"trained_models/{model_type}"
            print(f"\n✅ Training completed successfully!")
            print(f"   Model saved to: {model_path}/")
            print(f"   Timestamp: {datetime.now().isoformat()}")
        else:
            raise RuntimeError(f"Training failed for {model_type.upper()} model")
