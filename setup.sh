#!/bin/bash
# Setup script for AirML
echo "Setting up IAQ-Forge environment..."

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Show available commands
echo ""
echo "Available commands (use ./venv/bin/python):"
echo "  ./venv/bin/python -m airml list                     # List available models"
echo "  ./venv/bin/python -m airml train --model mlp        # Train MLP model"
echo "  ./venv/bin/python -m airml train --model lstm       # Train LSTM model"
echo "  ./venv/bin/python -m iaqforge train --model kan        # Train KAN model"
echo "  ./venv/bin/python -m iaqforge train --model cnn        # Train CNN model"
echo "  ./venv/bin/python -m iaqforge train --model all        # Train all models"
echo "  ./venv/bin/python -m app.main                        # Start API server"
echo "  ./venv/bin/python test_client.py                      # Test API endpoints"
echo ""
echo "Example usage:"
echo "  ./venv/bin/python -m iaqforge train --model lstm --epochs 50"