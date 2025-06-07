#!/bin/bash

# TimesNet Proof of Concept - Complete Setup Script
# This script sets up the entire project from scratch

set -e  # Exit on any error

echo "🚀 TimesNet Proof of Concept Setup"
echo "=================================="

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed. Please install pyenv first:"
    echo "   curl https://pyenv.run | bash"
    exit 1
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Create project directory if it doesn't exist
PROJECT_NAME="timesnet_poc"
if [ ! -d "$PROJECT_NAME" ]; then
    echo "📁 Creating project directory..."
    mkdir -p "$PROJECT_NAME"
    cd "$PROJECT_NAME"
else
    echo "📁 Using existing project directory..."
    cd "$PROJECT_NAME"
fi

# Set Python version
echo "🐍 Setting Python version to 3.11.11..."
pyenv local 3.11.11

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✅ Python version: $PYTHON_VERSION"

# Initialize Poetry project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "📦 Initializing Poetry project..."
    poetry init --name timesnet-poc \
                --version 0.1.0 \
                --description "TimesNet Proof of Concept Implementation" \
                --author "Your Name <your.email@example.com>" \
                --python "^3.11" \
                --no-interaction
fi

# Create project structure
echo "🏗️  Creating project structure..."
mkdir -p src/timesnet_poc/{models,data,utils,visualization}
mkdir -p tests
mkdir -p notebooks
mkdir -p data/{raw,processed}
mkdir -p experiments
mkdir -p configs

# Create __init__.py files
touch src/__init__.py
touch src/timesnet_poc/__init__.py
touch src/timesnet_poc/models/__init__.py
touch src/timesnet_poc/data/__init__.py
touch src/timesnet_poc/utils/__init__.py
touch src/timesnet_poc/visualization/__init__.py
touch tests/__init__.py

# Create basic configuration files
echo "⚙️  Creating configuration files..."

# Create a simple config file
cat > configs/default_config.yaml << EOF
# TimesNet Default Configuration
model:
  seq_len: 96
  pred_len: 24
  d_model: 64
  d_ff: 256
  e_layers: 2
  top_k: 5
  num_kernels: 6
  dropout: 0.1

data:
  symbols: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
  target_symbol: 'AAPL'
  period: '2y'
  test_size: 0.2
  val_size: 0.1

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  patience: 15
  save_dir: 'experiments/default'

visualization:
  save_plots: true
  interactive: true
  plot_samples: 5
EOF

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Experiments
experiments/*
!experiments/.gitkeep

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
EOF

# Create placeholder files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch experiments/.gitkeep

# Install dependencies
echo "📚 Installing dependencies..."
poetry add torch torchvision torchaudio --source pytorch
poetry add numpy pandas matplotlib seaborn scikit-learn yfinance plotly jupyter tqdm pyyaml

# Install development dependencies
echo "🛠️  Installing development dependencies..."
poetry add --group dev pytest black flake8 mypy notebook ipykernel pre-commit

# Install the package in editable mode
echo "📦 Installing package in editable mode..."
poetry install

# Create a simple test file
cat > tests/test_basic.py << EOF
"""Basic tests for TimesNet PoC."""
import pytest
import torch
import numpy as np


def test_torch_installation():
    """Test that PyTorch is properly installed."""
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)


def test_numpy_installation():
    """Test that NumPy is properly installed."""
    x = np.random.randn(2, 3)
    assert x.shape == (2, 3)


def test_project_structure():
    """Test that project structure is created correctly."""
    import os
    assert os.path.exists('src/timesnet_poc')
    assert os.path.exists('src/timesnet_poc/models')
    assert os.path.exists('src/timesnet_poc/data')
    assert os.path.exists('notebooks')
    assert os.path.exists('experiments')
EOF

# Create a simple demo script
cat > demo.py << EOF
#!/usr/bin/env python3
"""
TimesNet Demo Script
Simple demonstration of the TimesNet implementation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.timesnet_poc.models.timesnet import TimesNet, FFTPeriodDiscovery


def create_synthetic_data(seq_len=200, periods=[20, 30, 50], noise_level=0.1):
    """Create synthetic time series with multiple periodicities."""
    t = np.linspace(0, 10, seq_len)
    signal = np.zeros_like(t)
    
    for period in periods:
        frequency = 2 * np.pi / period
        amplitude = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        signal += amplitude * np.sin(frequency * t + phase)
    
    # Add noise
    noise = noise_level * np.random.randn(len(t))
    signal += noise
    
    return signal, periods


def demo_period_discovery():
    """Demonstrate FFT-based period discovery."""
    print("🔍 Demonstrating Period Discovery...")
    
    # Create synthetic data
    signal, true_periods = create_synthetic_data(seq_len=200, periods=[20, 30, 50])
    
    # Convert to tensor
    x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    
    # Create period discovery module
    period_discovery = FFTPeriodDiscovery(top_k=5)
    
    # Discover periods
    periods, amplitudes = period_discovery(x)
    
    print(f"True periods: {true_periods}")
    print(f"Discovered periods: {periods[0].numpy()}")
    print(f"Amplitudes: {amplitudes[0].numpy()}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot original signal
    plt.subplot(1, 3, 1)
    plt.plot(signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot FFT spectrum
    plt.subplot(1, 3, 2)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(fft_result)//2])
    plt.title('FFT Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    
    # Plot discovered periods
    plt.subplot(1, 3, 3)
    discovered_periods = periods[0].numpy()
    discovered_amplitudes = amplitudes[0].numpy()
    plt.bar(range(len(discovered_periods)), discovered_periods, alpha=0.7)
    plt.title('Discovered Periods')
    plt.xlabel('Period Rank')
    plt.ylabel('Period Length')
    
    plt.tight_layout()
    plt.savefig('demo_period_discovery.png', dpi=150, bbox_inches='tight')
    print("📊 Period discovery plot saved as 'demo_period_discovery.png'")
    plt.show()


def demo_timesnet_model():
    """Demonstrate TimesNet model forward pass."""
    print("🤖 Demonstrating TimesNet Model...")
    
    # Model parameters
    seq_len = 96
    pred_len = 24
    batch_size = 4
    
    # Create model
    model = TimesNet(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=32,  # Smaller for demo
        e_layers=1,
        top_k=3,
        enc_in=1,
        c_out=1
    )
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, 1)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("✅ Model forward pass successful!")
    
    return model, x, output


def main():
    """Run the complete demo."""
    print("🎯 TimesNet Proof of Concept Demo")
    print("=" * 40)
    
    try:
        # Demo 1: Period Discovery
        demo_period_discovery()
        print()
        
        # Demo 2: TimesNet Model
        model, input_data, predictions = demo_timesnet_model()
        print()
        
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python src/timesnet_poc/train.py' for full training")
        print("2. Open 'notebooks/timesnet_analysis.ipynb' for interactive analysis")
        print("3. Check the README.md for detailed usage instructions")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()
EOF

# Make demo script executable
chmod +x demo.py

# Create a quick start script
cat > quickstart.sh << EOF
#!/bin/bash

echo "🚀 TimesNet Quick Start"
echo "======================"

echo "1. Activating Poetry environment..."
poetry shell

echo "2. Running basic tests..."
poetry run pytest tests/ -v

echo "3. Running demo..."
poetry run python demo.py

echo "4. Quick training run (10 epochs)..."
poetry run python -c "
import sys
sys.path.append('src')
from timesnet_poc.train import main
config = {
    'epochs': 10,
    'batch_size': 16,
    'd_model': 32,
    'symbols': ['AAPL', 'MSFT'],
    'period': '6mo'
}
main()
"

echo "✅ Quick start completed!"
echo "Check the 'experiments/' directory for results."
EOF

chmod +x quickstart.sh

# Run basic tests
echo "🧪 Running basic tests..."
poetry run pytest tests/ -v

# Create initial commit if git is available
if command -v git &> /dev/null; then
    if [ ! -d ".git" ]; then
        echo "📝 Initializing git repository..."
        git init
        git add .
        git commit -m "Initial TimesNet PoC setup"
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 What was created:"
echo "   - Complete project structure"
echo "   - Poetry environment with all dependencies"
echo "   - Basic tests and configuration files"
echo "   - Demo script and quick start guide"
echo ""
echo "🚀 Next steps:"
echo "   1. Run the demo: poetry run python demo.py"
echo "   2. Quick start: ./quickstart.sh"
echo "   3. Full training: poetry run python src/timesnet_poc/train.py"
echo "   4. Interactive analysis: poetry run jupyter notebook notebooks/"
echo ""
echo "💡 Tips:"
echo "   - Activate environment: poetry shell"
echo "   - Run tests: poetry run pytest"
echo "   - Format code: poetry run black src/"
echo "   - Check README.md for detailed documentation"
echo ""

# Final verification
echo "🔍 Verification:"
echo "   Python version: $(python --version)"
echo "   Poetry version: $(poetry --version)"
echo "   PyTorch version: $(poetry run python -c 'import torch; print(torch.__version__)')"
echo "   Project ready: ✅"