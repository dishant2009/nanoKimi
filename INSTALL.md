# Installation Guide

## Quick Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install nanoKimi in development mode**:
   ```bash
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   python3 examples/basic_usage.py
   ```

## Alternative: Using Make

If you have `make` installed:

```bash
make dev-setup    # Sets up environment and creates toy data
make example      # Runs basic usage example
```

## Dependencies

Core dependencies:
- PyTorch >= 2.0.0
- transformers >= 4.20.0
- numpy >= 1.21.0

Optional dependencies:
- wandb (for experiment tracking)
- accelerate (for distributed training)
- flash-attn (for efficient attention)

## Platform Notes

### macOS
- Use Python 3.8+ 
- May use MPS backend for Apple Silicon

### Linux
- CUDA support available with appropriate PyTorch installation
- Recommended for training larger models

### Windows
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- May need to install Visual Studio Build Tools for some dependencies

## Troubleshooting

### Common Issues

1. **"No module named torch"**
   - Install PyTorch: `pip install torch`

2. **CUDA out of memory**
   - Reduce batch size in config files
   - Use gradient accumulation

3. **Missing wandb**
   - Install: `pip install wandb`
   - Or disable in config: `use_wandb: false`

### GPU Setup

For CUDA support:
```bash
# Check CUDA version
nvidia-smi

# Install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For ROCm (AMD):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Development Setup

For contributors:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Check code style
make lint
make format
```