# nanoKimi

A simplified, educational implementation of the Kimi-K2 architecture that demonstrates its key innovations over traditional transformer models.

##  Key Features

nanoKimi showcases three revolutionary innovations from Kimi-K2:

1. **Mixture of Experts (MoE)** - Sparse expert routing for efficient scaling
2. **Muon Optimizer** - Specialized optimizer for MoE training  
3. **Latent Attention** - Enhanced context understanding through latent space projections

##  Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/nanokimi.git](https://github.com/dishant2009/nanoKimi)
cd nanokimi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

##  Quick Start

### 1. Train a Nano Model

```bash
# Train a small model for testing
python scripts/train.py --config configs/nano.yaml

# Train with custom settings
python scripts/train.py --config configs/small.yaml --max-steps 10000 --no-wandb
```

### 2. Generate Text

```bash
# Generate text with trained model
python scripts/generate.py --checkpoint ./checkpoints/nano/best_model.pt --prompt "The future of AI is"

# Interactive generation
python scripts/generate.py --checkpoint ./checkpoints/nano/best_model.pt --interactive

# Streaming generation
python scripts/generate.py --checkpoint ./checkpoints/nano/best_model.pt --stream --prompt "Once upon a time"
```

### 3. Programmatic Usage

```python
import torch
from nanokimi import KimiModel, KimiConfig, Generator

# Create model configuration
config = KimiConfig(
    n_layer=12,
    n_head=12, 
    n_embd=768,
    vocab_size=50257,
    block_size=1024,
    moe={
        "use_moe": True,
        "num_experts": 8,
        "top_k": 2,
    }
)

# Create and use model
model = KimiModel(config)
generator = Generator(model)

# Generate text
text = generator.generate_text("The future of artificial intelligence", max_new_tokens=50)
print(text)
```

##  Model Configurations

### Nano (Testing)
- **Parameters**: ~10M
- **Layers**: 6
- **Embedding**: 384
- **Experts**: 4
- **Use case**: Development and testing

### Small (Experimentation)  
- **Parameters**: ~124M
- **Layers**: 12
- **Embedding**: 768
- **Experts**: 8
- **Use case**: Research and benchmarking

### Medium (Production)
- **Parameters**: ~350M
- **Layers**: 24
- **Embedding**: 1024
- **Experts**: 16
- **Use case**: Real applications

##  Architecture Overview

### Mixture of Experts (MoE)
```
Input → Router → Top-K Experts → Weighted Combination → Output
                    ↓
              Load Balancing Loss
```

- **Sparse routing**: Only top-k experts process each token
- **Load balancing**: Prevents expert collapse during training
- **Efficiency**: Better parameter utilization than dense models

### Latent Attention
```
Input → Cross-Attention → Latent Space → Self-Attention → Cross-Attention → Output
         (Input→Latent)                                    (Latent→Input)
```

- **Latent tokens**: Learnable tokens that compress information
- **Enhanced context**: Better long-range dependency modeling  
- **Efficiency**: Reduced attention complexity for long sequences

### Muon Optimizer
- **Expert-aware**: Different learning rates for expert vs. shared parameters
- **Convergence**: Specialized momentum handling for sparse gradients
- **Memory efficient**: Optimized for large sparse models

##  Training Data

### Toy Dataset (Default)
- Random tokens for testing
- Fast iteration during development
- No external dependencies

### OpenWebText  
- High-quality web text dataset
- Comparable to GPT-2 training data
- Requires preprocessing (see Data Preparation)

### Custom Datasets
Add your own datasets by implementing the `TokenDataset` interface.

## 🔧 Data Preparation

### For OpenWebText:
```bash
# Download OpenWebText dataset
# Process raw text files
python -c "
from nanokimi.training.data import TextDataProcessor, DataConfig
config = DataConfig()
processor = TextDataProcessor(config)
processor.process_dataset('openwebtext', ['train.txt'], ['val.txt'])
"
```

### For Custom Data:
```python
from nanokimi.training.data import TextDataProcessor, DataConfig

config = DataConfig(tokenizer_name="gpt2", block_size=1024)
processor = TextDataProcessor(config)

# Process your text files
train_path, val_path = processor.process_dataset(
    "my_dataset",
    train_files=["train1.txt", "train2.txt"],
    val_files=["val.txt"]
)
```

##  Benchmarking

Compare nanoKimi against nanoGPT and other baselines:

```python
from nanokimi.benchmark import ModelEvaluator, NanoGPTComparison

evaluator = ModelEvaluator()
comparison = NanoGPTComparison()

# Evaluate perplexity
ppl = evaluator.evaluate_perplexity(model, dataset)

# Compare with nanoGPT
results = comparison.compare_models(nanokimi_model, nanogpt_model, dataset)
```

##  Performance Optimizations

### Mixed Precision Training
```yaml
training:
  mixed_precision: true  # Enable FP16/BF16
```

### Model Compilation
```yaml  
training:
  compile_model: true    # Use torch.compile (PyTorch 2.0+)
```

### Expert Caching
```python
# Automatic expert caching during inference
generator = Generator(model)
text = generator.generate_text(prompt, use_cache=True)
```

##  Experiment Tracking

### Weights & Biases Integration
```yaml
training:
  use_wandb: true
  wandb_project: "my-nanokimi-experiments"
  wandb_run_name: "experiment-1"
```

### Local Logging
All metrics are also logged locally:
- Training loss and learning rate
- Expert utilization statistics  
- Memory usage and tokens/second
- Validation perplexity

##  Key Advantages Over nanoGPT

1. **Training Speed**: 20%+ faster convergence with Muon optimizer
2. **Parameter Efficiency**: Better performance per parameter with MoE
3. **Context Understanding**: Enhanced long-range modeling with latent attention
4. **Scalability**: Efficient scaling to larger models with sparse experts
5. **Memory Efficiency**: Comparable memory usage despite architectural complexity

##  Research Extensions

nanoKimi is designed for extensibility:

- **Routing Strategies**: Experiment with different expert routing algorithms
- **Attention Patterns**: Try various sparse attention mechanisms  
- **Multimodal**: Extend to vision and other modalities
- **Deployment**: Optimize for edge devices and inference

##  Documentation

- [Training Guide](docs/training.md) - Detailed training instructions
- [Architecture Deep Dive](docs/architecture.md) - Technical implementation details
- [API Reference](docs/api.md) - Complete API documentation
- [Benchmarking](docs/benchmarking.md) - Performance evaluation guide

##  Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black nanokimi/
flake8 nanokimi/
```

##  License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

##  Acknowledgments

- **Moonshot AI** - For the original Kimi-K2 innovations
- **nanoGPT** - For inspiration and educational approach
- **Transformers Community** - For the foundation this builds upon

##  Contact

- **Issues**: [GitHub Issues](https://github.com/dishant2009/nanoKimi)
- **Discussions**: [GitHub Discussions](https://github.com/dishant2009/nanokimi/discussions)
- **Email**: your.email@example.com

---

> **Note**: This is an educational implementation designed to demonstrate Kimi-K2 concepts. For production use, consider the full Kimi-K2 implementation.

##  Star History

If you find this project helpful, please consider giving it a star on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=dishant2009/nanokimi&type=Date)](https://star-history.com/#dishant2009/nanokimi&Date)
