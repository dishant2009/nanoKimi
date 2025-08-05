.PHONY: help install train-nano train-small generate test clean lint format setup-data

help:
	@echo "nanoKimi - Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package and dependencies"
	@echo "  install-flash Install Flash Attention (requires CUDA)"
	@echo "  install-xformers Install xFormers for efficient attention"
	@echo "  install-accelerate Install Accelerate for distributed training"
	@echo "  setup-data   Create toy dataset for testing"
	@echo ""
	@echo "Training:"
	@echo "  train-nano   Train nano model (quick test)"
	@echo "  train-small  Train small model"
	@echo "  train-distributed Train with distributed (multi-GPU)"
	@echo "  example      Run basic usage example"
	@echo "  flash-demo   Run Flash Attention demonstration"
	@echo ""
	@echo "Inference:"
	@echo "  generate     Generate text with nano model"
	@echo "  interactive  Interactive text generation"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-fast    Run fast tests (exclude slow tests)"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -e .

setup-data:
	mkdir -p data checkpoints
	python -c "from nanokimi.training.data import load_toy_dataset; load_toy_dataset('./data')"

train-nano:
	python scripts/train.py --config configs/nano.yaml --no-wandb

train-small:
	python scripts/train.py --config configs/small.yaml

train-distributed:
	python scripts/train_distributed.py --config configs/small.yaml --nprocs 2

install-accelerate:
	pip install accelerate

example:
	python examples/basic_usage.py

flash-demo:
	python examples/flash_attention_demo.py

install-flash:
	pip install flash-attn --no-build-isolation

install-xformers:
	pip install xformers

generate:
	@if [ ! -f "./checkpoints/nano/best_model.pt" ]; then \
		echo "No trained model found. Run 'make train-nano' first."; \
		exit 1; \
	fi
	python scripts/generate.py --checkpoint ./checkpoints/nano/best_model.pt --prompt "The future of AI is"

interactive:
	@if [ ! -f "./checkpoints/nano/best_model.pt" ]; then \
		echo "No trained model found. Run 'make train-nano' first."; \
		exit 1; \
	fi
	python scripts/generate.py --checkpoint ./checkpoints/nano/best_model.pt --interactive

test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m "unit or not integration"

test-integration:
	pytest tests/ -v -m integration

test-fast:
	pytest tests/ -v -m "not slow"

test-coverage:
	pytest tests/ --cov=nanokimi --cov-report=html --cov-report=term

lint:
	flake8 nanokimi/
	flake8 scripts/
	flake8 examples/

format:
	black nanokimi/
	black scripts/
	black examples/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

# Development shortcuts
dev-setup: install setup-data
	@echo "Development environment ready!"

quick-test: train-nano generate
	@echo "Quick test completed!"

# Model size variants
train-medium:
	python scripts/train.py --config configs/medium.yaml

benchmark:
	@if [ ! -f "./checkpoints/nano/best_model.pt" ]; then \
		echo "No trained model found. Run 'make train-nano' first."; \
		exit 1; \
	fi
	python scripts/benchmark.py --checkpoint ./checkpoints/nano/best_model.pt --dataset toy --batch-size 2 --max-batches 5 --no-generation

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Docker (placeholder)
docker-build:
	@echo "Docker build not yet implemented"

docker-run:
	@echo "Docker run not yet implemented"