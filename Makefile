.PHONY: build format mypy test test-gpu

build:
	uv sync

format:
	uv run ruff check --fix src/ tests/ examples/
	uv run ruff format src/ tests/ examples/

mypy:
	uv run mypy src/

test:
	uv run pytest tests/ -v --ignore=tests/test_triton_kernel.py --ignore=tests/test_inference.py --ignore=tests/test_posterior.py --ignore=tests/test_integration.py --ignore=tests/test_gru_gradient_proof.py

test-gpu:
	uv run pytest tests/ -v
