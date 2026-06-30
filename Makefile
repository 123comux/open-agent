.PHONY: install dev test lint typecheck serve clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,all]"

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src/open_agent

serve:
	uvicorn open_agent.server.api:app --reload --port 8000

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
