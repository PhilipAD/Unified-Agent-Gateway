.DEFAULT_GOAL := help

.PHONY: help install test test-cov lint format serve clean build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install in editable mode with dev dependencies
	pip install -e ".[dev]"

test: ## Run unit tests (no live API keys needed)
	pytest -q -m "not integration"

test-cov: ## Run tests with coverage report
	pytest -q --cov=. --cov-report=term-missing -m "not integration"

lint: ## Run ruff linter and format checker
	ruff check .
	ruff format --check .

format: ## Auto-format code with ruff
	ruff check --fix .
	ruff format .

serve: ## Start the gateway server with auto-reload
	uag serve --reload

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov

build: ## Build wheel and sdist
	pip install hatch
	hatch build
