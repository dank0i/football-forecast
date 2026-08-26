.PHONY: help setup fetch build backtest ablate bet audit report test lint all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup:  ## Install dependencies
	uv sync

fetch:  ## Download and verify the database (313 MB)
	uv run pitchcast fetch

build:  ## Parse feeds, tune Elo, build the feature matrix
	uv run pitchcast build

backtest:  ## Walk-forward evaluation
	uv run pitchcast dixon-coles
	uv run pitchcast backtest

ablate:  ## Feature-block ablation
	uv run pitchcast ablate

bet:  ## Staking simulation
	uv run pitchcast bet

audit:  ## Reproduce and re-score the original notebook
	uv run pitchcast audit

report:  ## Regenerate figures
	uv run pitchcast report

test:  ## Run the test suite
	uv run pytest -q

lint:  ## Lint and format check
	uv run ruff check src tests
	uv run ruff format --check src tests

all: setup fetch build backtest ablate bet report  ## Full pipeline from scratch
