# CitiBike NYC Inventory Prediction

[![CI](https://github.com/stasahani1/AM215_final_project/actions/workflows/ci.yml/badge.svg)](https://github.com/stasahani1/AM215_final_project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A machine learning project for predicting bike station inventory at NYC CitiBike stations using time-series forecasting and Markov chain models.

## Overview

This project implements multiple models to predict how many bikes will be available at each CitiBike station over time. This is useful for:
- **Riders**: Know if a station will have bikes when you arrive
- **Operations**: Predict which stations will be empty/full and need rebalancing

## Quick Start

```bash
# Clone and setup
git clone https://github.com/stasahani1/AM215_final_project.git
cd AM215_final_project
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run prediction pipeline
python run.py --model temporal_flow
```

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/stasahani1/AM215_final_project.git
cd AM215_final_project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .
```

### Development Installation

```bash
# Install with dev dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Reproducible Environment

For exact reproducibility of results:

```bash
pip install -r requirements-lock.txt
```

## Usage

### Run the Prediction Pipeline

```bash
# Run with default config (temporal_flow model)
python run.py

# Use a different model
python run.py --model markov
python run.py --model persistence
python run.py --model station_average

# Compare models
python run.py --model markov --compare baseline

# Set random seed for reproducibility
python run.py --model markov --seed 42
```

### Configuration

Edit `config.yaml` to customize:
- Data paths and time ranges
- Cross-validation parameters (folds, window sizes)
- Model-specific settings (smoothing, simulations)
- Empty/full station thresholds

## Models

| Model | Description |
|-------|-------------|
| **PersistenceModel** | Baseline: assumes inventory stays constant |
| **StationAverageModel** | Uses average net flow per station (ignores time) |
| **TemporalFlowModel** | Time-conditioned flow predictions (hour + weekend patterns) |
| **MarkovModel** | Markov chain with transition matrices and Monte Carlo simulation |

## Project Structure

```
AM215_final_project/
├── run.py                      # Main entry point
├── config.yaml                 # Configuration file
├── pyproject.toml              # Package configuration & dependencies
├── requirements-lock.txt       # Pinned dependencies for reproducibility
│
├── src/citibike/               # Main package (src layout)
│   ├── __init__.py
│   ├── models/                 # Prediction models
│   │   ├── base.py             # Abstract base class (ABC)
│   │   ├── naive.py            # Simple baselines
│   │   ├── baseline.py         # TemporalFlowModel
│   │   └── markov.py           # Markov chain model
│   ├── evaluation/             # Metrics and cross-validation
│   │   ├── metrics.py          # MAE, RMSE, state metrics
│   │   └── cross_validation.py # Rolling window CV
│   └── utils/                  # Data loading utilities
│       ├── data_loader.py      # Load trips/stations
│       └── helpers.py          # Config and project utilities
│
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_models.py          # Unit tests for models
│   ├── test_metrics.py         # Unit tests for metrics
│   ├── test_utils.py           # Unit tests for utilities
│   ├── test_regression.py      # Regression tests (golden files)
│   └── golden_files/           # Expected outputs for regression tests
│
├── scripts/                    # Analysis and data scripts
│   ├── analysis/               # Model analysis scripts
│   └── data/                   # Data processing scripts
│
├── data/                       # Raw and processed data
├── results/                    # Model outputs and figures
└── eda.ipynb                   # Exploratory data analysis notebook
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/citibike --cov-report=term-missing

# Run only unit tests (fast)
pytest -m "not regression"

# Run regression tests only
pytest -m regression
```

## Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking
mypy src/citibike/
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `inventory_mae` | Mean Absolute Error on bike counts |
| `inventory_rmse` | Root Mean Squared Error on bike counts |
| `empty_recall/precision/f1` | Classification metrics for empty stations |
| `full_recall/precision/f1` | Classification metrics for full stations |
| `state_accuracy` | Overall classification accuracy (empty/normal/full) |

## Dependencies

Core dependencies (see `pyproject.toml` for full list):
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `duckdb` - Efficient data querying
- `pyyaml` - Configuration parsing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## LLM Attribution

This project utilized Claude (Anthropic) Opus 4.5 for code generation assistance, including:
- Model Configuration & Setup
- Test suite implementation
- CI/CD pipeline configuration
- Code restructuring and refactoring
- Documentation generation

---

*AM215 Final Project - Harvard University*
