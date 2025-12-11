# CitiBike NYC Inventory Prediction Model

[![CI](https://github.com/stasahani1/AM215_final_project/actions/workflows/ci.yml/badge.svg)](https://github.com/stasahani1/AM215_final_project/actions/workflows/ci.yml)

A machine learning project for predicting bike station inventory at NYC CitiBike stations using time-series forecasting and Markov chain models.

## Overview

This project implements multiple models to predict how many bikes will be available at each CitiBike station over time. This is useful for:
- **Riders**: Know if a station will have bikes when you arrive
- **Operations**: Predict which stations will be empty/full and need rebalancing

## Models

| Model | Description |
|-------|-------------|
| **PersistenceModel** | Baseline: assumes inventory stays constant |
| **StationAverageModel** | Uses average net flow per station (ignores time) |
| **TemporalFlowModel** | Time-conditioned flow (hour + weekend patterns) |
| **MarkovModel** | Markov chain with transition matrices and Monte Carlo simulation |

## Installation

```bash
# Clone the repository
git clone https://github.com/stasahani1/AM215_final_project.git
cd AM215_final_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (testing, linting)
pip install -r requirements-dev.txt
```

## Usage

### Run the main pipeline

```bash
# Run with default config (temporal_flow model)
python run.py

# Use a different model
python run.py --model markov
python run.py --model persistence

# Compare models
python run.py --model markov --compare baseline
```

### Configuration

Edit `config.yaml` to customize:
- Data paths
- Time range for analysis
- Cross-validation parameters
- Model settings
- Empty/full thresholds

## Project Structure

```
AM215_final_project/
├── run.py                    # Main entry point
├── config.yaml               # Configuration
├── models/                   # Prediction models
│   ├── base.py               # Abstract base class
│   ├── naive.py              # Simple baselines
│   ├── baseline.py           # TemporalFlowModel
│   └── markov.py             # Markov chain model
├── evaluation/               # Metrics and cross-validation
│   ├── metrics.py            # MAE, RMSE, state metrics
│   └── cross_validation.py   # Rolling window CV
├── utils/                    # Data loading utilities
│   ├── data_loader.py        # Load trips/stations
│   └── duckdb_utils.py       # DuckDB helpers
├── tests/                    # Test suite
│   ├── test_models.py        # Unit tests for models
│   ├── test_metrics.py       # Unit tests for metrics
│   ├── test_utils.py         # Unit tests for utilities
│   ├── test_regression.py    # Regression tests
│   └── golden_files/         # Expected outputs for regression tests
├── scripts/                  # Analysis scripts
├── data/                     # Raw and processed data
└── eda.ipynb                 # Exploratory data analysis
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov=evaluation --cov=utils

# Run only unit tests (fast)
pytest -m "not regression"

# Run regression tests
pytest -m regression
```

## Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy models/ evaluation/ utils/
```

## Evaluation Metrics

- **inventory_mae**: Mean Absolute Error on bike counts
- **inventory_rmse**: Root Mean Squared Error
- **empty_recall/precision/f1**: For empty station prediction
- **full_recall/precision/f1**: For full station prediction
- **state_accuracy**: Overall classification accuracy (empty/normal/full)

## License

MIT License
