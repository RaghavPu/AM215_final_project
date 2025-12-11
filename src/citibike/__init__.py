"""CitiBike NYC Inventory Prediction Package.

This package provides models and utilities for predicting bike station
inventory at NYC CitiBike stations.

Modules:
    models: Prediction models (Markov, TemporalFlow, baselines)
    evaluation: Cross-validation and metrics
    utils: Data loading and helper functions
"""

from citibike.evaluation import (
    RollingWindowCV,
    compute_inventory_metrics,
    compute_mae,
    compute_rmse,
    run_cross_validation,
)
from citibike.models import (
    BaseModel,
    MarkovModel,
    PersistenceModel,
    StationAverageModel,
    TemporalFlowModel,
    get_model,
)
from citibike.utils import (
    load_config,
    load_station_info,
    load_trip_data,
    prepare_data,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "BaseModel",
    "MarkovModel",
    "PersistenceModel",
    "StationAverageModel",
    "TemporalFlowModel",
    "get_model",
    # Evaluation
    "RollingWindowCV",
    "compute_inventory_metrics",
    "compute_mae",
    "compute_rmse",
    "run_cross_validation",
    # Utils
    "load_config",
    "load_station_info",
    "load_trip_data",
    "prepare_data",
]
