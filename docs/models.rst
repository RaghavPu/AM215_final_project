Models
======

This project implements several models for predicting bike station inventory.

Model Hierarchy
---------------

All models inherit from ``BaseModel``:

.. code-block:: text

   BaseModel (ABC)
   ├── PersistenceModel
   ├── StationAverageModel
   ├── TemporalFlowModel
   └── MarkovModel

PersistenceModel
----------------

The simplest baseline model that assumes inventory stays constant.

**Use case**: Establishing a lower bound for model performance.

StationAverageModel
-------------------

Predicts inventory based on historical average net flow per station.

**Parameters**: None

**Use case**: Simple baseline that captures station-level trends.

TemporalFlowModel
-----------------

Time-conditioned flow predictions using hour-of-day and weekend patterns.

**Parameters**:

- Learns flow patterns conditioned on:
  - Hour of day (0-23)
  - Weekend vs. weekday

**Use case**: Captures temporal patterns in bike usage.

MarkovModel
-----------

Markov chain model with transition matrices and Monte Carlo simulation.

**Parameters**:

- ``smoothing_alpha``: Laplace smoothing parameter (default: 0.0)
- ``min_transitions``: Minimum transitions required (default: 10)
- ``n_simulations``: Number of Monte Carlo simulations (default: 1)
- ``random_seed``: Seed for reproducibility (default: None)

**Use case**: Best performing model, captures stochastic dynamics.

Model Selection
---------------

Use the factory function to create models:

.. code-block:: python

   from citibike.models import get_model
   
   config = load_config("config.yaml")
   model = get_model("markov", config)

