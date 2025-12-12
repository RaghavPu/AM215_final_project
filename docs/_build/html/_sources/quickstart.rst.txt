Quick Start
===========

Basic Usage
-----------

Run the prediction pipeline with the default model:

.. code-block:: bash

   python run.py

Selecting a Model
-----------------

Choose a specific model:

.. code-block:: bash

   # Markov chain model
   python run.py --model markov
   
   # Temporal flow model
   python run.py --model temporal_flow
   
   # Persistence baseline
   python run.py --model persistence
   
   # Station average model
   python run.py --model station_average

Setting Random Seed
-------------------

For reproducible results:

.. code-block:: bash

   python run.py --model markov --seed 42

Configuration
-------------

Edit ``config.yaml`` to customize:

- Data paths and time ranges
- Cross-validation parameters
- Model-specific settings
- Empty/full station thresholds

Example configuration:

.. code-block:: yaml

   data:
     trips_path: "data/parquet/trips"
     station_info_path: "data/parquet/stations/station_info.parquet"
   
   model:
     name: "markov"
     markov:
       smoothing_alpha: 0.0
       min_transitions: 10
       n_simulations: 1
       random_seed: 42
   
   cross_validation:
     n_folds: 4
     train_window_days: 7
     test_window_hours: 24

