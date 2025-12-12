CitiBike Inventory Prediction
=============================

A machine learning project for predicting bike station inventory at NYC CitiBike stations 
using time-series forecasting and Markov chain models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   models
   api

Overview
--------

This project implements multiple models to predict how many bikes will be available 
at each CitiBike station over time:

- **PersistenceModel**: Baseline that assumes inventory stays constant
- **StationAverageModel**: Uses average net flow per station
- **TemporalFlowModel**: Time-conditioned flow predictions
- **MarkovModel**: Markov chain with transition matrices and Monte Carlo simulation

Features
--------

- Rolling window cross-validation for time-series evaluation
- Multiple evaluation metrics (MAE, RMSE, state classification)
- Configurable via YAML configuration files
- Reproducible random seeds for stochastic models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

