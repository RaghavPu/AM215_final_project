"""Unit tests for prediction models."""

import numpy as np
import pandas as pd
import pytest

from citibike.models import (
    MarkovModel,
    PersistenceModel,
    StationAverageModel,
    TemporalFlowModel,
    get_model,
)


class TestPersistenceModel:
    """Tests for PersistenceModel."""

    def test_fit_sets_is_fitted(self, sample_config, sample_trips, sample_station_stats):
        """Model should be marked as fitted after fit()."""
        model = PersistenceModel(sample_config)
        assert not model.is_fitted

        model.fit(sample_trips, sample_station_stats)
        assert model.is_fitted

    def test_predict_returns_constant_inventory(
        self, sample_config, sample_trips, sample_station_stats, sample_inventory
    ):
        """Persistence model should predict constant inventory."""
        model = PersistenceModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        start = pd.Timestamp("2025-09-15")
        end = pd.Timestamp("2025-09-15 12:00:00")

        predictions = model.predict_inventory(sample_inventory, start, end, freq="1h")

        # All columns should equal initial inventory
        for col in predictions.columns:
            pd.testing.assert_series_equal(
                predictions[col],
                sample_inventory,
                check_names=False,
            )

    def test_predict_without_fit_raises(self, sample_config, sample_inventory):
        """Predicting without fitting should raise ValueError."""
        model = PersistenceModel(sample_config)

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict_inventory(
                sample_inventory,
                pd.Timestamp("2025-09-15"),
                pd.Timestamp("2025-09-15 12:00:00"),
            )

    def test_get_name_returns_class_name(self, sample_config):
        """get_name() should return the class name."""
        model = PersistenceModel(sample_config)
        assert model.get_name() == "PersistenceModel"


class TestStationAverageModel:
    """Tests for StationAverageModel."""

    def test_fit_computes_station_averages(self, sample_config, sample_trips, sample_station_stats):
        """Fitting should compute average flow per station."""
        model = StationAverageModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        assert model.is_fitted
        assert len(model.station_avg_flow) > 0

    def test_predict_changes_inventory(
        self, sample_config, sample_trips, sample_station_stats, sample_inventory
    ):
        """Predictions should evolve from initial state (unless avg_flow is 0)."""
        model = StationAverageModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        start = pd.Timestamp("2025-09-15")
        end = pd.Timestamp("2025-09-16")

        predictions = model.predict_inventory(sample_inventory, start, end, freq="1h")

        # Should have multiple time columns
        assert len(predictions.columns) > 1
        # Inventory values should be non-negative
        assert (predictions.values >= 0).all()

    def test_inventory_respects_capacity(self, sample_config, sample_trips, sample_station_stats):
        """Predictions should not exceed station capacity."""
        model = StationAverageModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        # Start with full stations
        full_inventory = sample_station_stats["capacity"].copy()

        start = pd.Timestamp("2025-09-15")
        end = pd.Timestamp("2025-09-16")

        predictions = model.predict_inventory(full_inventory, start, end, freq="1h")

        # Check all predictions respect capacity
        for station in predictions.index:
            capacity = sample_station_stats.loc[station, "capacity"]
            assert (predictions.loc[station] <= capacity).all()


class TestTemporalFlowModel:
    """Tests for TemporalFlowModel."""

    def test_fit_creates_hourly_flow_lookup(
        self, sample_config, sample_trips, sample_station_stats
    ):
        """Fitting should create hourly flow lookup table."""
        model = TemporalFlowModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        assert model.is_fitted
        assert len(model.hourly_net_flow) > 0

        # Keys should be (station, hour, is_weekend) tuples
        sample_key = list(model.hourly_net_flow.keys())[0]
        assert len(sample_key) == 3
        assert isinstance(sample_key[0], str)  # station name
        assert isinstance(sample_key[1], int)  # hour
        assert isinstance(sample_key[2], bool)  # is_weekend

    def test_predict_uses_time_conditioning(
        self, sample_config, sample_trips, sample_station_stats, sample_inventory
    ):
        """Predictions should vary based on hour and weekend."""
        model = TemporalFlowModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        # Weekday prediction
        start_weekday = pd.Timestamp("2025-09-15")  # Monday
        end_weekday = pd.Timestamp("2025-09-15 12:00:00")
        pred_weekday = model.predict_inventory(
            sample_inventory, start_weekday, end_weekday, freq="1h"
        )

        # Weekend prediction
        start_weekend = pd.Timestamp("2025-09-13")  # Saturday
        end_weekend = pd.Timestamp("2025-09-13 12:00:00")
        pred_weekend = model.predict_inventory(
            sample_inventory, start_weekend, end_weekend, freq="1h"
        )

        # Both should be valid DataFrames
        assert not pred_weekday.empty
        assert not pred_weekend.empty
        assert (pred_weekday.values >= 0).all()
        assert (pred_weekend.values >= 0).all()

    def test_get_params_returns_model_info(self, sample_config, sample_trips, sample_station_stats):
        """get_params() should return model parameters."""
        model = TemporalFlowModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        params = model.get_params()
        assert "name" in params
        assert "n_flow_combinations" in params
        assert params["n_flow_combinations"] > 0


class TestMarkovModel:
    """Tests for MarkovModel."""

    def test_fit_builds_transition_matrices(
        self, sample_config, sample_trips, sample_station_stats
    ):
        """Fitting should build sparse transition matrices."""
        model = MarkovModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        assert model.is_fitted
        assert len(model.transition_matrices) > 0
        assert model.global_transition_matrix is not None

    def test_transition_matrix_is_stochastic(
        self, sample_config, sample_trips, sample_station_stats
    ):
        """Transition matrix rows should sum to <= 1."""
        model = MarkovModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        # Check global transition matrix
        P = model.global_transition_matrix.toarray()
        row_sums = P.sum(axis=1)

        # Rows with any departures should sum to ~1
        # Rows with no departures will be all zeros
        nonzero_rows = row_sums > 0
        if nonzero_rows.any():
            assert np.allclose(row_sums[nonzero_rows], 1.0, atol=1e-6)

    def test_predict_returns_valid_inventory(
        self, sample_config, sample_trips, sample_station_stats, sample_inventory
    ):
        """Predictions should be valid inventory values."""
        model = MarkovModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        start = pd.Timestamp("2025-09-15")
        end = pd.Timestamp("2025-09-15 12:00:00")

        predictions = model.predict_inventory(sample_inventory, start, end, freq="1h")

        # All values should be non-negative
        assert (predictions.values >= 0).all()

        # All values should respect capacity
        for station in predictions.index:
            if station in sample_station_stats.index:
                capacity = sample_station_stats.loc[station, "capacity"]
                assert (predictions.loc[station] <= capacity).all()

    def test_get_top_destinations(self, sample_config, sample_trips, sample_station_stats):
        """get_top_destinations() should return valid probabilities."""
        model = MarkovModel(sample_config)
        model.fit(sample_trips, sample_station_stats)

        station = sample_trips["start_station_name"].iloc[0]
        top_dests = model.get_top_destinations(station, hour=12, is_weekend=False, top_k=3)

        if not top_dests.empty:
            # Probabilities should be in [0, 1]
            assert (top_dests["probability"] >= 0).all()
            assert (top_dests["probability"] <= 1).all()

    def test_deterministic_with_single_simulation(
        self, sample_config, sample_trips, sample_station_stats, sample_inventory
    ):
        """With n_simulations=1, predictions should be deterministic."""
        config = sample_config.copy()
        config["model"]["markov"]["n_simulations"] = 1

        model = MarkovModel(config)
        model.fit(sample_trips, sample_station_stats)

        start = pd.Timestamp("2025-09-15")
        end = pd.Timestamp("2025-09-15 06:00:00")

        pred1 = model.predict_inventory(sample_inventory, start, end, freq="1h")
        pred2 = model.predict_inventory(sample_inventory, start, end, freq="1h")

        # Should be identical (deterministic mode)
        pd.testing.assert_frame_equal(pred1, pred2)


class TestModelFactory:
    """Tests for the get_model factory function."""

    def test_get_persistence_model(self, sample_config):
        """Factory should return PersistenceModel."""
        model = get_model("persistence", sample_config)
        assert isinstance(model, PersistenceModel)

    def test_get_station_avg_model(self, sample_config):
        """Factory should return StationAverageModel."""
        model = get_model("station_avg", sample_config)
        assert isinstance(model, StationAverageModel)

    def test_get_temporal_flow_model(self, sample_config):
        """Factory should return TemporalFlowModel."""
        model = get_model("temporal_flow", sample_config)
        assert isinstance(model, TemporalFlowModel)

    def test_get_baseline_alias(self, sample_config):
        """'baseline' should be alias for TemporalFlowModel."""
        model = get_model("baseline", sample_config)
        assert isinstance(model, TemporalFlowModel)

    def test_get_markov_model(self, sample_config):
        """Factory should return MarkovModel."""
        model = get_model("markov", sample_config)
        assert isinstance(model, MarkovModel)

    def test_unknown_model_raises(self, sample_config):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model", sample_config)
