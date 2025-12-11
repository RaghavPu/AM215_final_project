"""Regression tests with golden files.

These tests compare model outputs against known "correct" outputs to ensure
that changes to the codebase don't silently break the model's behavior.

To regenerate golden files after intentional changes:
    pytest tests/test_regression.py --regenerate-golden

Or manually:
    python tests/generate_golden_files.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import compute_inventory_metrics, inventory_to_states
from models import MarkovModel, TemporalFlowModel

# Path to golden files
GOLDEN_DIR = Path(__file__).parent / "golden_files"


def load_golden_json(filename: str) -> dict:
    """Load a golden file as JSON."""
    filepath = GOLDEN_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Golden file not found: {filepath}. Run generate_golden_files.py first.")
    with open(filepath) as f:
        return json.load(f)


def load_golden_npy(filename: str) -> np.ndarray:
    """Load a golden file as numpy array."""
    filepath = GOLDEN_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Golden file not found: {filepath}. Run generate_golden_files.py first.")
    return np.load(filepath)


def create_deterministic_trips() -> pd.DataFrame:
    """Create deterministic trip data for regression tests."""
    np.random.seed(12345)
    n_trips = 500

    stations = ["Station A", "Station B", "Station C"]

    # Deterministic "random" trips
    base_date = pd.Timestamp("2025-09-01")
    hours = np.random.randint(0, 24 * 14, n_trips)

    started_at = [base_date + pd.Timedelta(hours=int(h)) for h in hours]
    ended_at = [t + pd.Timedelta(minutes=30) for t in started_at]

    trips = pd.DataFrame(
        {
            "started_at": started_at,
            "ended_at": ended_at,
            "start_station_name": np.random.choice(stations, n_trips),
            "end_station_name": np.random.choice(stations, n_trips),
        }
    )

    trips["hour"] = trips["started_at"].dt.hour
    trips["is_weekend"] = trips["started_at"].dt.dayofweek.isin([5, 6])

    return trips.sort_values("started_at").reset_index(drop=True)


def create_deterministic_station_stats() -> pd.DataFrame:
    """Create deterministic station stats for regression tests."""
    return pd.DataFrame(
        {
            "capacity": [30, 25, 40],
        },
        index=["Station A", "Station B", "Station C"],
    )


def create_deterministic_config() -> dict:
    """Create deterministic config for regression tests."""
    return {
        "thresholds": {"empty": 0.1, "full": 0.9},
        "model": {
            "markov": {
                "smoothing_alpha": 0.0,
                "min_transitions": 1,
                "n_simulations": 1,  # Deterministic
            }
        },
    }


@pytest.mark.regression
class TestTemporalFlowRegression:
    """Regression tests for TemporalFlowModel."""

    def test_hourly_flow_values(self):
        """Test that learned hourly flow values match golden file."""
        trips = create_deterministic_trips()
        station_stats = create_deterministic_station_stats()
        config = create_deterministic_config()

        model = TemporalFlowModel(config)
        model.fit(trips, station_stats)

        # Load expected values
        expected = load_golden_json("temporal_flow_hourly_flows.json")

        # Compare a subset of key values
        for key_str, expected_value in expected.items():
            # Parse key: "station,hour,is_weekend"
            parts = key_str.split(",")
            station = parts[0]
            hour = int(parts[1])
            is_weekend = parts[2] == "True"

            key = (station, hour, is_weekend)
            if key in model.hourly_net_flow:
                actual_value = model.hourly_net_flow[key]
                assert np.isclose(actual_value, expected_value, atol=1e-6), (
                    f"Mismatch for {key}: expected {expected_value}, got {actual_value}"
                )

    def test_prediction_output(self):
        """Test that predictions match golden file."""
        trips = create_deterministic_trips()
        station_stats = create_deterministic_station_stats()
        config = create_deterministic_config()

        model = TemporalFlowModel(config)
        model.fit(trips, station_stats)

        initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
        start = pd.Timestamp("2025-09-15 08:00:00")
        end = pd.Timestamp("2025-09-15 14:00:00")

        predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")

        # Load expected
        expected = load_golden_npy("temporal_flow_predictions.npy")

        np.testing.assert_allclose(
            predictions.values,
            expected,
            atol=1e-6,
            err_msg="TemporalFlowModel predictions don't match golden file",
        )


@pytest.mark.regression
class TestMarkovModelRegression:
    """Regression tests for MarkovModel."""

    def test_transition_matrix_structure(self):
        """Test that transition matrix structure matches golden file."""
        trips = create_deterministic_trips()
        station_stats = create_deterministic_station_stats()
        config = create_deterministic_config()

        model = MarkovModel(config)
        model.fit(trips, station_stats)

        # Load expected
        expected = load_golden_json("markov_transition_stats.json")

        # Check basic structure
        assert len(model.stations) == expected["n_stations"]
        assert len(model.transition_matrices) == expected["n_matrices"]

        # Check global matrix properties
        P = model.global_transition_matrix.toarray()
        assert P.shape == tuple(expected["global_matrix_shape"])

        # Check sparsity is similar
        actual_nnz = model.global_transition_matrix.nnz
        assert actual_nnz == expected["global_matrix_nnz"]

    def test_prediction_deterministic(self):
        """Test that deterministic predictions match golden file."""
        trips = create_deterministic_trips()
        station_stats = create_deterministic_station_stats()
        config = create_deterministic_config()
        config["model"]["markov"]["n_simulations"] = 1

        model = MarkovModel(config)
        model.fit(trips, station_stats)

        initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
        start = pd.Timestamp("2025-09-15 08:00:00")
        end = pd.Timestamp("2025-09-15 14:00:00")

        predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")

        # Load expected
        expected = load_golden_npy("markov_predictions.npy")

        np.testing.assert_allclose(
            predictions.values,
            expected,
            atol=1e-6,
            err_msg="MarkovModel predictions don't match golden file",
        )


@pytest.mark.regression
class TestMetricsRegression:
    """Regression tests for evaluation metrics."""

    def test_inventory_metrics_values(self):
        """Test that computed metrics match golden values."""
        # Create deterministic test data
        np.random.seed(42)
        stations = ["Station A", "Station B", "Station C"]
        times = pd.date_range("2025-09-01", periods=24, freq="1h")

        true_inv = pd.DataFrame(
            np.random.randint(5, 25, size=(3, 24)),
            index=stations,
            columns=times,
        )

        # Predictions with some error
        np.random.seed(43)
        pred_inv = true_inv + np.random.randint(-3, 4, size=(3, 24))
        pred_inv = pred_inv.clip(lower=0, upper=30)

        capacities = {"Station A": 30, "Station B": 25, "Station C": 40}
        thresholds = {"empty": 0.1, "full": 0.9}

        metrics = compute_inventory_metrics(true_inv, pred_inv, capacities, thresholds)

        # Load expected
        expected = load_golden_json("metrics_values.json")

        # Compare key metrics
        for key in ["inventory_mae", "inventory_rmse", "correlation", "state_accuracy"]:
            assert np.isclose(metrics[key], expected[key], atol=1e-6), (
                f"Metric {key}: expected {expected[key]}, got {metrics[key]}"
            )

    def test_state_conversion_consistency(self):
        """Test that state conversion is consistent with golden file."""
        stations = ["Station A", "Station B"]
        times = pd.date_range("2025-09-01", periods=5, freq="1h")

        # Specific inventory values to test edge cases
        inventory = pd.DataFrame(
            [[2, 5, 15, 27, 29], [1, 3, 12, 22, 24]],
            index=stations,
            columns=times,
        )

        capacities = {"Station A": 30, "Station B": 25}
        thresholds = {"empty": 0.1, "full": 0.9}

        states = inventory_to_states(inventory, capacities, thresholds)

        # Load expected
        expected = load_golden_json("state_conversion.json")

        # Compare as nested dict
        actual = states.to_dict()
        for col in states.columns:
            col_str = str(col)
            for station in states.index:
                assert actual[col][station] == expected[col_str][station], (
                    f"State mismatch at {station}, {col_str}"
                )


@pytest.mark.regression
class TestEndToEndRegression:
    """End-to-end regression test."""

    def test_full_pipeline_metrics(self):
        """Test that full pipeline produces consistent metrics."""
        from evaluation import track_inventory

        trips = create_deterministic_trips()
        station_stats = create_deterministic_station_stats()
        config = create_deterministic_config()

        # Fit model
        model = TemporalFlowModel(config)
        model.fit(trips, station_stats)

        # Set up test period
        initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
        start = pd.Timestamp("2025-09-10")
        end = pd.Timestamp("2025-09-11")

        # Get predictions
        predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")

        # Track actual inventory
        actual = track_inventory(trips, initial_inventory, start, end, freq="1h")

        # Compute metrics
        capacities = station_stats["capacity"].to_dict()
        thresholds = config["thresholds"]

        metrics = compute_inventory_metrics(actual, predictions, capacities, thresholds)

        # Load expected
        expected = load_golden_json("pipeline_metrics.json")

        # Check key metrics are consistent
        assert np.isclose(metrics["inventory_mae"], expected["inventory_mae"], atol=0.1), (
            f"MAE changed: expected {expected['inventory_mae']}, got {metrics['inventory_mae']}"
        )
        assert np.isclose(metrics["inventory_rmse"], expected["inventory_rmse"], atol=0.1), (
            f"RMSE changed: expected {expected['inventory_rmse']}, got {metrics['inventory_rmse']}"
        )
