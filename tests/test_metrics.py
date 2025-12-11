"""Unit tests for evaluation metrics."""

import numpy as np
import pandas as pd

from evaluation import (
    compute_inventory_metrics,
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_state_metrics,
    inventory_to_states,
)


class TestBasicMetrics:
    """Tests for basic error metrics."""

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert compute_mae(y_true, y_pred) == 0.0

    def test_mae_positive(self):
        """MAE should always be non-negative."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])

        mae = compute_mae(y_true, y_pred)
        assert mae >= 0
        assert mae == 1.0  # All predictions off by 1

    def test_mae_symmetric(self):
        """MAE should be symmetric (over/under prediction)."""
        y_true = np.array([10, 10, 10])
        y_pred_over = np.array([12, 12, 12])
        y_pred_under = np.array([8, 8, 8])

        assert compute_mae(y_true, y_pred_over) == compute_mae(y_true, y_pred_under)

    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert compute_rmse(y_true, y_pred) == 0.0

    def test_rmse_positive(self):
        """RMSE should always be non-negative."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 1, 4, 3, 6])

        rmse = compute_rmse(y_true, y_pred)
        assert rmse >= 0

    def test_rmse_penalizes_large_errors(self):
        """RMSE should penalize large errors more than MAE."""
        y_true = np.array([10, 10, 10, 10])
        y_pred_small = np.array([11, 11, 11, 11])  # All off by 1
        y_pred_large = np.array([10, 10, 10, 14])  # One off by 4

        # Same MAE
        assert compute_mae(y_true, y_pred_small) == compute_mae(y_true, y_pred_large)

        # But different RMSE (large error penalized more)
        assert compute_rmse(y_true, y_pred_large) > compute_rmse(y_true, y_pred_small)

    def test_mape_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert compute_mape(y_true, y_pred) == 0.0

    def test_mape_handles_zeros(self):
        """MAPE should handle zero values gracefully (with epsilon)."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 1, 2])

        # Should not raise division by zero
        mape = compute_mape(y_true, y_pred)
        assert np.isfinite(mape)


class TestInventoryToStates:
    """Tests for inventory to state conversion."""

    def test_empty_state_below_threshold(self):
        """Inventory below empty threshold should be 'empty'."""
        inventory = pd.DataFrame(
            [[2, 3]],
            index=["Station A"],
            columns=["t1", "t2"],
        )
        capacities = {"Station A": 30}
        thresholds = {"empty": 0.1, "full": 0.9}  # empty < 3 bikes

        states = inventory_to_states(inventory, capacities, thresholds)

        assert states.loc["Station A", "t1"] == "empty"
        assert states.loc["Station A", "t2"] == "empty"

    def test_full_state_above_threshold(self):
        """Inventory above full threshold should be 'full'."""
        inventory = pd.DataFrame(
            [[28, 29]],
            index=["Station A"],
            columns=["t1", "t2"],
        )
        capacities = {"Station A": 30}
        thresholds = {"empty": 0.1, "full": 0.9}  # full > 27 bikes

        states = inventory_to_states(inventory, capacities, thresholds)

        assert states.loc["Station A", "t1"] == "full"
        assert states.loc["Station A", "t2"] == "full"

    def test_normal_state_between_thresholds(self):
        """Inventory between thresholds should be 'normal'."""
        inventory = pd.DataFrame(
            [[15, 20]],
            index=["Station A"],
            columns=["t1", "t2"],
        )
        capacities = {"Station A": 30}
        thresholds = {"empty": 0.1, "full": 0.9}

        states = inventory_to_states(inventory, capacities, thresholds)

        assert states.loc["Station A", "t1"] == "normal"
        assert states.loc["Station A", "t2"] == "normal"

    def test_multiple_stations(self):
        """Should handle multiple stations correctly."""
        inventory = pd.DataFrame(
            [[1, 15, 29]],
            index=["Station A"],
            columns=["empty_time", "normal_time", "full_time"],
        )
        capacities = {"Station A": 30}
        thresholds = {"empty": 0.1, "full": 0.9}

        states = inventory_to_states(inventory, capacities, thresholds)

        assert states.loc["Station A", "empty_time"] == "empty"
        assert states.loc["Station A", "normal_time"] == "normal"
        assert states.loc["Station A", "full_time"] == "full"


class TestStateMetrics:
    """Tests for state-based metrics (precision, recall, F1)."""

    def test_perfect_recall(self):
        """Perfect recall when all true positives are found."""
        true_states = pd.DataFrame([["empty", "empty", "normal"]])
        pred_states = pd.DataFrame([["empty", "empty", "empty"]])

        metrics = compute_state_metrics(true_states, pred_states, "empty")

        assert metrics["empty_recall"] == 1.0  # Found all empty states

    def test_perfect_precision(self):
        """Perfect precision when all predictions are correct."""
        true_states = pd.DataFrame([["empty", "empty", "normal"]])
        pred_states = pd.DataFrame([["empty", "normal", "normal"]])

        metrics = compute_state_metrics(true_states, pred_states, "empty")

        assert metrics["empty_precision"] == 1.0  # All predicted empty are correct

    def test_zero_recall_when_missing_all(self):
        """Zero recall when no true positives are found."""
        true_states = pd.DataFrame([["empty", "empty", "empty"]])
        pred_states = pd.DataFrame([["normal", "normal", "normal"]])

        metrics = compute_state_metrics(true_states, pred_states, "empty")

        assert metrics["empty_recall"] == 0.0

    def test_f1_is_harmonic_mean(self):
        """F1 should be harmonic mean of precision and recall."""
        true_states = pd.DataFrame([["empty", "empty", "normal", "normal"]])
        pred_states = pd.DataFrame([["empty", "normal", "empty", "normal"]])

        metrics = compute_state_metrics(true_states, pred_states, "empty")

        precision = metrics["empty_precision"]
        recall = metrics["empty_recall"]
        expected_f1 = 2 * precision * recall / (precision + recall)

        assert np.isclose(metrics["empty_f1"], expected_f1)

    def test_count_tracking(self):
        """Should correctly count true and predicted instances."""
        true_states = pd.DataFrame([["empty", "empty", "normal"]])
        pred_states = pd.DataFrame([["empty", "normal", "empty"]])

        metrics = compute_state_metrics(true_states, pred_states, "empty")

        assert metrics["empty_count"] == 2  # 2 actual empty
        assert metrics["empty_predicted_count"] == 2  # 2 predicted empty


class TestInventoryMetrics:
    """Tests for the combined inventory metrics function."""

    def test_computes_all_metrics(self, sample_inventory_df):
        """Should compute all expected metrics."""
        true_inv = sample_inventory_df
        pred_inv = sample_inventory_df + np.random.normal(0, 2, sample_inventory_df.shape)

        capacities = dict.fromkeys(sample_inventory_df.index, 30)
        thresholds = {"empty": 0.1, "full": 0.9}

        metrics = compute_inventory_metrics(true_inv, pred_inv, capacities, thresholds)

        # Check all expected keys exist
        expected_keys = [
            "inventory_mae",
            "inventory_rmse",
            "inventory_mape",
            "correlation",
            "state_accuracy",
            "empty_recall",
            "empty_precision",
            "full_recall",
            "full_precision",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_handles_misaligned_dataframes(self):
        """Should handle DataFrames with different indices."""
        times = pd.date_range("2025-09-01", periods=5, freq="1h")

        true_inv = pd.DataFrame(
            [[10, 11, 12, 13, 14]],
            index=["Station A"],
            columns=times,
        )
        pred_inv = pd.DataFrame(
            [[10, 11, 12, 13, 14]],
            index=["Station A"],
            columns=times,
        )

        capacities = {"Station A": 30}
        thresholds = {"empty": 0.1, "full": 0.9}

        metrics = compute_inventory_metrics(true_inv, pred_inv, capacities, thresholds)

        assert "inventory_mae" in metrics
        assert metrics["inventory_mae"] == 0.0  # Perfect prediction

    def test_correlation_range(self, sample_inventory_df):
        """Correlation should be in [-1, 1]."""
        true_inv = sample_inventory_df
        pred_inv = sample_inventory_df * 0.9 + 1  # Linearly related

        capacities = dict.fromkeys(sample_inventory_df.index, 30)
        thresholds = {"empty": 0.1, "full": 0.9}

        metrics = compute_inventory_metrics(true_inv, pred_inv, capacities, thresholds)

        assert -1 <= metrics["correlation"] <= 1
