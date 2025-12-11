"""Evaluation metrics for inventory prediction models."""

import numpy as np
import pandas as pd


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Compute Mean Absolute Percentage Error."""
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))


def inventory_to_states(
    inventory: pd.DataFrame,
    capacities: dict[str, float],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Convert inventory counts to states (empty/normal/full).

    Args:
        inventory: DataFrame with bike counts (index=stations, columns=times)
        capacities: Dict mapping station -> capacity
        thresholds: Dict with "empty" and "full" thresholds (as fraction of capacity)

    Returns:
        DataFrame with states ("empty", "normal", "full")
    """
    states = pd.DataFrame(index=inventory.index, columns=inventory.columns, dtype=str)

    for station in inventory.index:
        capacity = capacities.get(station, 30)
        empty_thresh = capacity * thresholds.get("empty", 0.1)
        full_thresh = capacity * thresholds.get("full", 0.9)

        for col in inventory.columns:
            bikes = inventory.loc[station, col]
            if bikes <= empty_thresh:
                states.loc[station, col] = "empty"
            elif bikes >= full_thresh:
                states.loc[station, col] = "full"
            else:
                states.loc[station, col] = "normal"

    return states


def compute_state_metrics(
    true_states: pd.DataFrame,
    pred_states: pd.DataFrame,
    state: str,
) -> dict[str, float]:
    """Compute precision, recall, F1 for a specific state.

    Args:
        true_states: DataFrame with actual states
        pred_states: DataFrame with predicted states
        state: Which state to evaluate ("empty" or "full")

    Returns:
        Dictionary with precision, recall, f1, count
    """
    # Flatten and align
    true_flat = true_states.values.flatten()
    pred_flat = pred_states.values.flatten()

    # Binary classification metrics
    true_positive = np.sum((true_flat == state) & (pred_flat == state))
    false_positive = np.sum((true_flat != state) & (pred_flat == state))
    false_negative = np.sum((true_flat == state) & (pred_flat != state))
    # true_negative not used but kept for documentation: np.sum((true_flat != state) & (pred_flat != state))

    # Compute metrics
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        f"{state}_precision": precision,
        f"{state}_recall": recall,
        f"{state}_f1": f1,
        f"{state}_count": int(np.sum(true_flat == state)),
        f"{state}_predicted_count": int(np.sum(pred_flat == state)),
    }


def compute_inventory_metrics(
    true_inventory: pd.DataFrame,
    pred_inventory: pd.DataFrame,
    capacities: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Compute all evaluation metrics for inventory prediction.

    Args:
        true_inventory: DataFrame with actual bike counts
        pred_inventory: DataFrame with predicted bike counts
        capacities: Dict mapping station -> capacity
        thresholds: Dict with "empty" and "full" thresholds

    Returns:
        Dictionary with all metrics
    """
    # Align dataframes
    common_stations = true_inventory.index.intersection(pred_inventory.index)
    common_times = true_inventory.columns.intersection(pred_inventory.columns)

    if len(common_stations) == 0 or len(common_times) == 0:
        return {"error": "No overlap between true and predicted data"}

    true_inv = true_inventory.loc[common_stations, common_times]
    pred_inv = pred_inventory.loc[common_stations, common_times]

    # Flatten for basic metrics
    true_flat = true_inv.values.flatten()
    pred_flat = pred_inv.values.flatten()

    metrics = {
        "inventory_mae": compute_mae(true_flat, pred_flat),
        "inventory_rmse": compute_rmse(true_flat, pred_flat),
        "inventory_mape": compute_mape(true_flat, pred_flat),
        "n_predictions": len(true_flat),
        "n_stations": len(common_stations),
        "n_time_periods": len(common_times),
    }

    # Correlation
    if len(true_flat) > 1 and np.std(true_flat) > 0 and np.std(pred_flat) > 0:
        correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics["correlation"] = 0.0

    # Convert to states
    true_states = inventory_to_states(true_inv, capacities, thresholds)
    pred_states = inventory_to_states(pred_inv, capacities, thresholds)

    # State-based metrics
    for state in ["empty", "full"]:
        state_metrics = compute_state_metrics(true_states, pred_states, state)
        metrics.update(state_metrics)

    # Overall state accuracy
    correct = np.sum(true_states.values == pred_states.values)
    total = true_states.size
    metrics["state_accuracy"] = correct / total if total > 0 else 0.0

    # Per-station metrics
    station_maes = []
    for station in common_stations:
        true_s = true_inv.loc[station].values
        pred_s = pred_inv.loc[station].values
        station_maes.append(compute_mae(true_s, pred_s))

    metrics["station_mae_mean"] = np.mean(station_maes)
    metrics["station_mae_std"] = np.std(station_maes)

    return metrics


def summarize_fold_results(fold_results: list) -> dict[str, tuple[float, float]]:
    """Summarize results across cross-validation folds.

    Args:
        fold_results: List of metric dictionaries from each fold

    Returns:
        Dictionary mapping metric -> (mean, std)
    """
    if not fold_results:
        return {}

    # Get all metric names
    metric_names = fold_results[0].keys()

    # Skip non-numeric metrics
    skip_metrics = {"fold_id", "train_start", "test_start", "train_end", "test_end", "error"}

    summary = {}
    for metric in metric_names:
        if metric in skip_metrics:
            continue

        values = [r[metric] for r in fold_results if metric in r]
        if values:
            try:
                numeric_values = [float(v) for v in values]
                summary[metric] = (np.mean(numeric_values), np.std(numeric_values))
            except (ValueError, TypeError):
                continue

    return summary
