"""Evaluation metrics for flow prediction models."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Compute Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (as fraction, not percentage)
    """
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))


def compute_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy of predicting flow direction (positive/negative).
    
    This measures: Did we correctly predict if a station gains or loses bikes?
    
    Args:
        y_true: True net flow values
        y_pred: Predicted net flow values
        
    Returns:
        Fraction of correct direction predictions
    """
    # Get signs (positive = net inflow, negative = net outflow)
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    
    # Count matches (including both predicting 0)
    correct = np.sum(true_sign == pred_sign)
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0


def compute_flow_metrics(
    true_flow: pd.DataFrame,
    pred_flow: pd.DataFrame,
) -> Dict[str, float]:
    """Compute all flow prediction metrics.
    
    Args:
        true_flow: DataFrame with actual net flow values
                   (index=stations, columns=times)
        pred_flow: DataFrame with predicted net flow values
        
    Returns:
        Dictionary with all metrics
    """
    # Align dataframes
    common_stations = true_flow.index.intersection(pred_flow.index)
    common_times = true_flow.columns.intersection(pred_flow.columns)
    
    if len(common_stations) == 0 or len(common_times) == 0:
        return {"error": "No overlap between true and predicted data"}
    
    true_aligned = true_flow.loc[common_stations, common_times]
    pred_aligned = pred_flow.loc[common_stations, common_times]
    
    # Flatten for metric computation
    true_flat = true_aligned.values.flatten()
    pred_flat = pred_aligned.values.flatten()
    
    metrics = {
        "mae": compute_mae(true_flat, pred_flat),
        "rmse": compute_rmse(true_flat, pred_flat),
        "mape": compute_mape(true_flat, pred_flat),
        "direction_accuracy": compute_direction_accuracy(true_flat, pred_flat),
        "n_predictions": len(true_flat),
        "n_stations": len(common_stations),
        "n_time_periods": len(common_times),
    }
    
    # Correlation between predicted and actual
    if len(true_flat) > 1:
        correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
        metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics["correlation"] = 0.0
    
    # Per-station metrics (average across stations)
    station_maes = []
    station_rmses = []
    for station in common_stations:
        true_s = true_aligned.loc[station].values
        pred_s = pred_aligned.loc[station].values
        station_maes.append(compute_mae(true_s, pred_s))
        station_rmses.append(compute_rmse(true_s, pred_s))
    
    metrics["station_mae_mean"] = np.mean(station_maes)
    metrics["station_mae_std"] = np.std(station_maes)
    metrics["station_rmse_mean"] = np.mean(station_rmses)
    
    # Metrics for high-flow periods (when |flow| > threshold)
    high_flow_threshold = np.percentile(np.abs(true_flat), 75)
    high_flow_mask = np.abs(true_flat) > high_flow_threshold
    
    if np.sum(high_flow_mask) > 0:
        metrics["high_flow_mae"] = compute_mae(
            true_flat[high_flow_mask], 
            pred_flat[high_flow_mask]
        )
        metrics["high_flow_direction_acc"] = compute_direction_accuracy(
            true_flat[high_flow_mask],
            pred_flat[high_flow_mask]
        )
    
    return metrics


def summarize_fold_results(fold_results: list) -> Dict[str, Tuple[float, float]]:
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
            # Only compute stats for numeric values
            try:
                numeric_values = [float(v) for v in values]
                summary[metric] = (np.mean(numeric_values), np.std(numeric_values))
            except (ValueError, TypeError):
                # Skip non-numeric metrics
                continue
    
    return summary
