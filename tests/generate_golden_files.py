#!/usr/bin/env python3
"""Generate golden files for regression tests.

Run this script to create/update golden files after intentional model changes:
    python tests/generate_golden_files.py

Golden files are used by test_regression.py to detect unintended changes.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

import numpy as np
import pandas as pd

from citibike.evaluation import compute_inventory_metrics, inventory_to_states, track_inventory
from citibike.models import MarkovModel, TemporalFlowModel

GOLDEN_DIR = Path(__file__).parent / "golden_files"


def create_deterministic_trips() -> pd.DataFrame:
    """Create deterministic trip data for regression tests."""
    np.random.seed(12345)
    n_trips = 500

    stations = ["Station A", "Station B", "Station C"]

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
    """Create deterministic station stats."""
    return pd.DataFrame(
        {
            "capacity": [30, 25, 40],
        },
        index=["Station A", "Station B", "Station C"],
    )


def create_deterministic_config() -> dict:
    """Create deterministic config."""
    return {
        "thresholds": {"empty": 0.1, "full": 0.9},
        "model": {
            "markov": {
                "smoothing_alpha": 0.0,
                "min_transitions": 1,
                "n_simulations": 1,
            }
        },
    }


def generate_temporal_flow_golden():
    """Generate golden files for TemporalFlowModel."""
    print("Generating TemporalFlowModel golden files...")

    trips = create_deterministic_trips()
    station_stats = create_deterministic_station_stats()
    config = create_deterministic_config()

    model = TemporalFlowModel(config)
    model.fit(trips, station_stats)

    # Save hourly flows (convert tuple keys to strings for JSON)
    hourly_flows = {}
    for key, value in model.hourly_net_flow.items():
        key_str = f"{key[0]},{key[1]},{key[2]}"
        hourly_flows[key_str] = float(value)

    with open(GOLDEN_DIR / "temporal_flow_hourly_flows.json", "w") as f:
        json.dump(hourly_flows, f, indent=2)
    print(f"  Saved {len(hourly_flows)} hourly flow values")

    # Save predictions
    initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
    start = pd.Timestamp("2025-09-15 08:00:00")
    end = pd.Timestamp("2025-09-15 14:00:00")

    predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")
    np.save(GOLDEN_DIR / "temporal_flow_predictions.npy", predictions.values)
    print(f"  Saved predictions array shape {predictions.shape}")


def generate_markov_golden():
    """Generate golden files for MarkovModel."""
    print("Generating MarkovModel golden files...")

    trips = create_deterministic_trips()
    station_stats = create_deterministic_station_stats()
    config = create_deterministic_config()

    model = MarkovModel(config)
    model.fit(trips, station_stats)

    # Save transition matrix stats
    P = model.global_transition_matrix.toarray()
    stats = {
        "n_stations": len(model.stations),
        "n_matrices": len(model.transition_matrices),
        "global_matrix_shape": list(P.shape),
        "global_matrix_nnz": int(model.global_transition_matrix.nnz),
    }

    with open(GOLDEN_DIR / "markov_transition_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved transition matrix stats")

    # Save predictions
    initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
    start = pd.Timestamp("2025-09-15 08:00:00")
    end = pd.Timestamp("2025-09-15 14:00:00")

    predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")
    np.save(GOLDEN_DIR / "markov_predictions.npy", predictions.values)
    print(f"  Saved predictions array shape {predictions.shape}")


def generate_metrics_golden():
    """Generate golden files for metrics."""
    print("Generating metrics golden files...")

    # Create deterministic test data
    np.random.seed(42)
    stations = ["Station A", "Station B", "Station C"]
    times = pd.date_range("2025-09-01", periods=24, freq="1h")

    true_inv = pd.DataFrame(
        np.random.randint(5, 25, size=(3, 24)),
        index=stations,
        columns=times,
    )

    np.random.seed(43)
    pred_inv = true_inv + np.random.randint(-3, 4, size=(3, 24))
    pred_inv = pred_inv.clip(lower=0, upper=30)

    capacities = {"Station A": 30, "Station B": 25, "Station C": 40}
    thresholds = {"empty": 0.1, "full": 0.9}

    metrics = compute_inventory_metrics(true_inv, pred_inv, capacities, thresholds)

    # Save key metrics
    metrics_to_save = {
        "inventory_mae": float(metrics["inventory_mae"]),
        "inventory_rmse": float(metrics["inventory_rmse"]),
        "correlation": float(metrics["correlation"]),
        "state_accuracy": float(metrics["state_accuracy"]),
    }

    with open(GOLDEN_DIR / "metrics_values.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print("  Saved metrics values")

    # Generate state conversion golden
    stations2 = ["Station A", "Station B"]
    times2 = pd.date_range("2025-09-01", periods=5, freq="1h")

    inventory = pd.DataFrame(
        [[2, 5, 15, 27, 29], [1, 3, 12, 22, 24]],
        index=stations2,
        columns=times2,
    )

    capacities2 = {"Station A": 30, "Station B": 25}

    states = inventory_to_states(inventory, capacities2, thresholds)

    # Convert to serializable format
    states_dict = {}
    for col in states.columns:
        col_str = str(col)
        states_dict[col_str] = {}
        for station in states.index:
            states_dict[col_str][station] = states.loc[station, col]

    with open(GOLDEN_DIR / "state_conversion.json", "w") as f:
        json.dump(states_dict, f, indent=2)
    print("  Saved state conversion")


def generate_pipeline_golden():
    """Generate golden file for full pipeline test."""
    print("Generating pipeline golden files...")

    trips = create_deterministic_trips()
    station_stats = create_deterministic_station_stats()
    config = create_deterministic_config()

    model = TemporalFlowModel(config)
    model.fit(trips, station_stats)

    initial_inventory = pd.Series([15, 12, 20], index=["Station A", "Station B", "Station C"])
    start = pd.Timestamp("2025-09-10")
    end = pd.Timestamp("2025-09-11")

    predictions = model.predict_inventory(initial_inventory, start, end, freq="1h")
    actual = track_inventory(trips, initial_inventory, start, end, freq="1h")

    capacities = station_stats["capacity"].to_dict()
    thresholds = config["thresholds"]

    metrics = compute_inventory_metrics(actual, predictions, capacities, thresholds)

    pipeline_metrics = {
        "inventory_mae": float(metrics["inventory_mae"]),
        "inventory_rmse": float(metrics["inventory_rmse"]),
    }

    with open(GOLDEN_DIR / "pipeline_metrics.json", "w") as f:
        json.dump(pipeline_metrics, f, indent=2)
    print("  Saved pipeline metrics")


def main():
    """Generate all golden files."""
    print("=" * 60)
    print("Generating Golden Files for Regression Tests")
    print("=" * 60)

    # Create golden directory
    GOLDEN_DIR.mkdir(exist_ok=True)

    generate_temporal_flow_golden()
    generate_markov_golden()
    generate_metrics_golden()
    generate_pipeline_golden()

    print("=" * 60)
    print("Done! Golden files saved to:", GOLDEN_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
