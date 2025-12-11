"""Pytest configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    """Minimal configuration for testing."""
    return {
        "data": {
            "trip_data_dir": "data",
            "station_info_path": "data/stations/station_info.csv",
            "output_dir": "outputs",
        },
        "time": {
            "start_date": "2025-09-01",
            "end_date": "2025-09-30",
        },
        "cross_validation": {
            "train_weeks": 2,
            "test_weeks": 1,
            "increment_days": 7,
        },
        "stations": {
            "min_trips": 10,
        },
        "thresholds": {
            "empty": 0.1,
            "full": 0.9,
        },
        "model": {
            "name": "temporal_flow",
            "markov": {
                "smoothing_alpha": 0.0,
                "min_transitions": 1,
                "n_simulations": 1,
            },
        },
        "evaluation": {
            "freq": "1h",
        },
    }


@pytest.fixture
def sample_trips():
    """Generate sample trip data for testing."""
    np.random.seed(42)
    n_trips = 1000

    stations = ["Station A", "Station B", "Station C", "Station D"]

    # Generate random trips over 2 weeks
    base_date = pd.Timestamp("2025-09-01")
    random_hours = np.random.randint(0, 24 * 14, n_trips)  # 2 weeks of hours

    started_at = [base_date + pd.Timedelta(hours=int(h)) for h in random_hours]
    ended_at = [t + pd.Timedelta(minutes=np.random.randint(5, 60)) for t in started_at]

    trips = pd.DataFrame(
        {
            "started_at": started_at,
            "ended_at": ended_at,
            "start_station_name": np.random.choice(stations, n_trips),
            "end_station_name": np.random.choice(stations, n_trips),
            "rideable_type": np.random.choice(["classic_bike", "electric_bike"], n_trips),
        }
    )

    # Add derived columns
    trips["hour"] = trips["started_at"].dt.hour
    trips["day_of_week"] = trips["started_at"].dt.dayofweek
    trips["is_weekend"] = trips["day_of_week"].isin([5, 6])

    return trips.sort_values("started_at").reset_index(drop=True)


@pytest.fixture
def sample_station_stats():
    """Generate sample station statistics."""
    stations = ["Station A", "Station B", "Station C", "Station D"]
    return pd.DataFrame(
        {
            "capacity": [30, 25, 40, 20],
        },
        index=stations,
    )


@pytest.fixture
def sample_inventory():
    """Generate sample inventory series."""
    stations = ["Station A", "Station B", "Station C", "Station D"]
    return pd.Series([15, 12, 20, 10], index=stations)


@pytest.fixture
def sample_inventory_df():
    """Generate sample inventory DataFrame (stations x times)."""
    stations = ["Station A", "Station B", "Station C"]
    times = pd.date_range("2025-09-01", periods=24, freq="1h")

    np.random.seed(42)
    data = np.random.randint(5, 25, size=(len(stations), len(times)))

    return pd.DataFrame(data, index=stations, columns=times)


@pytest.fixture
def golden_files_dir():
    """Path to golden files directory."""
    return Path(__file__).parent / "golden_files"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
