"""Unit tests for utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import RollingWindowCV, compute_initial_inventory_for_fold, track_inventory
from utils.helpers import get_project_root, load_config


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_existing_config(self):
        """Should load the main config file."""
        config = load_config("config.yaml")

        assert isinstance(config, dict)
        assert "data" in config
        assert "model" in config
        assert "thresholds" in config

    def test_load_config_has_required_keys(self):
        """Config should have all required keys."""
        config = load_config("config.yaml")

        required_keys = ["data", "time", "cross_validation", "stations", "thresholds", "model"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_load_missing_config_raises(self):
        """Loading non-existent config should raise."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestGetProjectRoot:
    """Tests for project root detection."""

    def test_returns_path(self):
        """Should return a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_config(self):
        """Project root should contain config.yaml."""
        root = get_project_root()
        assert (root / "config.yaml").exists()


class TestTrackInventory:
    """Tests for the inventory tracking function."""

    def test_initial_inventory_at_start(self, sample_trips, sample_inventory):
        """First time point should equal initial inventory."""
        start = pd.Timestamp("2025-09-01")
        end = pd.Timestamp("2025-09-01 12:00:00")

        inventory = track_inventory(sample_trips, sample_inventory, start, end, freq="1h")

        # First column should be initial inventory
        first_time = inventory.columns[0]
        pd.testing.assert_series_equal(
            inventory[first_time],
            sample_inventory,
            check_names=False,
        )

    def test_inventory_non_negative(self, sample_trips, sample_inventory):
        """Tracked inventory should never be negative."""
        start = pd.Timestamp("2025-09-01")
        end = pd.Timestamp("2025-09-02")

        inventory = track_inventory(sample_trips, sample_inventory, start, end, freq="1h")

        assert (inventory.values >= 0).all()

    def test_inventory_changes_with_trips(self, sample_inventory):
        """Inventory should change when trips occur."""
        # Create specific trips
        trips = pd.DataFrame(
            {
                "started_at": [
                    pd.Timestamp("2025-09-01 10:30:00"),
                    pd.Timestamp("2025-09-01 10:45:00"),
                ],
                "ended_at": [
                    pd.Timestamp("2025-09-01 11:00:00"),
                    pd.Timestamp("2025-09-01 11:15:00"),
                ],
                "start_station_name": ["Station A", "Station A"],
                "end_station_name": ["Station B", "Station C"],
            }
        )

        start = pd.Timestamp("2025-09-01 10:00:00")
        end = pd.Timestamp("2025-09-01 12:00:00")

        inventory = track_inventory(trips, sample_inventory, start, end, freq="1h")

        # Station A should have fewer bikes after the trips
        initial_a = sample_inventory["Station A"]
        # At 11:00, 2 bikes left Station A
        assert inventory.loc["Station A", pd.Timestamp("2025-09-01 11:00:00")] <= initial_a

    def test_handles_empty_trip_period(self, sample_inventory):
        """Should handle time periods with no trips."""
        empty_trips = pd.DataFrame(
            {
                "started_at": pd.Series([], dtype="datetime64[ns]"),
                "ended_at": pd.Series([], dtype="datetime64[ns]"),
                "start_station_name": pd.Series([], dtype="str"),
                "end_station_name": pd.Series([], dtype="str"),
            }
        )

        start = pd.Timestamp("2025-09-01")
        end = pd.Timestamp("2025-09-01 06:00:00")

        inventory = track_inventory(empty_trips, sample_inventory, start, end, freq="1h")

        # All time points should equal initial (no trips)
        for col in inventory.columns:
            pd.testing.assert_series_equal(
                inventory[col],
                sample_inventory,
                check_names=False,
            )


class TestComputeInitialInventory:
    """Tests for initial inventory computation."""

    def test_returns_series(self, sample_trips):
        """Should return a pandas Series."""
        stations = ["Station A", "Station B", "Station C", "Station D"]
        fold_start = pd.Timestamp("2025-09-08")

        inventory = compute_initial_inventory_for_fold(sample_trips, stations, fold_start)

        assert isinstance(inventory, pd.Series)
        assert len(inventory) == len(stations)

    def test_all_stations_present(self, sample_trips):
        """All requested stations should be in output."""
        stations = ["Station A", "Station B", "Station C", "Station D"]
        fold_start = pd.Timestamp("2025-09-08")

        inventory = compute_initial_inventory_for_fold(sample_trips, stations, fold_start)

        for station in stations:
            assert station in inventory.index

    def test_handles_no_burnin_data(self):
        """Should handle case with no burn-in data."""
        trips = pd.DataFrame(
            {
                "started_at": [pd.Timestamp("2025-09-15")],
                "ended_at": [pd.Timestamp("2025-09-15 00:30:00")],
                "start_station_name": ["Station A"],
                "end_station_name": ["Station B"],
            }
        )
        stations = ["Station A", "Station B"]
        fold_start = pd.Timestamp("2025-09-01")  # Before any trips

        inventory = compute_initial_inventory_for_fold(trips, stations, fold_start)

        # Should return default values (15 bikes)
        assert (inventory == 15.0).all()


class TestRollingWindowCV:
    """Tests for rolling window cross-validation splitter."""

    def test_creates_folds(self):
        """Should create at least one fold."""
        # Create trips spanning 4 weeks to ensure folds can be created
        np.random.seed(42)
        n_trips = 500
        stations = ["Station A", "Station B"]

        base_date = pd.Timestamp("2025-09-01")
        random_hours = np.random.randint(0, 24 * 28, n_trips)  # 4 weeks
        started_at = [base_date + pd.Timedelta(hours=int(h)) for h in random_hours]

        trips = pd.DataFrame(
            {
                "started_at": started_at,
                "start_station_name": np.random.choice(stations, n_trips),
                "end_station_name": np.random.choice(stations, n_trips),
            }
        )

        cv = RollingWindowCV(train_weeks=1, test_weeks=1)
        folds = list(cv.split(trips))

        assert len(folds) >= 1

    def test_fold_dates_are_ordered(self, sample_trips):
        """Fold dates should be properly ordered."""
        cv = RollingWindowCV(train_weeks=1, test_weeks=1, increment_days=3)

        for fold in cv.split(sample_trips):
            assert fold.train_start < fold.train_end
            assert fold.train_end == fold.test_start
            assert fold.test_start < fold.test_end

    def test_train_test_no_overlap(self, sample_trips):
        """Train and test periods should not overlap."""
        cv = RollingWindowCV(train_weeks=1, test_weeks=1)

        for fold in cv.split(sample_trips):
            # Train period ends where test begins
            assert fold.train_end <= fold.test_start

    def test_increment_moves_window(self, sample_trips):
        """Window should move by increment_days."""
        cv = RollingWindowCV(train_weeks=1, test_weeks=1, increment_days=3)
        folds = list(cv.split(sample_trips))

        if len(folds) >= 2:
            delta = folds[1].train_start - folds[0].train_start
            assert delta == pd.Timedelta(days=3)

    def test_get_n_splits(self, sample_trips):
        """get_n_splits should match actual number of folds."""
        cv = RollingWindowCV(train_weeks=1, test_weeks=1)

        n_splits = cv.get_n_splits(sample_trips)
        actual_folds = len(list(cv.split(sample_trips)))

        assert n_splits == actual_folds


class TestConfigThresholds:
    """Tests for configuration threshold values."""

    def test_empty_threshold_valid(self):
        """Empty threshold should be in valid range."""
        config = load_config("config.yaml")
        empty_thresh = config["thresholds"]["empty"]

        assert 0 <= empty_thresh <= 1
        assert empty_thresh < config["thresholds"]["full"]

    def test_full_threshold_valid(self):
        """Full threshold should be in valid range."""
        config = load_config("config.yaml")
        full_thresh = config["thresholds"]["full"]

        assert 0 <= full_thresh <= 1
        assert full_thresh > config["thresholds"]["empty"]
