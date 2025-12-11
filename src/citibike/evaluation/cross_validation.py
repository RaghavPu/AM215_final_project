"""Rolling window cross-validation for inventory prediction."""

from collections.abc import Generator
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .metrics import compute_inventory_metrics, summarize_fold_results


@dataclass
class CVFold:
    """A single cross-validation fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class RollingWindowCV:
    """Rolling window cross-validation splitter."""

    def __init__(
        self,
        train_weeks: int = 3,
        test_weeks: int = 1,
        increment_days: int = None,
    ):
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        # If increment_days not specified, default to test_weeks worth of days
        self.increment_days = increment_days if increment_days is not None else (test_weeks * 7)

    def split(
        self,
        trips: pd.DataFrame,
    ) -> Generator[CVFold, None, None]:
        """Generate train/test splits."""
        min_date = trips["started_at"].min().normalize()
        max_date = trips["started_at"].max().normalize()

        train_delta = pd.Timedelta(weeks=self.train_weeks)
        test_delta = pd.Timedelta(weeks=self.test_weeks)
        increment_delta = pd.Timedelta(days=self.increment_days)

        fold_id = 0
        train_start = min_date

        while True:
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > max_date:
                break

            yield CVFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            fold_id += 1
            train_start = train_start + increment_delta

    def get_n_splits(self, trips: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(trips))


def track_inventory(
    trips: pd.DataFrame,
    initial_inventory: pd.Series,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq: str = "1h",
) -> pd.DataFrame:
    """Track actual inventory by applying trips to initial state.

    This gives us GROUND TRUTH inventory - what actually happened.

    Args:
        trips: Trip data for the period
        initial_inventory: Starting bike count per station
        start_time: Start of tracking period
        end_time: End of tracking period
        freq: Time frequency

    Returns:
        DataFrame with actual inventory (index=stations, columns=times)
    """
    stations = initial_inventory.index.tolist()
    times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")

    # Initialize
    inventory = pd.DataFrame(index=stations, columns=times, dtype=float)
    inventory[times[0]] = initial_inventory

    # Filter trips to time range
    mask = (trips["started_at"] >= start_time) & (trips["started_at"] < end_time)
    period_trips = trips[mask].copy()

    if len(period_trips) == 0:
        # No trips, inventory stays constant
        for t in times:
            inventory[t] = initial_inventory
        return inventory

    # Bucket trips by hour
    period_trips["hour_bucket"] = period_trips["started_at"].dt.floor(freq)

    # Track hour by hour
    current_inventory = initial_inventory.copy()

    for i, t in enumerate(times[:-1]):
        # Get trips in this hour
        hour_trips = period_trips[period_trips["hour_bucket"] == t]

        if len(hour_trips) > 0:
            # Count arrivals and departures
            arrivals = hour_trips.groupby("end_station_name").size()
            departures = hour_trips.groupby("start_station_name").size()

            # Apply to inventory
            for station in stations:
                arr = arrivals.get(station, 0)
                dep = departures.get(station, 0)
                current_inventory[station] = max(0, current_inventory[station] - dep + arr)

        # Store state at next time
        inventory[times[i + 1]] = current_inventory.copy()

    return inventory


def compute_initial_inventory_for_fold(
    trips: pd.DataFrame,
    stations: list,
    fold_start: pd.Timestamp,
) -> pd.Series:
    """Compute initial inventory at start of fold using backward tracking.

    Uses trips before fold_start to infer the bike distribution.

    Args:
        trips: All trip data
        stations: List of station names
        fold_start: Start time of the fold

    Returns:
        Series with estimated bike count per station
    """
    # Get trips before fold start (use last week for burn-in)
    burn_in_start = fold_start - pd.Timedelta(weeks=1)
    mask = (trips["started_at"] >= burn_in_start) & (trips["started_at"] < fold_start)
    burn_in_trips = trips[mask].copy()

    if len(burn_in_trips) == 0:
        # No burn-in data, use uniform distribution
        # Assume 50% of typical capacity (15 bikes)
        return pd.Series(15.0, index=stations)

    # Track forward from zero to get ending state
    inventory = dict.fromkeys(stations, 0)

    burn_in_trips["hour_bucket"] = burn_in_trips["started_at"].dt.floor("1h")
    hours = sorted(burn_in_trips["hour_bucket"].unique())

    for hour in hours:
        hour_trips = burn_in_trips[burn_in_trips["hour_bucket"] == hour]

        arrivals = hour_trips.groupby("end_station_name").size()
        departures = hour_trips.groupby("start_station_name").size()

        for station, count in arrivals.items():
            if station in inventory:
                inventory[station] += count

        for station, count in departures.items():
            if station in inventory:
                inventory[station] = max(0, inventory[station] - count)

    return pd.Series(inventory)


def run_cross_validation(
    model,
    trips: pd.DataFrame,
    station_stats: pd.DataFrame,
    config: dict,
    verbose: bool = True,
) -> tuple[list[dict[str, float]], dict[str, tuple[float, float]]]:
    """Run rolling window cross-validation for inventory prediction.

    Args:
        model: Model instance (must have fit/predict_inventory methods)
        trips: Trip data
        station_stats: Station information with capacity
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Tuple of (fold_results, summary)
    """
    cv_config = config.get("cross_validation", {})
    cv = RollingWindowCV(
        train_weeks=cv_config.get("train_weeks", 3),
        test_weeks=cv_config.get("test_weeks", 1),
        increment_days=cv_config.get("increment_days", None),
    )

    stations = station_stats.index.tolist()
    capacities = station_stats["capacity"].to_dict()
    thresholds = config.get("thresholds", {"empty": 0.1, "full": 0.9})

    fold_results = []
    folds = list(cv.split(trips))

    if verbose:
        print(f"\nRunning {len(folds)}-fold cross-validation...")
        print(f"Predicting inventory for {len(stations)} stations")

    for fold in tqdm(folds, desc="CV Folds", disable=not verbose):
        # Split data
        train_mask = (trips["started_at"] >= fold.train_start) & (
            trips["started_at"] < fold.train_end
        )
        test_mask = (trips["started_at"] >= fold.test_start) & (trips["started_at"] < fold.test_end)

        train_trips = trips[train_mask]
        test_trips = trips[test_mask]

        if len(train_trips) == 0 or len(test_trips) == 0:
            continue

        # Fit model on training data
        model.fit(train_trips, station_stats)

        # Compute initial inventory at start of test period
        # Use end of training period to estimate starting state
        initial_inventory = compute_initial_inventory_for_fold(
            trips,
            stations,
            fold.test_start,
        )

        # Clamp to capacity
        for station in stations:
            cap = capacities.get(station, 30)
            initial_inventory[station] = min(initial_inventory[station], cap)

        # Track actual inventory (GROUND TRUTH)
        true_inventory = track_inventory(
            test_trips,
            initial_inventory,
            fold.test_start,
            fold.test_end,
            freq="1h",
        )

        # Predict inventory
        pred_inventory = model.predict_inventory(
            initial_inventory,
            fold.test_start,
            fold.test_end,
            freq="1h",
        )

        # Compute metrics
        metrics = compute_inventory_metrics(
            true_inventory,
            pred_inventory,
            capacities,
            thresholds,
        )
        metrics["fold_id"] = fold.fold_id
        metrics["train_start"] = str(fold.train_start.date())
        metrics["test_start"] = str(fold.test_start.date())

        fold_results.append(metrics)

        if verbose:
            print(
                f"  Fold {fold.fold_id}: "
                f"MAE={metrics.get('inventory_mae', 0):.2f} bikes, "
                f"Empty Recall={metrics.get('empty_recall', 0):.1%}, "
                f"Full Recall={metrics.get('full_recall', 0):.1%}, "
                f"State Acc={metrics.get('state_accuracy', 0):.1%}"
            )

    # Summarize
    summary = summarize_fold_results(fold_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Cross-Validation Summary (mean ± std across folds):")
        print("=" * 60)
        key_metrics = [
            "inventory_mae",
            "inventory_rmse",
            "correlation",
            "empty_recall",
            "empty_precision",
            "empty_f1",
            "full_recall",
            "full_precision",
            "full_f1",
            "state_accuracy",
        ]
        for metric in key_metrics:
            if metric in summary:
                mean, std = summary[metric]
                if (
                    "recall" in metric
                    or "precision" in metric
                    or "accuracy" in metric
                    or "f1" in metric
                ):
                    print(f"  {metric}: {mean:.1%} ± {std:.1%}")
                else:
                    print(f"  {metric}: {mean:.2f} ± {std:.2f}")

    return fold_results, summary
