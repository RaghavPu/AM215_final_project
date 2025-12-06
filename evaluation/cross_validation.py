"""Rolling window cross-validation for time series flow prediction."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

from .metrics import compute_flow_metrics, summarize_fold_results


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class RollingWindowCV:
    """Rolling window cross-validation splitter.
    
    Creates train/test splits where:
    - Train: N weeks of data
    - Test: M weeks of data immediately following train
    - Rolls forward by M weeks for each fold
    """
    
    def __init__(
        self,
        train_weeks: int = 3,
        test_weeks: int = 1,
    ):
        """Initialize the splitter.
        
        Args:
            train_weeks: Number of weeks in training window
            test_weeks: Number of weeks in test window
        """
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
    
    def split(
        self,
        trips: pd.DataFrame,
    ) -> Generator[CVFold, None, None]:
        """Generate train/test splits.
        
        Args:
            trips: Trip data with 'started_at' column
            
        Yields:
            CVFold objects with train/test date ranges
        """
        # Get date range
        min_date = trips["started_at"].min().normalize()
        max_date = trips["started_at"].max().normalize()
        
        train_delta = pd.Timedelta(weeks=self.train_weeks)
        test_delta = pd.Timedelta(weeks=self.test_weeks)
        
        fold_id = 0
        train_start = min_date
        
        while True:
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta
            
            # Check if we have enough data
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
            train_start = train_start + test_delta  # Roll forward
    
    def get_n_splits(self, trips: pd.DataFrame) -> int:
        """Count number of splits."""
        return sum(1 for _ in self.split(trips))


def compute_actual_flow(
    trips: pd.DataFrame,
    stations: list,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    freq: str = "1h",
) -> pd.DataFrame:
    """Compute actual net flow from trip data.
    
    Net flow = arrivals - departures per station per time period.
    This is GROUND TRUTH - directly observable from the data.
    
    Args:
        trips: Trip data
        stations: List of station names
        start_time: Start of time range
        end_time: End of time range
        freq: Time frequency for aggregation
        
    Returns:
        DataFrame with actual net flow (index=stations, columns=times)
    """
    # Filter to time range
    mask = (trips["started_at"] >= start_time) & (trips["started_at"] < end_time)
    trips_subset = trips[mask].copy()
    
    # Generate time buckets
    times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
    
    # Initialize flow DataFrame
    flow = pd.DataFrame(0.0, index=stations, columns=times)
    
    if len(trips_subset) == 0:
        return flow
    
    # Bucket trips by hour
    trips_subset["time_bucket"] = trips_subset["started_at"].dt.floor(freq)
    
    # Compute departures per (station, time_bucket)
    departures = (
        trips_subset.groupby(["start_station_name", "time_bucket"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Compute arrivals per (station, time_bucket)  
    arrivals = (
        trips_subset.groupby(["end_station_name", "time_bucket"])
        .size()
        .unstack(fill_value=0)
    )
    
    # Fill in the flow DataFrame
    for station in stations:
        for t in times:
            dep = departures.loc[station, t] if (station in departures.index and t in departures.columns) else 0
            arr = arrivals.loc[station, t] if (station in arrivals.index and t in arrivals.columns) else 0
            flow.loc[station, t] = arr - dep
    
    return flow


def run_cross_validation(
    model,
    trips: pd.DataFrame,
    station_stats: pd.DataFrame,
    config: dict,
    verbose: bool = True,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """Run rolling window cross-validation for flow prediction.
    
    Args:
        model: Model instance (must have fit/predict_flow methods)
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
    )
    
    stations = station_stats.index.tolist()
    
    fold_results = []
    folds = list(cv.split(trips))
    
    if verbose:
        print(f"\nRunning {len(folds)}-fold cross-validation...")
        print(f"Predicting net flow for {len(stations)} stations")
    
    for fold in tqdm(folds, desc="CV Folds", disable=not verbose):
        # Split data
        train_mask = (
            (trips["started_at"] >= fold.train_start) &
            (trips["started_at"] < fold.train_end)
        )
        test_mask = (
            (trips["started_at"] >= fold.test_start) &
            (trips["started_at"] < fold.test_end)
        )
        
        train_trips = trips[train_mask]
        test_trips = trips[test_mask]
        
        if len(train_trips) == 0 or len(test_trips) == 0:
            continue
        
        # Fit model on training data
        model.fit(train_trips, station_stats)
        
        # Compute actual flow for test period (GROUND TRUTH)
        true_flow = compute_actual_flow(
            test_trips,
            stations,
            fold.test_start,
            fold.test_end,
            freq="1h",
        )
        
        # Predict flow for test period
        pred_flow = model.predict_flow(
            stations,
            fold.test_start,
            fold.test_end,
            freq="1h",
        )
        
        # Compute metrics
        metrics = compute_flow_metrics(true_flow, pred_flow)
        metrics["fold_id"] = fold.fold_id
        metrics["train_start"] = str(fold.train_start.date())
        metrics["test_start"] = str(fold.test_start.date())
        
        fold_results.append(metrics)
        
        if verbose:
            print(f"  Fold {fold.fold_id}: MAE={metrics['mae']:.2f}, "
                  f"RMSE={metrics['rmse']:.2f}, "
                  f"Direction Acc={metrics['direction_accuracy']:.1%}, "
                  f"Corr={metrics['correlation']:.3f}")
    
    # Summarize
    summary = summarize_fold_results(fold_results)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Cross-Validation Summary (mean ± std across folds):")
        print("=" * 60)
        key_metrics = ["mae", "rmse", "direction_accuracy", "correlation", "high_flow_mae"]
        for metric in key_metrics:
            if metric in summary:
                mean, std = summary[metric]
                if "accuracy" in metric or "correlation" in metric:
                    print(f"  {metric}: {mean:.3f} ± {std:.3f}")
                else:
                    print(f"  {metric}: {mean:.2f} ± {std:.2f}")
    
    return fold_results, summary
