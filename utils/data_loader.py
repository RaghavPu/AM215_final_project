"""Data loading utilities for CitiBike trip data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm


def load_trip_data(
    data_dir: str = "data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load all trip data from CSV files.
    
    Args:
        data_dir: Directory containing trip data folders
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with all trip data
    """
    data_path = Path(data_dir)
    
    # Find all CSV files in subdirectories
    csv_files = list(data_path.glob("**/202*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No trip data CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} trip data files")
    
    # Load all files
    dfs = []
    for f in tqdm(csv_files, desc="Loading trip data"):
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Parse timestamps
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    
    # Filter by date if specified
    if start_date:
        df = df[df["started_at"] >= start_date]
    if end_date:
        df = df[df["started_at"] <= end_date]
    
    print(f"Loaded {len(df):,} trips from {df['started_at'].min().date()} to {df['started_at'].max().date()}")
    
    return df


def load_station_info(station_path: str = "data/stations/station_info.csv") -> pd.DataFrame:
    """Load station information including capacity.
    
    Args:
        station_path: Path to station info CSV
        
    Returns:
        DataFrame with station information
    """
    df = pd.read_csv(station_path)
    print(f"Loaded {len(df)} stations with total capacity {df['capacity'].sum():,}")
    return df


def prepare_data(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for modeling.
    
    - Filters to valid stations
    - Adds time features
    - Merges with station capacity
    
    Args:
        trips: Raw trip data
        stations: Station information
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed_trips, station_stats)
    """
    # Filter out missing station names
    trips = trips.dropna(subset=["start_station_name", "end_station_name"]).copy()
    
    # Add time features
    trips["hour"] = trips["started_at"].dt.hour
    trips["day_of_week"] = trips["started_at"].dt.dayofweek
    trips["date"] = trips["started_at"].dt.date
    trips["is_weekend"] = trips["day_of_week"].isin([5, 6])
    
    # Filter stations by minimum trips
    min_trips = config.get("stations", {}).get("min_trips", 100)
    station_trip_counts = (
        trips.groupby("start_station_name").size() +
        trips.groupby("end_station_name").size()
    )
    valid_stations = station_trip_counts[station_trip_counts >= min_trips].index.tolist()
    
    trips = trips[
        trips["start_station_name"].isin(valid_stations) &
        trips["end_station_name"].isin(valid_stations)
    ]
    
    print(f"After filtering: {len(trips):,} trips, {len(valid_stations)} stations")
    
    # Create station stats with capacity
    station_stats = stations[["name", "capacity"]].copy()
    station_stats = station_stats.rename(columns={"name": "station_name"})
    station_stats = station_stats.drop_duplicates(subset=["station_name"])
    station_stats = station_stats.set_index("station_name")
    
    # Only keep stations that appear in our trip data
    station_stats = station_stats[station_stats.index.isin(valid_stations)]
    
    # For stations in trips but not in station info, estimate capacity
    missing_stations = set(valid_stations) - set(station_stats.index)
    if missing_stations:
        print(f"Warning: {len(missing_stations)} stations missing capacity info, using median")
        median_capacity = station_stats["capacity"].median()
        for station in missing_stations:
            station_stats.loc[station, "capacity"] = median_capacity
    
    return trips, station_stats

