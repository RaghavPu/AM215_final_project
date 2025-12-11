"""Data loading utilities for CitiBike trip data."""

from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm


def load_trip_data(
    data_dir: str = "data",
    start_date: str | None = None,
    end_date: str | None = None,
    use_parquet: bool = True,
) -> pd.DataFrame:
    """Load trip data from Parquet or CSV files using DuckDB.

    Args:
        data_dir: Directory containing trip data folders
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        use_parquet: If True, use Parquet files; otherwise fall back to CSV

    Returns:
        DataFrame with all trip data
    """
    data_path = Path(data_dir)

    # Try Parquet first if requested
    if use_parquet:
        parquet_path = data_path / "parquet" / "trips"

        if parquet_path.exists():
            print("Loading trip data from Parquet files using DuckDB...")
            return _load_from_parquet_duckdb(parquet_path, start_date, end_date)
        else:
            print(f"Parquet directory not found at {parquet_path}, falling back to CSV...")

    # Fallback to CSV loading
    return _load_from_csv(data_path, start_date, end_date)


def _load_from_parquet_duckdb(
    parquet_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load trip data from Parquet files using DuckDB for efficient querying.

    Args:
        parquet_path: Path to parquet directory (with year=YYYY partitions)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with trip data
    """
    con = duckdb.connect()

    try:
        # Build query with optional date filters
        where_clauses = []
        if start_date:
            where_clauses.append(f"started_at >= '{start_date}'")
        if end_date:
            where_clauses.append(f"started_at <= '{end_date}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Use glob pattern to read all parquet files
        parquet_pattern = str(parquet_path / "**" / "*.parquet")

        query = f"""
            SELECT *
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
            {where_sql}
        """

        print("  Executing DuckDB query on Parquet files...")
        print(f"  Pattern: {parquet_pattern}")
        if where_sql:
            print(f"  Filters: {where_sql}")

        df = con.execute(query).fetchdf()

        # Ensure datetime columns are parsed
        if "started_at" in df.columns:
            df["started_at"] = pd.to_datetime(df["started_at"])
        if "ended_at" in df.columns:
            df["ended_at"] = pd.to_datetime(df["ended_at"])

        print(
            f"  ✓ Loaded {len(df):,} trips from {df['started_at'].min().date()} to {df['started_at'].max().date()}"
        )

        return df

    finally:
        con.close()


def _load_from_csv(
    data_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load trip data from CSV files (legacy fallback).

    Args:
        data_path: Path to data directory
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with trip data
    """
    # Find all CSV files in subdirectories
    csv_files = list(data_path.glob("**/202*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No trip data files found in {data_path}")

    print(f"Found {len(csv_files)} trip data CSV files")

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

    print(
        f"Loaded {len(df):,} trips from {df['started_at'].min().date()} to {df['started_at'].max().date()}"
    )

    return df


def load_station_info(
    station_path: str = "data/stations/station_info.csv", use_parquet: bool = True
) -> pd.DataFrame:
    """Load station information including capacity.

    Args:
        station_path: Path to station info file (CSV or Parquet)
        use_parquet: If True, try Parquet first

    Returns:
        DataFrame with station information
    """
    station_path_obj = Path(station_path)

    # Try Parquet version first if requested
    if use_parquet:
        # Try replacing extension
        parquet_path = station_path_obj.with_suffix(".parquet")

        # Also try common parquet directory structure
        if not parquet_path.exists():
            data_dir = Path(station_path_obj.parts[0]) if station_path_obj.parts else Path("data")
            parquet_path = data_dir / "parquet" / "stations" / "station_info.parquet"

        if parquet_path.exists():
            print(f"Loading station info from Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            print(f"Loaded {len(df)} stations with total capacity {df['capacity'].sum():,}")
            return df

    # Fallback to CSV
    if station_path_obj.exists():
        df = pd.read_csv(station_path)
        print(f"Loaded {len(df)} stations with total capacity {df['capacity'].sum():,}")
        return df
    else:
        raise FileNotFoundError(
            f"Station info file not found: {station_path} or parquet alternative"
        )


def prepare_data(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        trips.groupby("start_station_name").size() + trips.groupby("end_station_name").size()
    )
    valid_stations = station_trip_counts[station_trip_counts >= min_trips].index.tolist()

    trips = trips[
        trips["start_station_name"].isin(valid_stations)
        & trips["end_station_name"].isin(valid_stations)
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
