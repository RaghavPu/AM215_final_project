"""DuckDB utilities for efficient data querying and processing."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any


class DuckDBConnection:
    """Context manager for DuckDB connections."""

    def __init__(self, database: Optional[str] = None):
        """Initialize connection.

        Args:
            database: Path to database file. If None, uses in-memory database.
        """
        self.database = database
        self.con = None

    def __enter__(self):
        """Enter context and create connection."""
        self.con = duckdb.connect(self.database)
        return self.con

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and close connection."""
        if self.con:
            self.con.close()


def query_parquet(
    parquet_path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Query Parquet files with DuckDB.

    Args:
        parquet_path: Path to parquet directory or file
        columns: List of columns to select (None = all)
        filters: Dictionary of column filters (e.g., {"year": 2025, "month": 9})
        limit: Maximum number of rows to return

    Returns:
        DataFrame with query results
    """
    with DuckDBConnection() as con:
        # Build SELECT clause
        select_cols = ", ".join(columns) if columns else "*"

        # Build WHERE clause
        where_clauses = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    where_clauses.append(f"{col} = '{val}'")
                else:
                    where_clauses.append(f"{col} = {val}")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Build LIMIT clause
        limit_sql = f"LIMIT {limit}" if limit else ""

        # Construct pattern
        if parquet_path.is_dir():
            pattern = str(parquet_path / "**" / "*.parquet")
        else:
            pattern = str(parquet_path)

        # Execute query
        query = f"""
            SELECT {select_cols}
            FROM read_parquet('{pattern}', hive_partitioning=1)
            {where_sql}
            {limit_sql}
        """

        return con.execute(query).fetchdf()


def aggregate_trips(
    parquet_path: Path,
    group_by: List[str],
    aggregations: Dict[str, str],
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Aggregate trip data using DuckDB.

    Args:
        parquet_path: Path to parquet files
        group_by: Columns to group by
        aggregations: Dict of {output_col: aggregation_expr}
            Example: {"trip_count": "COUNT(*)", "avg_duration": "AVG(duration)"}
        filters: Optional filters to apply before aggregation

    Returns:
        DataFrame with aggregated results
    """
    with DuckDBConnection() as con:
        # Build WHERE clause
        where_clauses = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    where_clauses.append(f"{col} = '{val}'")
                else:
                    where_clauses.append(f"{col} = {val}")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Build aggregation expressions
        agg_exprs = [f"{expr} as {col}" for col, expr in aggregations.items()]
        agg_sql = ", ".join(agg_exprs)

        # Build GROUP BY clause
        group_sql = ", ".join(group_by)

        # Construct pattern
        pattern = str(parquet_path / "**" / "*.parquet")

        # Execute query
        query = f"""
            SELECT {group_sql}, {agg_sql}
            FROM read_parquet('{pattern}', hive_partitioning=1)
            {where_sql}
            GROUP BY {group_sql}
            ORDER BY {group_sql}
        """

        return con.execute(query).fetchdf()


def get_trip_stats(
    parquet_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Get summary statistics for trip data.

    Args:
        parquet_path: Path to parquet files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Dictionary with statistics
    """
    with DuckDBConnection() as con:
        # Build date filters
        where_clauses = []
        if start_date:
            where_clauses.append(f"started_at >= '{start_date}'")
        if end_date:
            where_clauses.append(f"started_at <= '{end_date}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        pattern = str(parquet_path / "**" / "*.parquet")

        # Get overall stats
        query = f"""
            SELECT
                COUNT(*) as total_trips,
                COUNT(DISTINCT start_station_name) as n_start_stations,
                COUNT(DISTINCT end_station_name) as n_end_stations,
                MIN(started_at) as first_trip,
                MAX(started_at) as last_trip,
                AVG(EXTRACT(EPOCH FROM (ended_at - started_at)) / 60) as avg_duration_minutes
            FROM read_parquet('{pattern}', hive_partitioning=1)
            {where_sql}
        """

        result = con.execute(query).fetchdf()

        return {
            "total_trips": int(result["total_trips"].iloc[0]),
            "n_start_stations": int(result["n_start_stations"].iloc[0]),
            "n_end_stations": int(result["n_end_stations"].iloc[0]),
            "first_trip": result["first_trip"].iloc[0],
            "last_trip": result["last_trip"].iloc[0],
            "avg_duration_minutes": float(result["avg_duration_minutes"].iloc[0]),
        }


def count_trips_by_station(
    parquet_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Count trips by station (both starts and ends).

    Args:
        parquet_path: Path to parquet files
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with station trip counts
    """
    with DuckDBConnection() as con:
        where_clauses = []
        if start_date:
            where_clauses.append(f"started_at >= '{start_date}'")
        if end_date:
            where_clauses.append(f"started_at <= '{end_date}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        pattern = str(parquet_path / "**" / "*.parquet")

        # Count starts and ends separately, then combine
        query = f"""
            WITH starts AS (
                SELECT
                    start_station_name as station_name,
                    COUNT(*) as start_count
                FROM read_parquet('{pattern}', hive_partitioning=1)
                {where_sql}
                GROUP BY start_station_name
            ),
            ends AS (
                SELECT
                    end_station_name as station_name,
                    COUNT(*) as end_count
                FROM read_parquet('{pattern}', hive_partitioning=1)
                {where_sql}
                GROUP BY end_station_name
            )
            SELECT
                COALESCE(starts.station_name, ends.station_name) as station_name,
                COALESCE(start_count, 0) as departures,
                COALESCE(end_count, 0) as arrivals,
                COALESCE(start_count, 0) + COALESCE(end_count, 0) as total_trips
            FROM starts
            FULL OUTER JOIN ends ON starts.station_name = ends.station_name
            ORDER BY total_trips DESC
        """

        return con.execute(query).fetchdf()


def export_to_parquet(
    df: pd.DataFrame,
    output_path: Path,
    partition_cols: Optional[List[str]] = None,
    compression: str = "zstd",
) -> None:
    """Export DataFrame to Parquet using DuckDB.

    Args:
        df: DataFrame to export
        output_path: Output path for parquet file
        partition_cols: Columns to partition by (creates subdirectories)
        compression: Compression algorithm (zstd, snappy, gzip, etc.)
    """
    with DuckDBConnection() as con:
        # Register DataFrame as a view
        con.register("temp_df", df)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build partition clause
        partition_sql = ""
        if partition_cols:
            partition_sql = f"PARTITION_BY ({', '.join(partition_cols)})"

        # Export query
        query = f"""
            COPY temp_df TO '{output_path}' (
                FORMAT PARQUET,
                COMPRESSION '{compression}',
                {partition_sql}
            )
        """

        con.execute(query)


def create_summary_table(
    parquet_path: Path,
    output_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """Create a summary table for faster querying.

    Aggregates trips by hour and station for modeling.

    Args:
        parquet_path: Path to raw parquet files
        output_path: Path for summary parquet output
        start_date: Optional start date filter
        end_date: Optional end date filter
    """
    with DuckDBConnection() as con:
        where_clauses = []
        if start_date:
            where_clauses.append(f"started_at >= '{start_date}'")
        if end_date:
            where_clauses.append(f"started_at <= '{end_date}'")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        pattern = str(parquet_path / "**" / "*.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create hourly summary
        query = f"""
            COPY (
                SELECT
                    DATE_TRUNC('hour', started_at) as hour,
                    start_station_name,
                    end_station_name,
                    EXTRACT(HOUR FROM started_at) as hour_of_day,
                    EXTRACT(DOW FROM started_at) IN (0, 6) as is_weekend,
                    COUNT(*) as trip_count,
                    AVG(EXTRACT(EPOCH FROM (ended_at - started_at)) / 60) as avg_duration_minutes
                FROM read_parquet('{pattern}', hive_partitioning=1)
                {where_sql}
                GROUP BY
                    DATE_TRUNC('hour', started_at),
                    start_station_name,
                    end_station_name,
                    EXTRACT(HOUR FROM started_at),
                    EXTRACT(DOW FROM started_at) IN (0, 6)
                ORDER BY hour, start_station_name, end_station_name
            ) TO '{output_path}' (
                FORMAT PARQUET,
                COMPRESSION 'zstd'
            )
        """

        print(f"Creating summary table at {output_path}...")
        con.execute(query)
        print(f"✓ Summary table created")
