"""
Compute station inventory through bidirectional tracking.

This script infers bike counts at each station by:
1. Forward pass: Track inventory from start → end
2. Backward pass: Track inventory from end → start

The results are cached in data/inventory/ for use in training.

Usage:
    python scripts/compute_inventory.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "inventory"


def load_trips_for_month(month_dir: Path) -> pd.DataFrame:
    """Load all trip data for a month."""
    csv_files = sorted(month_dir.glob("*.csv"))
    
    if not csv_files:
        return pd.DataFrame()
    
    dfs = []
    for f in tqdm(csv_files, desc=f"Loading {month_dir.name}"):
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    
    # Sort chronologically
    df = df.sort_values("started_at").reset_index(drop=True)
    
    return df


def forward_pass_fast(trips: pd.DataFrame, stations: list) -> dict:
    """
    Track inventory forward through time (FAST vectorized version).
    
    Instead of iterating row by row, we:
    1. Count total arrivals per station
    2. Count total departures per station
    3. Track the "unknown departures" (when we try to subtract below 0)
    
    For accurate tracking, we still need to process chronologically to handle
    the clamping at 0, but we batch by hour for speed.
    """
    inventory = {station: 0 for station in stations}
    
    # Filter to valid stations
    valid_trips = trips[
        trips["start_station_name"].isin(stations) &
        trips["end_station_name"].isin(stations)
    ].copy()
    
    # Process in hourly batches for speed
    valid_trips["hour"] = valid_trips["started_at"].dt.floor("h")
    hours = sorted(valid_trips["hour"].unique())
    
    for hour in tqdm(hours, desc="Forward pass (hourly batches)"):
        hour_trips = valid_trips[valid_trips["hour"] == hour]
        
        # Count arrivals and departures for this hour
        arrivals = hour_trips.groupby("end_station_name").size()
        departures = hour_trips.groupby("start_station_name").size()
        
        # Apply arrivals (always works)
        for station, count in arrivals.items():
            inventory[station] += count
        
        # Apply departures (clamp at 0)
        for station, count in departures.items():
            inventory[station] = max(0, inventory[station] - count)
    
    return inventory


def backward_pass_fast(trips: pd.DataFrame, stations: list) -> dict:
    """
    Track inventory backward through time (FAST vectorized version).
    
    Process in reverse: trips from A→B become B→A conceptually.
    """
    inventory = {station: 0 for station in stations}
    
    # Filter to valid stations
    valid_trips = trips[
        trips["start_station_name"].isin(stations) &
        trips["end_station_name"].isin(stations)
    ].copy()
    
    # Process in hourly batches, in REVERSE order
    valid_trips["hour"] = valid_trips["started_at"].dt.floor("h")
    hours = sorted(valid_trips["hour"].unique(), reverse=True)
    
    for hour in tqdm(hours, desc="Backward pass (hourly batches)"):
        hour_trips = valid_trips[valid_trips["hour"] == hour]
        
        # In reverse: "arrivals" at origin (bike was there before leaving)
        arrivals = hour_trips.groupby("start_station_name").size()
        
        # In reverse: "departures" from destination (bike left after arriving)
        departures = hour_trips.groupby("end_station_name").size()
        
        # Apply arrivals
        for station, count in arrivals.items():
            inventory[station] += count
        
        # Apply departures (clamp at 0)
        for station, count in departures.items():
            inventory[station] = max(0, inventory[station] - count)
    
    return inventory


def process_month(month_dir: Path, stations: list) -> dict:
    """Process a single month of data."""
    print(f"\n{'='*60}")
    print(f"Processing: {month_dir.name}")
    print(f"{'='*60}")
    
    # Load trips
    trips = load_trips_for_month(month_dir)
    
    if len(trips) == 0:
        print(f"No data found in {month_dir}")
        return None
    
    print(f"Loaded {len(trips):,} trips")
    print(f"Date range: {trips['started_at'].min()} to {trips['started_at'].max()}")
    
    # Get stations that appear in this month's data
    month_stations = set(trips["start_station_name"].dropna()) | set(trips["end_station_name"].dropna())
    valid_stations = [s for s in stations if s in month_stations]
    print(f"Stations in this month: {len(valid_stations)}")
    
    # Forward pass: get end-of-month inventory
    print("\nRunning forward pass...")
    end_inventory = forward_pass_fast(trips, valid_stations)
    
    # Backward pass: get start-of-month inventory
    print("\nRunning backward pass...")
    start_inventory = backward_pass_fast(trips, valid_stations)
    
    # Compute total bikes (sanity check)
    total_start = sum(start_inventory.values())
    total_end = sum(end_inventory.values())
    
    print(f"\nTotal bikes inferred:")
    print(f"  Start of month: {total_start:,}")
    print(f"  End of month: {total_end:,}")
    print(f"  Difference: {total_end - total_start:,} (rebalancing/data issues)")
    
    # Top 10 stations by bike count
    print(f"\nTop 10 stations at START of month:")
    sorted_start = sorted(start_inventory.items(), key=lambda x: x[1], reverse=True)[:10]
    for station, count in sorted_start:
        print(f"  {station}: {count}")
    
    # Get timestamps
    start_time = trips["started_at"].min()
    end_time = trips["started_at"].max()
    
    return {
        "month": month_dir.name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "n_trips": len(trips),
        "n_stations": len(valid_stations),
        "start_inventory": start_inventory,
        "end_inventory": end_inventory,
        "total_bikes_start": total_start,
        "total_bikes_end": total_end,
    }


def main():
    print("="*60)
    print("Computing Station Inventory from Trip Data")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load station info to get list of valid stations
    station_info_path = DATA_DIR / "stations" / "station_info.csv"
    if station_info_path.exists():
        stations_df = pd.read_csv(station_info_path)
        all_stations = stations_df["name"].tolist()
        print(f"Loaded {len(all_stations)} stations from station_info.csv")
    else:
        print("Warning: station_info.csv not found, will use stations from trip data")
        all_stations = None
    
    # Find all month directories
    month_dirs = sorted([
        d for d in DATA_DIR.iterdir() 
        if d.is_dir() and d.name.startswith("2025") and "citibike" in d.name
    ])
    
    print(f"\nFound {len(month_dirs)} months of data:")
    for d in month_dirs:
        print(f"  - {d.name}")
    
    # Process each month
    all_results = {}
    
    for month_dir in month_dirs:
        # If we don't have station list, build it from first month
        if all_stations is None:
            trips = load_trips_for_month(month_dir)
            all_stations = list(
                set(trips["start_station_name"].dropna()) | 
                set(trips["end_station_name"].dropna())
            )
        
        result = process_month(month_dir, all_stations)
        
        if result:
            all_results[month_dir.name] = result
            
            # Save individual month result as JSON
            month_output = OUTPUT_DIR / f"{month_dir.name}_inventory.json"
            with open(month_output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved: {month_output}")
            
            # Save as CSV for easy loading
            start_df = pd.DataFrame({
                "station_name": list(result["start_inventory"].keys()),
                "bikes": list(result["start_inventory"].values()),
            })
            start_csv = OUTPUT_DIR / f"{month_dir.name}_start_inventory.csv"
            start_df.to_csv(start_csv, index=False)
            
            end_df = pd.DataFrame({
                "station_name": list(result["end_inventory"].keys()),
                "bikes": list(result["end_inventory"].values()),
            })
            end_csv = OUTPUT_DIR / f"{month_dir.name}_end_inventory.csv"
            end_df.to_csv(end_csv, index=False)
            
            print(f"Saved CSVs: {start_csv.name}, {end_csv.name}")
    
    # Save summary
    summary = {
        "computed_at": datetime.now().isoformat(),
        "months": list(all_results.keys()),
        "total_stations": len(all_stations) if all_stations else 0,
        "results": {
            month: {
                "n_trips": r["n_trips"],
                "n_stations": r["n_stations"],
                "total_bikes_start": r["total_bikes_start"],
                "total_bikes_end": r["total_bikes_end"],
            }
            for month, r in all_results.items()
        }
    }
    
    summary_path = OUTPUT_DIR / "inventory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
    
    print("\n" + "="*60)
    print("Done! Inventory data saved to: data/inventory/")
    print("="*60)
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
