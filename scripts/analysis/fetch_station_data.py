"""
Fetch CitiBike station information from the GBFS feed.

This script downloads station data including:
- Station name
- Station ID (short_name matches our trip data)
- Capacity (number of docks)
- Coordinates (lat, lng)

Source: https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_information.json
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Configuration
STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_information.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "stations"


def fetch_station_data():
    """Fetch station information from CitiBike GBFS feed."""
    print(f"Fetching station data from: {STATION_INFO_URL}")

    response = requests.get(STATION_INFO_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Extract metadata
    last_updated = data.get("last_updated")
    if last_updated:
        last_updated_dt = datetime.fromtimestamp(last_updated)
        print(f"Data last updated: {last_updated_dt}")

    stations = data["data"]["stations"]
    print(f"Found {len(stations)} stations")

    return stations, data


def process_stations(stations):
    """Process station data into a clean DataFrame."""

    records = []
    for station in stations:
        records.append(
            {
                "station_id": station.get("station_id"),
                "short_name": station.get("short_name"),  # This matches our trip data!
                "name": station.get("name"),
                "capacity": station.get("capacity"),
                "lat": station.get("lat"),
                "lng": station.get("lon"),  # Note: API uses "lon", we use "lng"
                "region_id": station.get("region_id"),
                "station_type": station.get("station_type"),
                "has_kiosk": station.get("has_kiosk"),
            }
        )

    df = pd.DataFrame(records)

    # Print summary
    print("\nStation Summary:")
    print(f"  Total stations: {len(df)}")
    print(f"  Capacity range: {df['capacity'].min()} - {df['capacity'].max()}")
    print(f"  Average capacity: {df['capacity'].mean():.1f}")
    print(f"  Total bike capacity: {df['capacity'].sum():,}")

    # Region breakdown
    print("\nStations by region_id:")
    print(df["region_id"].value_counts())

    return df


def save_data(df, raw_data):
    """Save processed and raw station data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed CSV
    csv_path = OUTPUT_DIR / "station_info.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved processed data to: {csv_path}")

    # Save raw JSON (for reference)
    json_path = OUTPUT_DIR / "station_info_raw.json"
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved raw JSON to: {json_path}")

    # Save a summary
    summary = {
        "fetch_timestamp": datetime.now().isoformat(),
        "source_url": STATION_INFO_URL,
        "total_stations": len(df),
        "total_capacity": int(df["capacity"].sum()),
        "avg_capacity": float(df["capacity"].mean()),
        "capacity_range": [int(df["capacity"].min()), int(df["capacity"].max())],
    }

    summary_path = OUTPUT_DIR / "fetch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")


def main():
    print("=" * 60)
    print("CitiBike Station Data Fetcher")
    print("=" * 60)

    # Fetch data
    stations, raw_data = fetch_station_data()

    # Process into DataFrame
    df = process_stations(stations)

    # Save outputs
    save_data(df, raw_data)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
