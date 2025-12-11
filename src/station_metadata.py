"""
Download and process CitiBike station metadata from GBFS API.

This script fetches station information (capacity, location, etc.)
from the General Bikeshare Feed Specification (GBFS) API.
"""

from pathlib import Path

import pandas as pd
import requests


def fetch_gbfs_station_info(region="bkn", language="en"):
    """
    Fetch station information from GBFS API.

    Args:
        region: Region code (e.g., 'bkn' for Brooklyn)
        language: Language code ('en', 'fr', 'es')

    Returns:
        DataFrame with station metadata
    """
    url = f"https://gbfs.lyft.com/gbfs/2.3/{region}/{language}/station_information.json"

    print(f"Fetching station data from: {url}")
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    stations = data["data"]["stations"]

    # Convert to DataFrame
    df = pd.DataFrame(stations)

    print(f"Found {len(df)} stations")
    print(f"Columns: {df.columns.tolist()}")

    return df


def fetch_all_regions():
    """
    Fetch station data from all available regions.
    CitiBike operates across multiple regions in NYC area.
    """
    # Common region codes - we'll try these
    regions = ["bkn", "man", "que", "jc", "nyc"]

    all_stations = []

    for region in regions:
        try:
            df = fetch_gbfs_station_info(region=region)
            df["region"] = region
            all_stations.append(df)
            print(f"✓ Successfully fetched {len(df)} stations from {region}")
        except Exception as e:
            print(f"✗ Failed to fetch {region}: {e}")

    if all_stations:
        combined = pd.concat(all_stations, ignore_index=True)
        # Remove duplicates based on station_id
        combined = combined.drop_duplicates(subset="station_id", keep="first")
        print(f"\nTotal unique stations: {len(combined)}")
        return combined
    else:
        raise Exception("Failed to fetch data from any region")


def save_station_metadata(df, output_path):
    """
    Save station metadata to parquet file.

    Args:
        df: DataFrame with station data
        output_path: Path to save parquet file
    """
    # Select key columns
    columns_to_keep = ["station_id", "name", "short_name", "lat", "lon", "capacity", "region_id"]

    # Only keep columns that exist
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df_subset = df[columns_to_keep].copy()

    # Ensure proper data types
    df_subset["station_id"] = df_subset["station_id"].astype(str)
    df_subset["capacity"] = pd.to_numeric(df_subset["capacity"], errors="coerce")
    df_subset["lat"] = pd.to_numeric(df_subset["lat"], errors="coerce")
    df_subset["lon"] = pd.to_numeric(df_subset["lon"], errors="coerce")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_parquet(output_path, index=False, compression="snappy")

    print(f"\n✓ Saved {len(df_subset)} stations to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Display summary statistics
    print("\nStation Metadata Summary:")
    print(
        f"  Capacity range: {df_subset['capacity'].min():.0f} - {df_subset['capacity'].max():.0f}"
    )
    print(f"  Average capacity: {df_subset['capacity'].mean():.1f}")
    print(f"  Missing capacity values: {df_subset['capacity'].isna().sum()}")

    return df_subset


def main():
    """Download and save station metadata."""
    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "parquet" / "stations" / "station_info.parquet"

    print("=== CitiBike Station Metadata Download ===\n")

    # Try fetching from multiple regions
    df = fetch_all_regions()

    # Save to parquet
    df_saved = save_station_metadata(df, output_path)

    # Show sample
    print("\nSample stations:")
    print(df_saved.head(10).to_string())

    return df_saved


if __name__ == "__main__":
    main()
