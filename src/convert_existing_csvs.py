"""
Convert existing CSV files to Parquet format.

This script looks for CSV files you've already downloaded and converts them to Parquet.
You can manually download files from: https://s3.amazonaws.com/tripdata/index.html
"""

import json
import re
import time
from pathlib import Path

import duckdb


def find_csv_files(search_paths):
    """
    Find all CitiBike CSV files in given paths.

    Args:
        search_paths: List of paths to search

    Returns:
        List of CSV file paths with metadata
    """
    csv_files = []

    for search_path in search_paths:
        path = Path(search_path).expanduser()

        if not path.exists():
            print(f"Path does not exist: {path}")
            continue

        # Find CSV files matching CitiBike pattern
        for csv_file in path.rglob("*citibike*.csv"):
            # Extract year/month from filename
            match = re.search(r"(\d{4})(\d{2})", csv_file.name)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                csv_files.append(
                    {"path": csv_file, "year": year, "month": month, "name": csv_file.name}
                )

    return sorted(csv_files, key=lambda x: (x["year"], x["month"]))


def convert_csv_to_parquet(csv_path, parquet_path, year, month):
    """
    Convert a single CSV to Parquet.

    Args:
        csv_path: Path to CSV file
        parquet_path: Output Parquet path
        year: Year of data
        month: Month of data

    Returns:
        Conversion statistics
    """
    try:
        start_time = time.time()

        print(f"  Reading CSV: {csv_path.name}")
        con = duckdb.connect()

        # Create output directory
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert with compression
        # Force station IDs to be strings to handle mixed types (numeric and "JC" prefixes)
        query = f"""
            COPY (
                SELECT
                    *,
                    {year} as year,
                    {month} as month
                FROM read_csv_auto('{csv_path}', types={{'start_station_id': 'VARCHAR', 'end_station_id': 'VARCHAR'}})
            ) TO '{parquet_path}' (
                FORMAT PARQUET,
                COMPRESSION 'ZSTD',
                ROW_GROUP_SIZE 100000
            );
        """

        con.execute(query)

        # Get row count
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')").fetchone()[0]

        con.close()

        elapsed = time.time() - start_time
        csv_size = csv_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0

        print(f"  ✓ Converted {row_count:,} rows in {elapsed:.1f}s")
        print(
            f"  ✓ {csv_size / 1024 / 1024:.1f} MB → {parquet_size / 1024 / 1024:.1f} MB ({compression_ratio:.1f}x compression)"
        )

        return {
            "success": True,
            "row_count": row_count,
            "csv_size_mb": csv_size / 1024 / 1024,
            "parquet_size_mb": parquet_size / 1024 / 1024,
            "compression_ratio": compression_ratio,
            "time_seconds": elapsed,
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"success": False, "error": str(e)}


def main(search_paths=None, months_to_convert=None):
    """
    Convert existing CSV files to Parquet.

    Args:
        search_paths: List of directories to search for CSV files
        months_to_convert: Number of months to convert (None = all found)
    """
    project_root = Path(__file__).parent.parent

    # Default search paths
    if search_paths is None:
        search_paths = [Path.home() / "Downloads", project_root / "data" / "raw"]

    print("=" * 70)
    print("CitiBike CSV to Parquet Converter")
    print("=" * 70)
    print("\nSearching for CSV files in:")
    for path in search_paths:
        print(f"  - {path}")
    print()

    # Find CSV files
    csv_files = find_csv_files(search_paths)

    if not csv_files:
        print("No CitiBike CSV files found!")
        print("\nTo download files manually:")
        print("1. Go to: https://s3.amazonaws.com/tripdata/index.html")
        print("2. Download desired months (e.g., 202501-citibike-tripdata.csv.zip)")
        print("3. Unzip and place CSV files in ~/Downloads or data/raw/")
        print("4. Run this script again")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f['year']}-{f['month']:02d}: {f['name']}")
    print()

    # Limit number of months if specified
    if months_to_convert:
        csv_files = csv_files[:months_to_convert]
        print(f"Converting first {len(csv_files)} month(s)...\n")

    # Convert each file
    all_stats = []

    for file_info in csv_files:
        print("=" * 70)
        print(f"Processing: {file_info['year']}-{file_info['month']:02d}")
        print("=" * 70)

        parquet_path = (
            project_root
            / "data"
            / "parquet"
            / "trips"
            / f"year={file_info['year']}"
            / f"month={file_info['month']:02d}"
            / "trips.parquet"
        )

        stats = convert_csv_to_parquet(
            file_info["path"], parquet_path, file_info["year"], file_info["month"]
        )

        stats.update(
            {
                "year": file_info["year"],
                "month": file_info["month"],
                "source_file": str(file_info["path"]),
            }
        )

        all_stats.append(stats)
        print()

    # Summary
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    successful = [s for s in all_stats if s.get("success")]
    failed = [s for s in all_stats if not s.get("success")]

    print(f"Successful: {len(successful)}/{len(all_stats)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_rows = sum(s["row_count"] for s in successful)
        total_parquet_mb = sum(s["parquet_size_mb"] for s in successful)
        avg_compression = sum(s["compression_ratio"] for s in successful) / len(successful)

        print(f"\nTotal rows: {total_rows:,}")
        print(f"Total Parquet size: {total_parquet_mb:.1f} MB ({total_parquet_mb / 1024:.2f} GB)")
        print(f"Average compression: {avg_compression:.1f}x")

    if failed:
        print("\nFailed conversions:")
        for s in failed:
            print(f"  - {s['year']}-{s['month']:02d}: {s.get('error', 'unknown')}")

    # Save stats
    stats_path = project_root / "data" / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    print(f"\n✓ Stats saved to: {stats_path}")
    print(f"\n✓ Parquet files saved to: {project_root / 'data' / 'parquet' / 'trips'}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Convert all files found (change to a number to limit)
    main(months_to_convert=None)
