"""
Download CitiBike trip data from S3 and convert to Parquet format.

This script:
1. Downloads all 2024-2025 trip data (22 months)
2. Unzips CSV files
3. Converts to Parquet with compression
4. Cleans up original files to save space
"""

import json
import time
import zipfile
from pathlib import Path

import duckdb
import requests


def download_file(url, output_path, chunk_size=8192):
    """
    Download a file with progress tracking.

    Args:
        url: URL to download from
        output_path: Path to save file
        chunk_size: Download chunk size in bytes

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\r  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)",
                            end="",
                        )

        print()  # New line after progress
        return True

    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def unzip_file(zip_path, extract_to):
    """
    Unzip a file.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to

    Returns:
        Path to extracted CSV file
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # Find the CSV file
        csv_files = list(extract_to.glob("*.csv"))
        if csv_files:
            return csv_files[0]
        else:
            raise Exception("No CSV file found in zip")

    except Exception as e:
        print(f"  Error unzipping: {e}")
        return None


def convert_to_parquet(csv_path, parquet_path, year, month):
    """
    Convert CSV to Parquet using DuckDB.

    Args:
        csv_path: Path to CSV file
        parquet_path: Path to output Parquet file
        year: Year of the data
        month: Month of the data

    Returns:
        Dictionary with conversion stats
    """
    try:
        start_time = time.time()

        # Create connection
        con = duckdb.connect()

        # Read CSV and write to Parquet with partitioning
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        query = f"""
            COPY (
                SELECT
                    *,
                    {year} as year,
                    {month} as month
                FROM read_csv_auto('{csv_path}')
            ) TO '{parquet_path}' (
                FORMAT PARQUET,
                COMPRESSION 'ZSTD',
                ROW_GROUP_SIZE 100000
            );
        """

        con.execute(query)

        # Get stats
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')").fetchone()[0]
        con.close()

        elapsed = time.time() - start_time

        csv_size = csv_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        compression_ratio = csv_size / parquet_size if parquet_size > 0 else 0

        return {
            "success": True,
            "row_count": row_count,
            "csv_size_mb": csv_size / 1024 / 1024,
            "parquet_size_mb": parquet_size / 1024 / 1024,
            "compression_ratio": compression_ratio,
            "time_seconds": elapsed,
        }

    except Exception as e:
        print(f"  Error converting: {e}")
        return {"success": False, "error": str(e)}


def process_month(year, month, base_url, project_root, keep_intermediates=False):
    """
    Download and convert one month of data.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        base_url: S3 base URL
        project_root: Project root directory
        keep_intermediates: If True, keep zip and CSV files

    Returns:
        Dictionary with processing stats
    """
    filename = f"{year}{month:02d}-citibike-tripdata.csv.zip"
    url = f"{base_url}/{filename}"

    # Paths
    zip_path = project_root / "data" / "raw" / "downloads" / filename
    extract_dir = project_root / "data" / "raw" / "extracted" / f"{year}{month:02d}"
    parquet_path = (
        project_root
        / "data"
        / "parquet"
        / "trips"
        / f"year={year}"
        / f"month={month:02d}"
        / "trips.parquet"
    )

    print(f"\n{'=' * 60}")
    print(f"Processing: {year}-{month:02d}")
    print(f"{'=' * 60}")

    stats = {"year": year, "month": month, "filename": filename}

    # Step 1: Download
    print("[1/4] Downloading from S3...")
    if not zip_path.exists():
        if not download_file(url, zip_path):
            stats["status"] = "download_failed"
            return stats
    else:
        print(f"  Already downloaded: {zip_path}")

    # Step 2: Unzip
    print("[2/4] Extracting CSV...")
    csv_path = unzip_file(zip_path, extract_dir)
    if not csv_path:
        stats["status"] = "unzip_failed"
        return stats
    print(f"  Extracted: {csv_path.name}")

    # Step 3: Convert to Parquet
    print("[3/4] Converting to Parquet...")
    conversion_stats = convert_to_parquet(csv_path, parquet_path, year, month)
    stats.update(conversion_stats)

    if not conversion_stats["success"]:
        stats["status"] = "conversion_failed"
        return stats

    print(f"  ✓ Converted {conversion_stats['row_count']:,} rows")
    print(
        f"  ✓ CSV: {conversion_stats['csv_size_mb']:.1f} MB → Parquet: {conversion_stats['parquet_size_mb']:.1f} MB"
    )
    print(f"  ✓ Compression: {conversion_stats['compression_ratio']:.1f}x")
    print(f"  ✓ Time: {conversion_stats['time_seconds']:.1f}s")

    # Step 4: Cleanup
    if not keep_intermediates:
        print("[4/4] Cleaning up...")
        try:
            csv_path.unlink()
            zip_path.unlink()
            # Remove empty directories
            try:
                extract_dir.rmdir()
            except:
                pass
            print("  ✓ Removed intermediate files")
        except Exception as e:
            print(f"  Warning: Could not remove files: {e}")
    else:
        print("[4/4] Keeping intermediate files")

    stats["status"] = "success"
    return stats


def main(years=[2024, 2025], keep_intermediates=False):
    """
    Download and convert all CitiBike data.

    Args:
        years: List of years to process
        keep_intermediates: If True, keep zip and CSV files
    """
    project_root = Path(__file__).parent.parent
    base_url = "https://s3.amazonaws.com/tripdata"

    print("=" * 70)
    print("CitiBike Data Download & Parquet Conversion")
    print("=" * 70)
    print(f"Years: {years}")
    print(f"Keep intermediates: {keep_intermediates}")
    print()

    # Generate list of months to process
    months_to_process = []
    for year in years:
        if year == 2024:
            month_range = range(1, 13)  # Jan-Dec
        elif year == 2025:
            month_range = range(1, 11)  # Jan-Oct
        else:
            month_range = range(1, 13)

        for month in month_range:
            months_to_process.append((year, month))

    print(f"Total months to process: {len(months_to_process)}\n")

    # Process each month
    all_stats = []
    start_time = time.time()

    for year, month in months_to_process:
        stats = process_month(year, month, base_url, project_root, keep_intermediates)
        all_stats.append(stats)

        # Brief pause between downloads to be respectful to S3
        time.sleep(1)

    # Summary
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [s for s in all_stats if s.get("status") == "success"]
    failed = [s for s in all_stats if s.get("status") != "success"]

    print(f"Successful: {len(successful)}/{len(all_stats)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_rows = sum(s.get("row_count", 0) for s in successful)
        total_parquet_size = sum(s.get("parquet_size_mb", 0) for s in successful)
        avg_compression = sum(s.get("compression_ratio", 0) for s in successful) / len(successful)

        print(f"\nTotal rows: {total_rows:,}")
        print(
            f"Total Parquet size: {total_parquet_size:.1f} MB ({total_parquet_size / 1024:.2f} GB)"
        )
        print(f"Average compression ratio: {avg_compression:.1f}x")
        print(f"Total time: {total_time / 60:.1f} minutes")

    if failed:
        print("\nFailed months:")
        for s in failed:
            print(f"  - {s['year']}-{s['month']:02d}: {s.get('status', 'unknown')}")

    # Save stats
    stats_path = project_root / "data" / "conversion_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Stats saved to: {stats_path}")

    print("\n" + "=" * 70)
    print("✓ CONVERSION COMPLETE!")
    print("=" * 70)
    print("\nYour Parquet files are in: data/parquet/trips/")
    print("You can now use DuckDB to query them efficiently!")
    print("\nNext steps:")
    print("  1. Verify data: python src/verify_data.py")
    print("  2. Start analysis: jupyter notebook notebooks/eda.ipynb")


if __name__ == "__main__":
    # Set keep_intermediates=True if you want to keep CSV/zip files
    main(years=[2024, 2025], keep_intermediates=False)
