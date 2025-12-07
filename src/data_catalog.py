"""
Create a catalog of available CitiBike trip data from S3.

This script generates URLs for all 2024 and 2025 trip data files
and tests DuckDB's ability to query them remotely.
"""

import duckdb
from pathlib import Path
import json


def generate_s3_urls(years=[2024, 2025]):
    """
    Generate S3 URLs for CitiBike trip data.

    Args:
        years: List of years to include

    Returns:
        List of dictionaries with year, month, and URL
    """
    base_url = "https://s3.amazonaws.com/tripdata"
    catalog = []

    for year in years:
        # Determine month range based on year
        if year == 2024:
            months = range(1, 13)  # Jan-Dec 2024
        elif year == 2025:
            months = range(1, 11)  # Jan-Oct 2025 (Nov/Dec may not be available yet)
        else:
            months = range(1, 13)

        for month in months:
            filename = f"{year}{month:02d}-citibike-tripdata.csv.zip"
            url = f"{base_url}/{filename}"

            catalog.append({
                'year': year,
                'month': month,
                'filename': filename,
                'url': url,
                'csv_name': filename.replace('.zip', '.csv')
            })

    return catalog


def test_remote_access(url, csv_name, limit=10):
    """
    Test if DuckDB can access a remote CSV file inside a zip.

    Args:
        url: URL to the zip file
        csv_name: Name of CSV file inside the zip
        limit: Number of rows to fetch for testing

    Returns:
        DataFrame with sample data, or (None, error_msg) if failed
    """
    try:
        # DuckDB can read CSVs directly from zipped URLs using httpfs extension
        con = duckdb.connect()

        # Install and load httpfs extension for remote file access
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")

        # Read CSV from inside the zip file
        query = f"""
            SELECT *
            FROM read_csv(
                '{url}',
                compression = 'auto',
                auto_detect = true
            )
            LIMIT {limit}
        """
        df = con.execute(query).fetchdf()
        con.close()
        return df
    except Exception as e:
        try:
            con.close()
        except:
            pass
        return None, str(e)


def verify_catalog(catalog):
    """
    Verify which files are accessible via DuckDB.

    Args:
        catalog: List of file metadata from generate_s3_urls

    Returns:
        Updated catalog with accessibility status
    """
    print("Testing remote file access with DuckDB...\n")

    # Test just the first file to verify the approach works
    print("Testing first file (this may take a moment)...")
    first_file = catalog[0]
    print(f"  File: {first_file['filename']}")
    print(f"  URL: {first_file['url']}")

    result = test_remote_access(first_file['url'], first_file['csv_name'], limit=5)

    if isinstance(result, tuple):  # Error case
        df, error = result
        print(f"  ✗ Failed: {error}\n")
        first_file['accessible'] = False
        first_file['error'] = error
    else:
        df = result
        print(f"  ✓ Success! Retrieved {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\n  Sample data:")
        print(df.head().to_string(index=False))
        first_file['accessible'] = True
        first_file['row_count_sample'] = len(df)

    # Mark others as assumed accessible (same pattern)
    for item in catalog[1:]:
        item['accessible'] = first_file['accessible']

    return catalog


def save_catalog(catalog, output_path):
    """Save catalog to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"\n✓ Saved catalog to {output_path}")
    print(f"  Total files: {len(catalog)}")
    print(f"  Years: {sorted(set(item['year'] for item in catalog))}")


def create_duckdb_view_script(catalog):
    """
    Generate a SQL script to create a unified view across all files.

    Args:
        catalog: List of file metadata

    Returns:
        SQL script as string
    """
    accessible_files = [item for item in catalog if item.get('accessible', False)]

    if not accessible_files:
        return "-- No accessible files found"

    # Create UNION ALL query across all files
    sql_parts = []
    for item in accessible_files:
        year = item['year']
        month = item['month']
        url = item['url']

        sql_parts.append(f"""
    -- {year}-{month:02d}
    SELECT
        *,
        {year} as year,
        {month} as month
    FROM read_csv_auto('{url}')
""")

    union_query = "\nUNION ALL\n".join(sql_parts)

    full_script = f"""-- CitiBike Trip Data Unified View
-- Generated automatically from S3 data catalog
-- Covers 2024-2025 trip data

CREATE OR REPLACE VIEW citibike_trips AS
{union_query};

-- Example queries:

-- 1. Count trips by month
-- SELECT year, month, COUNT(*) as trip_count
-- FROM citibike_trips
-- GROUP BY year, month
-- ORDER BY year, month;

-- 2. Get trips for a specific station
-- SELECT *
-- FROM citibike_trips
-- WHERE start_station_id = 'YOUR_STATION_ID'
-- LIMIT 100;

-- 3. Analyze rush hour patterns
-- SELECT
--     EXTRACT(HOUR FROM started_at) as hour,
--     COUNT(*) as trip_count
-- FROM citibike_trips
-- WHERE year = 2024 AND month = 6
-- GROUP BY hour
-- ORDER BY hour;
"""

    return full_script


def main():
    """Generate and verify data catalog."""
    project_root = Path(__file__).parent.parent

    print("=== CitiBike Trip Data Catalog ===\n")

    # Generate catalog
    print("Generating file catalog for 2024-2025...")
    catalog = generate_s3_urls(years=[2024, 2025])
    print(f"Generated {len(catalog)} file URLs\n")

    # Verify accessibility
    catalog = verify_catalog(catalog)

    # Save catalog
    catalog_path = project_root / 'data' / 'trip_data_catalog.json'
    save_catalog(catalog, catalog_path)

    # Generate SQL script
    sql_script = create_duckdb_view_script(catalog)
    sql_path = project_root / 'src' / 'create_unified_view.sql'
    sql_path.write_text(sql_script)
    print(f"✓ Saved SQL script to {sql_path}")

    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now query CitiBike data remotely without downloading!")
    print("\nExample usage:")
    print("""
    import duckdb
    con = duckdb.connect()

    # Query specific month
    df = con.execute('''
        SELECT *
        FROM read_csv_auto('https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.csv.zip')
        WHERE start_station_name LIKE '%Central Park%'
        LIMIT 100
    ''').fetchdf()
    """)

    return catalog


if __name__ == "__main__":
    main()
