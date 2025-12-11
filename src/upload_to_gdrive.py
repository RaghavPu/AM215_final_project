"""
Upload Parquet files to Google Drive.

This script uploads the converted Parquet files to Google Drive
for storage and sharing.

Setup:
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials.json and place in project root
"""

import json
import pickle
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def authenticate_gdrive():
    """
    Authenticate with Google Drive API.

    Returns:
        Google Drive service object
    """
    creds = None
    token_path = Path("token.pickle")
    credentials_path = Path("credentials.json")

    # The file token.pickle stores the user's access and refresh tokens
    if token_path.exists():
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                print("ERROR: credentials.json not found!")
                print("\nPlease follow these steps:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project")
                print("3. Enable Google Drive API")
                print("4. Create OAuth 2.0 credentials (Desktop app)")
                print("5. Download credentials.json to project root")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    service = build("drive", "v3", credentials=creds)
    return service


def create_folder(service, folder_name, parent_id=None):
    """
    Create a folder in Google Drive.

    Args:
        service: Google Drive service object
        folder_name: Name of folder to create
        parent_id: Parent folder ID (None for root)

    Returns:
        Folder ID
    """
    file_metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}

    if parent_id:
        file_metadata["parents"] = [parent_id]

    folder = service.files().create(body=file_metadata, fields="id").execute()
    print(f"Created folder: {folder_name} (ID: {folder.get('id')})")
    return folder.get("id")


def upload_file(service, file_path, parent_id=None):
    """
    Upload a file to Google Drive.

    Args:
        service: Google Drive service object
        file_path: Path to file to upload
        parent_id: Parent folder ID (None for root)

    Returns:
        File ID
    """
    file_metadata = {"name": file_path.name}

    if parent_id:
        file_metadata["parents"] = [parent_id]

    media = MediaFileUpload(str(file_path), resumable=True)

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id,name,size")
        .execute()
    )

    size_mb = int(file.get("size", 0)) / 1024 / 1024
    print(f"  ✓ Uploaded: {file.get('name')} ({size_mb:.1f} MB)")

    return file.get("id")


def upload_parquet_files(service, project_root):
    """
    Upload all Parquet files to Google Drive.

    Args:
        service: Google Drive service object
        project_root: Project root directory

    Returns:
        Dictionary with upload stats
    """
    parquet_dir = project_root / "data" / "parquet" / "trips"

    if not parquet_dir.exists():
        print(f"ERROR: Parquet directory not found: {parquet_dir}")
        return None

    # Create main folder
    print("\n" + "=" * 60)
    print("Creating folder structure in Google Drive...")
    print("=" * 60)

    main_folder_id = create_folder(service, "CitiBike_Data")

    # Find all parquet files
    parquet_files = list(parquet_dir.rglob("*.parquet"))

    if not parquet_files:
        print(f"ERROR: No Parquet files found in {parquet_dir}")
        return None

    print(f"\nFound {len(parquet_files)} Parquet files to upload\n")

    # Upload files with folder structure
    uploaded = []
    year_folders = {}

    for parquet_file in sorted(parquet_files):
        # Extract year and month from path
        # Path structure: data/parquet/trips/year=2024/month=01/trips.parquet
        parts = parquet_file.parts
        year_part = [p for p in parts if p.startswith("year=")]
        month_part = [p for p in parts if p.startswith("month=")]

        if year_part and month_part:
            year = year_part[0].split("=")[1]
            month = month_part[0].split("=")[1]

            # Create year folder if needed
            if year not in year_folders:
                year_folders[year] = create_folder(service, f"year_{year}", main_folder_id)

            # Create month folder
            month_folder_id = create_folder(service, f"month_{month}", year_folders[year])

            # Upload file
            print(f"Uploading {year}-{month}...")
            file_id = upload_file(service, parquet_file, month_folder_id)

            uploaded.append(
                {
                    "year": year,
                    "month": month,
                    "file_id": file_id,
                    "file_name": parquet_file.name,
                    "size_bytes": parquet_file.stat().st_size,
                }
            )

    # Upload station metadata
    print("\nUploading station metadata...")
    station_file = project_root / "data" / "parquet" / "stations" / "station_info.parquet"
    if station_file.exists():
        metadata_folder_id = create_folder(service, "metadata", main_folder_id)
        upload_file(service, station_file, metadata_folder_id)

    return {
        "main_folder_id": main_folder_id,
        "uploaded_files": uploaded,
        "total_files": len(uploaded),
        "total_size_mb": sum(f["size_bytes"] for f in uploaded) / 1024 / 1024,
    }


def main():
    """Upload Parquet files to Google Drive."""
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("CitiBike Data Upload to Google Drive")
    print("=" * 70)
    print()

    # Authenticate
    print("Authenticating with Google Drive...")
    service = authenticate_gdrive()

    if not service:
        print("\nAuthentication failed. Exiting.")
        return

    print("✓ Authentication successful\n")

    # Upload files
    stats = upload_parquet_files(service, project_root)

    if not stats:
        print("\nUpload failed. Exiting.")
        return

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD COMPLETE!")
    print("=" * 70)
    print(f"Total files uploaded: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB ({stats['total_size_mb'] / 1024:.2f} GB)")
    print(f"\nMain folder ID: {stats['main_folder_id']}")
    print(f"View in Drive: https://drive.google.com/drive/folders/{stats['main_folder_id']}")

    # Save upload stats
    stats_path = project_root / "data" / "gdrive_upload_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Upload stats saved to: {stats_path}")

    print("\n" + "=" * 70)
    print("IMPORTANT: Share the folder")
    print("=" * 70)
    print("To share with others:")
    print("1. Go to https://drive.google.com/")
    print("2. Find 'CitiBike_Data' folder")
    print("3. Right-click → Share → Add people/groups")
    print("4. Or: Right-click → Share → Get link → 'Anyone with the link'")


if __name__ == "__main__":
    main()
