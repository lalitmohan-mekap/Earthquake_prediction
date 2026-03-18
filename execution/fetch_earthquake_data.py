"""
fetch_earthquake_data.py
========================
Download earthquake catalog from the USGS ComCat API (CSV format).
Queries year-by-year to stay within the 20,000 row limit per request.

Input:  None (queries USGS API)
Output: .tmp/raw_earthquakes.csv
"""

import os
import sys
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
START_YEAR = 2000
END_YEAR = 2025
MIN_MAGNITUDE = 2.5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", ".tmp")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_earthquakes.csv")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles on each retry


def fetch_year(year: int) -> pd.DataFrame:
    """Fetch earthquake data for a single year from USGS."""
    params = {
        "format": "csv",
        "starttime": f"{year}-01-01",
        "endtime": f"{year}-12-31",
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time",
        "limit": 20000,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Fetching {year} (attempt {attempt})...", end=" ")
            resp = requests.get(BASE_URL, params=params, timeout=60)
            resp.raise_for_status()

            # Parse CSV from response text
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            print(f"{len(df)} events")
            return df

        except Exception as e:
            print(f"FAILED: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Skipping {year} after {MAX_RETRIES} failures.")
                return pd.DataFrame()


def main():
    """Main entry point — fetch all years and concatenate."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching USGS earthquake data ({START_YEAR}–{END_YEAR}, M≥{MIN_MAGNITUDE})")
    print("=" * 60)

    all_frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        df = fetch_year(year)
        if not df.empty:
            all_frames.append(df)
        time.sleep(1)  # Be respectful to the API

    if not all_frames:
        print("ERROR: No data fetched. Check network / API status.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.drop_duplicates(subset=["time", "latitude", "longitude", "mag"], inplace=True)
    combined.sort_values("time", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT_FILE, index=False)
    print("=" * 60)
    print(f"Done! Saved {len(combined)} events to {OUTPUT_FILE}")

    if len(combined) < 1000:
        print("WARNING: Dataset is unusually small (<1000 events). Verify data integrity.")


if __name__ == "__main__":
    main()
