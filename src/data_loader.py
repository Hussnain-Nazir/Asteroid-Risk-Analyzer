"""
data_loader.py
--------------
Fetches asteroid data from NASA NeoWs API, flattens nested JSON,
and saves to data/neos.csv. Also provides a load function for the app.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

NASA_API_KEY = "DEMO_KEY"  # Your NASA API key — get one free at https://api.nasa.gov/
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "neos.csv")


def fetch_neo_data(days: int = 7, start_date: str = None, save: bool = True) -> pd.DataFrame:
    """
    Fetch NEO data from NASA NeoWs for a date range.
    
    Args:
        days: Number of days to fetch (max 7 per request)
        start_date: Start date in YYYY-MM-DD format (defaults to 7 days ago)
        save: Whether to save the fetched data to CSV

    Returns:
        DataFrame with flattened NEO records
    """
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://api.nasa.gov/neo/rest/v1/feed"
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": NASA_API_KEY,
    }

    print(f"Fetching NEO data from {start_date} to {end_date}...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    records = []
    for date_str, neo_list in data["near_earth_objects"].items():
        for neo in neo_list:
            record = _flatten_neo(neo, date_str)
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Fetched {len(df)} NEO records.")

    if save:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved to {DATA_PATH}")

    return df


def _flatten_neo(neo: dict, date_str: str) -> dict:
    """Flatten a single NEO JSON object into a flat dictionary."""
    # Get first close approach data
    approach = neo.get("close_approach_data", [{}])[0]
    
    # Get diameter estimates in km
    diameter = neo.get("estimated_diameter", {}).get("kilometers", {})
    
    return {
        "id": neo.get("id"),
        "name": neo.get("name"),
        "date": date_str,
        "absolute_magnitude_h": neo.get("absolute_magnitude_h"),
        "diameter_min_km": diameter.get("estimated_diameter_min"),
        "diameter_max_km": diameter.get("estimated_diameter_max"),
        "is_potentially_hazardous_asteroid": neo.get("is_potentially_hazardous_asteroid"),
        "relative_velocity_km_s": approach.get("relative_velocity", {}).get("kilometers_per_second"),
        "miss_distance_km": approach.get("miss_distance", {}).get("kilometers"),
        "orbiting_body": approach.get("orbiting_body"),
        "orbital_eccentricity": None,  # Available via orbital data endpoint
    }


def load_data() -> pd.DataFrame:
    """
    Load NEO data from CSV. If CSV does not exist, auto-fetch from NASA API.
    
    Returns:
        DataFrame with NEO records
    """
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df
    else:
        print("CSV not found. Auto-fetching from NASA API...")
        return fetch_neo_data(days=7)


if __name__ == "__main__":
    df = fetch_neo_data(days=7)
    print(df.head())
    print(df.dtypes)
