"""
preprocessing.py
----------------
Handles data cleaning, type conversion, feature engineering,
and returns a clean feature matrix ready for ML training.
"""

import pandas as pd
import numpy as np
from typing import Tuple


FEATURE_COLS = [
    "absolute_magnitude_h",
    "diameter_min_km",
    "diameter_max_km",
    "diameter_mean_km",
    "relative_velocity_km_s",
    "miss_distance_km",
    "orbital_eccentricity",
]

TARGET_COL = "is_potentially_hazardous_asteroid"


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Convert types
    2. Drop rows with critical missing values
    3. Engineer derived features
    4. Select relevant columns

    Args:
        df: Raw DataFrame from data_loader

    Returns:
        Cleaned DataFrame with features and target
    """
    df = df.copy()

    # Convert numeric columns
    numeric_cols = [
        "absolute_magnitude_h",
        "diameter_min_km",
        "diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_km",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert orbital eccentricity (may be string or None)
    if "orbital_eccentricity" in df.columns:
        df["orbital_eccentricity"] = pd.to_numeric(df["orbital_eccentricity"], errors="coerce")

    # Engineer derived feature: mean diameter
    if "diameter_min_km" in df.columns and "diameter_max_km" in df.columns:
        df["diameter_mean_km"] = (df["diameter_min_km"] + df["diameter_max_km"]) / 2

    # Convert target to int (True/False → 1/0)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(bool).astype(int)

    # Drop rows missing target or key features
    required_cols = ["absolute_magnitude_h", "diameter_min_km", "relative_velocity_km_s", "miss_distance_km", TARGET_COL]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Fill missing orbital_eccentricity with median (it's NaN for API data)
    if "orbital_eccentricity" in df.columns:
        median_ecc = df["orbital_eccentricity"].median()
        if pd.isna(median_ecc):
            median_ecc = 0.5  # fallback
        df["orbital_eccentricity"] = df["orbital_eccentricity"].fillna(median_ecc)

    # Keep only relevant columns
    keep_cols = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    # Optionally keep name/id/date for display
    meta_cols = [c for c in ["id", "name", "date"] if c in df.columns]
    df = df[meta_cols + keep_cols]

    return df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split cleaned DataFrame into features (X) and target (y).

    Args:
        df: Cleaned DataFrame from clean_and_prepare()

    Returns:
        Tuple of (X, y)
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[TARGET_COL]
    return X, y


def get_feature_names(df: pd.DataFrame) -> list:
    """Return list of available feature column names."""
    return [c for c in FEATURE_COLS if c in df.columns]
