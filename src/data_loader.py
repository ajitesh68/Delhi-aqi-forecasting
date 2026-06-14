"""
Data loading and validation utilities.
"""
import os
import pandas as pd


def load_city_day(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the city_day.csv dataset and perform basic validation.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing raw CSV files.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with Date column parsed as datetime.
    """
    filepath = os.path.join(data_dir, "city_day.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"city_day.csv not found at '{filepath}'. "
            "Download from: https://www.kaggle.com/rohanrao/air-quality-data-in-india"
        )

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Cities: {df['City'].nunique()}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def load_extended_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the extended 2015-2023 AQI dataset (if available).

    Parameters
    ----------
    data_dir : str
        Path to the directory containing raw CSV files.

    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame, or None if file not found.
    """
    filepath = os.path.join(data_dir, "india_city_aqi_2015_2023.csv")
    if not os.path.exists(filepath):
        print(f"Extended dataset not found at '{filepath}'. Skipping.")
        return None

    df = pd.read_csv(filepath)
    print(f"Extended dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of missing values per column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Column, Missing_Count, Missing_Percent
    """
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100

    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values,
        "Missing_Percent": pct.values.round(2),
    })

    return summary[summary["Missing_Count"] > 0].sort_values(
        "Missing_Percent", ascending=False
    ).reset_index(drop=True)
