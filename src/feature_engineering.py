"""
Feature engineering utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Year and Month from the Date column.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    return df


def get_season(month: int) -> str:
    """
    Map month number to Indian season.

    Returns
    -------
    str
        One of: 'Winter', 'Spring', 'Monsoon', 'Autumn'
    """
    if month <= 2 or month == 12:
        return "Winter"
    elif 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 9:
        return "Monsoon"
    else:
        return "Autumn"


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Season' column based on the Month column.
    """
    df = df.copy()
    if "Month" not in df.columns:
        df = add_date_features(df)
    df["Season"] = df["Month"].apply(get_season)
    return df


def compute_composite_score(
    df: pd.DataFrame,
    weights: dict = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute a weighted Composite Pollution Score.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with pollutant columns.
    weights : dict, optional
        Pollutant weights. Defaults to domain-informed weights.
    normalize : bool
        Whether to MinMax-normalize pollutants before weighting.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Composite_Score' column.
    """
    df = df.copy()

    default_weights = {
        "PM2.5": 0.35,
        "PM10": 0.20,
        "NO2": 0.15,
        "CO": 0.10,
        "SO2": 0.10,
        "O3": 0.10,
    }
    weights = weights or default_weights

    cols = list(weights.keys())
    existing_cols = [c for c in cols if c in df.columns]

    if normalize:
        scaler = MinMaxScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])

    df["Composite_Score"] = sum(
        df[col] * weights[col] for col in existing_cols
    )

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling:
    - Season: LabelEncoder
    - City: One-hot encoding
    - AQI_Bucket: Ordinal encoding

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded features.
    """
    df = df.copy()

    # Season encoding
    le = LabelEncoder()
    df["Season_Encoded"] = le.fit_transform(df["Season"])

    # City one-hot encoding
    city_dummies = pd.get_dummies(df["City"], prefix="City")
    df = pd.concat([df, city_dummies], axis=1)

    # AQI Bucket ordinal encoding
    bucket_order = {
        "Good": 0,
        "Satisfactory": 1,
        "Moderate": 2,
        "Poor": 3,
        "Very Poor": 4,
        "Severe": 5,
    }
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket_Encoded"] = df["AQI_Bucket"].map(bucket_order)

    return df


def prepare_regression_data(df: pd.DataFrame):
    """
    Prepare X, y for AQI regression.

    Returns
    -------
    tuple
        (X, y) where X is feature matrix and y is AQI values.
    """
    drop_cols = ["AQI", "AQI_Bucket", "City", "Season", "Date", "Month"]
    if "AQI_Bucket_Encoded" in df.columns:
        drop_cols.append("AQI_Bucket_Encoded")

    existing_drops = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drops)
    y = df["AQI"]

    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepare X, y for AQI bucket classification.

    Returns
    -------
    tuple
        (X, y) where X is feature matrix and y is encoded AQI buckets.
    """
    drop_cols = [
        "AQI", "AQI_Bucket", "City", "Season", "Date", "Month",
        "AQI_Bucket_Encoded",
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drops)
    y = df["AQI_Bucket_Encoded"]

    X = X.select_dtypes(include=[np.number])

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Classes: {y.nunique()}")
    return X, y
