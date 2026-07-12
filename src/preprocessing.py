"""
Data cleaning and preprocessing utilities.
"""
import pandas as pd
import numpy as np


# Columns to drop due to excessive missing values (>25%)
HIGH_NULLITY_COLUMNS = ["Xylene", "Toluene", "NH3"]


def drop_high_nullity_columns(
    df: pd.DataFrame,
    columns: list = None,
) -> pd.DataFrame:
    """
    Drop columns with excessive missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to drop. Defaults to HIGH_NULLITY_COLUMNS.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed.
    """
    cols_to_drop = columns or HIGH_NULLITY_COLUMNS
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"Dropped {len(existing)} columns: {existing}")
    return df


def impute_numerical_by_group(
    df: pd.DataFrame,
    group_cols: list = None,
) -> pd.DataFrame:
    """
    Impute missing numerical values using Season x City group medians.
    Falls back to global median if group median is also NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must have 'Season' and 'City' columns).
    group_cols : list, optional
        Columns to group by. Defaults to ['Season', 'City'].

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed numerical columns.
    """
    df = df.copy()
    group_cols = group_cols or ["Season", "City"]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude Year, Month from imputation
    num_cols = [c for c in num_cols if c not in ["Year", "Month"]]

    for col in num_cols:
        before = df[col].isnull().sum()
        df[col] = df.groupby(group_cols)[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback to global median
        df[col] = df[col].fillna(df[col].median())
        after = df[col].isnull().sum()
        if before > 0:
            print(f"  {col}: {before} -> {after} missing values")

    return df


def impute_aqi_bucket_by_group(
    df: pd.DataFrame,
    group_cols: list = None,
) -> pd.DataFrame:
    """
    Impute missing AQI_Bucket values using the mode per Season x City group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_cols : list, optional
        Columns to group by. Defaults to ['Season', 'City'].

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed AQI_Bucket.
    """
    df = df.copy()
    group_cols = group_cols or ["Season", "City"]

    before = df["AQI_Bucket"].isnull().sum()
    df["AQI_Bucket"] = df.groupby(group_cols)["AQI_Bucket"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
    )
    after = df["AQI_Bucket"].isnull().sum()
    print(f"  AQI_Bucket: {before} -> {after} missing values")
    return df


def cap_aqi_outliers(
    df: pd.DataFrame,
    city: str,
    threshold: float = 500,
    method: str = "season_median",
) -> pd.DataFrame:
    """
    Cap AQI outliers for a specific city.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    city : str
        City name to apply capping.
    threshold : float
        AQI threshold above which to cap.
    method : str
        'season_median' — replace with season median for that city.
        'hard_cap' — replace with threshold value.

    Returns
    -------
    pd.DataFrame
        DataFrame with capped AQI values.
    """
    df = df.copy()
    mask = (df["City"] == city) & (df["AQI"] > threshold)
    count = mask.sum()

    if method == "season_median":
        city_data = df[(df["City"] == city) & (df["AQI"] <= threshold)]
        season_medians = city_data.groupby("Season")["AQI"].median().to_dict()
        df.loc[mask, "AQI"] = df.loc[mask, "Season"].map(season_medians)
    elif method == "hard_cap":
        df.loc[mask, "AQI"] = threshold

    print(f"  {city}: Capped {count} outlier(s) (>{threshold}) using {method}")
    return df


def full_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete cleaning pipeline:
    1. Drop high-nullity columns
    2. Impute numerical values (Season x City median)
    3. Impute AQI_Bucket (Season x City mode)
    4. Cap AQI outliers

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with Season column already created.

    Returns
    -------
    pd.DataFrame
        Fully cleaned DataFrame.
    """
    print("=== Step 1: Dropping high-nullity columns ===")
    df = drop_high_nullity_columns(df)

    print("\n=== Step 2: Numerical imputation (Season x City median) ===")
    df = impute_numerical_by_group(df)

    print("\n=== Step 3: AQI_Bucket imputation (Season x City mode) ===")
    df = impute_aqi_bucket_by_group(df)

    print("\n=== Step 4: Capping AQI outliers ===")
    df = cap_aqi_outliers(df, "Ahmedabad", 500, "season_median")
    for city in ["Delhi", "Patna", "Lucknow", "Gurugram", "Amritsar"]:
        df = cap_aqi_outliers(df, city, 500, "hard_cap")
    
    

    print("\n=== Step 5: Dropping Suspicious Outliers ===")
    df = drop_suspicious_outliers(df, cities=["Delhi", "Patna", "Lucknow", "Gurugram", "Amritsar"])

    print("\n=== Step 6: Fixing Ahmedabad Pollutants ===")
    df = fix_ahmedabad_pollutants(df)
    

    print(f"\nFinal shape: {df.shape}")
    print(f"Remaining nulls: {df.isnull().sum().sum()}")
    return df


def drop_suspicious_outliers(df,cities,threshold=500):
    df=df.copy()
    for city in cities:
        mask = (df['city']==city) & (df['AQI']>threshold)
        count = mask.sum()
        if mask > 0:
            df = df[~mask]
            print(f"  {city}: Dropped {count} suspicious outlier(s) (>{threshold})")
    return df

def fix_ahmedabad_pollutants(df,thresholds):
    
    """
    Ahmedabad ke pollutants jo Lucknow ke 95th percentile se zyada hain, 
    unko Ahmedabad ke season-wise median se replace karo.
    """
    df = df.copy()
    if thresholds is None:
        thresholds = {"CO": 7.44, "NO2": 72.24, "SO2": 15.70, 
                      "Benzene": 11.30, "O3": 75.60}

    for pollutant, threshold in thresholds.items():
        if pollutant not in df.columns:
            continue
        ahm_median = df[df['City']=="Ahmedabad"].groupby("Season")[pollutant].median()
        for season, med_val in ahm_median.items():
            mask = (df[pollutant]>threshold) & (df['City']=="Ahmedabad") & (df['Season']==season)
            count = mask.sum()
            if count>0:
                df.loc[mask,pollutant]= med_val
                print(f"  {pollutant} in Ahmedabad - {season}: {count} values > {threshold} replaced with median {med_val:.2f}")
    return df
            


    
        