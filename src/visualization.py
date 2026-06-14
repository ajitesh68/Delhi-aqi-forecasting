"""
Visualization utilities for AQI analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# Consistent style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_pollutant_histograms(
    df: pd.DataFrame,
    pollutants: list = None,
    save_dir: str = None,
):
    """
    Plot histograms for all specified pollutants.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    pollutants : list, optional
        List of pollutant column names.
    save_dir : str, optional
        Directory to save figures. If None, just displays.
    """
    pollutants = pollutants or [
        "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "Benzene", "AQI",
    ]

    for pol in pollutants:
        if pol not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.histplot(df[pol], bins=50, kde=True)
        plt.title(f"Distribution of {pol}")
        plt.xlabel(pol)
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"hist_{pol.lower().replace('.', '')}.png"),
                dpi=150, bbox_inches="tight",
            )
        plt.show()


def plot_aqi_trend_by_city(
    df: pd.DataFrame,
    cities: list = None,
    save_path: str = None,
):
    """
    Line plot showing AQI trend over years for selected cities.
    """
    if cities is None:
        # Top 5 most polluted + bottom 5 cleanest
        city_aqi = df.groupby("City")["AQI"].mean().sort_values()
        cities = list(city_aqi.head(5).index) + list(city_aqi.tail(5).index)

    line_data = (
        df[df["City"].isin(cities)]
        .groupby(["City", "Year"])["AQI"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=line_data, x="Year", y="AQI", hue="City", marker="o")
    plt.title("AQI Trend by City (2015–2020)")
    plt.xlabel("Year")
    plt.ylabel("Average AQI")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_covid_impact(
    df: pd.DataFrame,
    city: str = "Delhi",
    years: list = None,
    save_path: str = None,
):
    """
    Bar chart comparing monthly AQI between two years (e.g., 2019 vs 2020).
    """
    years = years or [2019, 2020]

    covid_data = (
        df[(df["City"] == city) & (df["Year"].isin(years))]
        .groupby(["Month", "Year"])["AQI"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=covid_data, x="Month", y="AQI", hue="Year")
    plt.title(f"{city} AQI: {years[0]} vs {years[1]} (COVID Impact)")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Heatmap of correlations between numerical features.
    """
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_boxplots_by_city(
    df: pd.DataFrame,
    cities: list = None,
    pollutants: list = None,
    save_dir: str = None,
):
    """
    Box plots of pollutant distributions across selected cities.
    """
    pollutants = pollutants or [
        "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "Benzene", "AQI",
    ]

    if cities:
        df = df[df["City"].isin(cities)]

    for pol in pollutants:
        if pol not in df.columns:
            continue
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x="City", y=pol)
        plt.title(f"{pol} Distribution by City")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"box_{pol.lower().replace('.', '')}.png"),
                dpi=150, bbox_inches="tight",
            )
        plt.show()


def plot_aqi_bucket_pie(
    df: pd.DataFrame,
    save_path: str = None,
):
    """
    Pie chart of AQI Bucket distribution.
    """
    bucket_counts = df["AQI_Bucket"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(
        bucket_counts.values,
        labels=bucket_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("AQI Bucket Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title: str = "Actual vs Predicted AQI",
    save_path: str = None,
):
    """
    Scatter plot of actual vs predicted values with perfect prediction line.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=15)
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    feat_imp_df: pd.DataFrame,
    top_n: int = 15,
    save_path: str = None,
):
    """
    Bar plot of top N feature importances.
    """
    data = feat_imp_df.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=data, x="Importance", y="Feature", orient="h")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
