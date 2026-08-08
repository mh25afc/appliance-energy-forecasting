"""Part 1: Data preparation and exploratory data analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from . import config

sns.set_style("whitegrid")


def load_raw_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    df_raw = pd.read_csv(path, parse_dates=["date"])
    df_raw = df_raw.set_index("date").sort_index()
    return df_raw


def data_quality_report(df_raw: pd.DataFrame) -> dict:
    missing = df_raw.isna().sum()
    full_range = pd.date_range(start=df_raw.index.min(), end=df_raw.index.max(), freq="10min")
    missing_timestamps = full_range.difference(df_raw.index)
    return {
        "shape": df_raw.shape,
        "date_range": (df_raw.index.min(), df_raw.index.max()),
        "n_missing_values": int(missing.sum()),
        "n_missing_timestamps": len(missing_timestamps),
    }


def resample_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df_raw.select_dtypes(include="number").columns
    return df_raw[numeric_cols].resample("1h").mean()


def run_stationarity_tests(series: pd.Series, name: str = "Series") -> dict:
    adf_result = adfuller(series.dropna(), autolag="AIC")
    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "name": name,
        "adf_stat": adf_result[0], "adf_p": adf_result[1],
        "adf_verdict": "stationary" if adf_result[1] < 0.05 else "non-stationary",
        "kpss_stat": kpss_result[0], "kpss_p": kpss_result[1],
        "kpss_verdict": "non-stationary" if kpss_result[1] < 0.05 else "stationary",
    }


def run_eda(df: pd.DataFrame, out_dir: str = config.RESULTS_DIR) -> None:
    """Reproduce every EDA plot and stationarity test from Part 1 of the notebook."""
    os.makedirs(out_dir, exist_ok=True)
    target = df[config.TARGET]

    fig, ax = plt.subplots(figsize=(14, 4))
    target.plot(ax=ax, linewidth=0.6)
    ax.set_title("Hourly Appliance Energy Use Full Series")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_full_series.png"), dpi=150)
    plt.close(fig)

    two_week = target.loc[target.index.min() + pd.Timedelta(days=20):
                           target.index.min() + pd.Timedelta(days=34)]
    fig, ax = plt.subplots(figsize=(14, 4))
    two_week.plot(ax=ax)
    ax.set_title("Hourly Appliance Energy Use Two Week Zoom")
    ax.set_ylabel("Appliances (Wh)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_two_week_zoom.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    target.hist(bins=50, ax=axes[0])
    axes[0].set_title("Distribution of Hourly Appliance Energy Use")
    axes[0].set_xlabel("Appliances (Wh)")
    np.log1p(target).hist(bins=50, ax=axes[1])
    axes[1].set_title("Log-transformed Distribution")
    axes[1].set_xlabel("log(1 + Appliances)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_distribution.png"), dpi=150)
    plt.close(fig)

    df_tmp = df.copy()
    df_tmp["hour"] = df_tmp.index.hour
    df_tmp["dayofweek"] = df_tmp.index.dayofweek
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df_tmp.groupby("hour")[config.TARGET].mean().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Average Appliance Use by Hour of Day")
    dow_avg = df_tmp.groupby("dayofweek")[config.TARGET].mean()
    dow_avg.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg.plot(kind="bar", ax=axes[1], color="darkorange")
    axes[1].set_title("Average Appliance Use by Day of Week")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_hour_dow_avg.png"), dpi=150)
    plt.close(fig)

    corr_cols = [config.TARGET, "T_out", "RH_out", "Windspeed", "Visibility", "T1", "RH_1", "lights"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: Appliances vs Sensor/Weather Variables")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    stl_daily = STL(target, period=24, robust=True).fit()
    fig = stl_daily.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_stl_decomposition_daily.png"), dpi=150)
    plt.close(fig)

    stl_weekly = STL(target, period=168, robust=True).fit()
    fig = stl_weekly.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_stl_decomposition_weekly.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(target, lags=72, ax=axes[0])
    axes[0].set_title("ACF up to 72 hours")
    plot_pacf(target, lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("PACF up to 72 hours")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_acf_pacf.png"), dpi=150)
    plt.close(fig)

    tests = [
        run_stationarity_tests(target, "Appliances (level)"),
        run_stationarity_tests(target.diff().dropna(), "Appliances (1st difference)"),
        run_stationarity_tests(target.diff(24).dropna(), "Appliances (24h seasonal difference)"),
    ]
    pd.DataFrame(tests).to_csv(os.path.join(out_dir, "stationarity_tests.csv"), index=False)
