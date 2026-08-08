"""Part 5: Feature engineering, including the "direct" multi-horizon dataset.

Unlike a single-step lag setup, this builds one row per (origin time t0,
horizon h) pair, so a single model is trained to predict every horizon
1..24 directly, rather than recursively.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def build_lag_rolling_features(series: pd.Series, lags=config.LAG_HOURS, rolling_windows=config.ROLLING_WINDOWS) -> pd.DataFrame:
    """All rolling stats are computed on lag-1 shifted data, so nothing at time t
    uses information from t itself."""
    feats = pd.DataFrame(index=series.index)
    for lag in lags:
        feats[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)
    for window in rolling_windows:
        feats[f"rollmean_{window}"] = shifted.rolling(window).mean()
        feats[f"rollstd_{window}"] = shifted.rolling(window).std()
    return feats


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features known in advance for any future timestamp."""
    feats = pd.DataFrame(index=index)
    feats["hour"] = index.hour
    feats["dayofweek"] = index.dayofweek
    feats["is_weekend"] = (index.dayofweek >= 5).astype(int)
    feats["month"] = index.month
    feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * feats["dayofweek"] / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * feats["dayofweek"] / 7)
    return feats


def build_weather_features(df: pd.DataFrame, weather_cols=config.WEATHER_COLS) -> pd.DataFrame:
    cols = [c for c in weather_cols if c in df.columns]
    weather = df[cols].copy()
    weather.columns = [f"{c}_t0" for c in weather.columns]
    return weather


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    lag_roll = build_lag_rolling_features(df[config.TARGET])
    calendar = build_calendar_features(df.index)
    weather = build_weather_features(df)
    return pd.concat([lag_roll, calendar, weather], axis=1)


def plot_feature_correlation(feature_table: pd.DataFrame, target: pd.Series, out_dir: str = config.RESULTS_DIR) -> pd.Series:
    os.makedirs(out_dir, exist_ok=True)
    corr = feature_table.assign(**{config.TARGET: target}).corr()[config.TARGET].drop(config.TARGET)
    corr = corr.sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    corr.plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Feature Correlation with Appliances (reference only)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "15_feature_correlations.png"), dpi=150)
    plt.close(fig)
    return corr


def build_direct_dataset(feature_table: pd.DataFrame, target_series: pd.Series, horizons=range(1, config.HORIZON + 1)) -> pd.DataFrame:
    """One row per (t0, horizon): all feature_table columns, a 'horizon' column,
    and 'target' = target_series value at t0 + horizon hours."""
    rows = []
    for h in horizons:
        block = feature_table.copy()
        block["horizon"] = h
        block["target"] = target_series.shift(-h)
        rows.append(block)
    long_df = pd.concat(rows, axis=0)
    return long_df.dropna()
