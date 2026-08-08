"""Part 2: Defining the forecasting problem - target, split, and rolling-origin folds."""

import os

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def train_test_split(df: pd.DataFrame, test_days: int = config.TEST_DAYS):
    test_hours = test_days * 24
    split_point = df.index[-test_hours]
    train = df.loc[:split_point].iloc[:-1]
    test = df.loc[split_point:]
    return train, test, split_point


def get_rolling_folds(df: pd.DataFrame, test_start, horizon: int = config.HORIZON, n_folds: int = config.TEST_DAYS):
    """Returns a list of (train_end_time, forecast_index) tuples: train_end_time is the
    last timestamp available to the model, forecast_index is the 24 timestamps to predict."""
    folds = []
    for i in range(n_folds):
        window_start = test_start + pd.Timedelta(hours=i * horizon)
        window_end = window_start + pd.Timedelta(hours=horizon - 1)
        if window_end > df.index.max():
            break
        forecast_index = df.loc[window_start:window_end].index
        train_end_time = window_start - pd.Timedelta(hours=1)
        folds.append((train_end_time, forecast_index))
    return folds


def plot_train_test_split(train, test, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    train[config.TARGET].plot(ax=ax, label="Train", linewidth=0.6)
    test[config.TARGET].plot(ax=ax, label=f"Test (last {config.TEST_DAYS} days)", linewidth=0.6, color="darkorange")
    ax.axvline(test.index.min(), color="red", linestyle="--", linewidth=1)
    ax.set_title("Train / Test Split")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_train_test_split.png"), dpi=150)
    plt.close(fig)
