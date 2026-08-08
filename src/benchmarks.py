"""Part 3: Benchmark forecasting models."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .metrics import evaluate_forecast, summarize_folds


def forecast_mean(history: pd.Series, horizon: int = config.HORIZON) -> np.ndarray:
    return np.full(horizon, history.mean())


def forecast_naive(history: pd.Series, horizon: int = config.HORIZON) -> np.ndarray:
    return np.full(horizon, history.iloc[-1])


def forecast_seasonal_naive(history: pd.Series, season: int, horizon: int = config.HORIZON) -> np.ndarray:
    """season=24 -> daily, season=168 -> weekly."""
    last_season = history.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def forecast_drift(history: pd.Series, horizon: int = config.HORIZON) -> np.ndarray:
    y_first, y_last, n = history.iloc[0], history.iloc[-1], len(history)
    slope = (y_last - y_first) / (n - 1)
    return y_last + slope * np.arange(1, horizon + 1)


BENCHMARK_MODELS = {
    "Mean": lambda h: forecast_mean(h),
    "Naive": lambda h: forecast_naive(h),
    "Daily Seasonal Naive": lambda h: forecast_seasonal_naive(h, season=24),
    "Weekly Seasonal Naive": lambda h: forecast_seasonal_naive(h, season=168),
    "Drift": lambda h: forecast_drift(h),
}


def run_benchmarks(df: pd.DataFrame, folds: list, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fold_results = {name: [] for name in BENCHMARK_MODELS}
    predictions = {name: [] for name in BENCHMARK_MODELS}

    for train_end_time, forecast_idx in folds:
        history = df.loc[:train_end_time, config.TARGET]
        y_true = df.loc[forecast_idx, config.TARGET].values

        for name, model_fn in BENCHMARK_MODELS.items():
            y_pred = model_fn(history)
            fold_results[name].append(evaluate_forecast(y_true, y_pred))
            predictions[name].append((forecast_idx, y_pred))

    summary_rows = [summarize_folds(fold_results[name], name) for name in BENCHMARK_MODELS]
    summary = pd.concat(summary_rows, ignore_index=True).sort_values("RMSE_mean").reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "benchmark_summary.csv"), index=False)

    _plot_rmse_bar(summary, out_dir)
    _plot_example_fold(df, predictions, out_dir)
    return fold_results, predictions, summary


def _plot_rmse_bar(summary, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary["Model"], summary["RMSE_mean"], xerr=summary["RMSE_std"], color="steelblue", capsize=4)
    ax.set_xlabel("RMSE (Wh) mean +/- std across folds")
    ax.set_title("Benchmark Model Comparison (lower is better)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "10_benchmark_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_example_fold(df, predictions, out_dir, example_fold_idx=5):
    example_fold_idx = min(example_fold_idx, len(predictions["Naive"]) - 1)
    forecast_idx, _ = predictions["Naive"][example_fold_idx]
    y_true = df.loc[forecast_idx, config.TARGET]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_idx, y_true.values, label="Actual", color="black", linewidth=2)
    for name in BENCHMARK_MODELS:
        _, y_pred = predictions[name][example_fold_idx]
        ax.plot(forecast_idx, y_pred, label=name, linestyle="--", alpha=0.8)
    ax.set_title(f"Example Fold 24h Forecast Comparison\n({forecast_idx.min()} to {forecast_idx.max()})")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "11_benchmark_example_fold.png"), dpi=150)
    plt.close(fig)
