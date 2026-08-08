"""Part 7: Time-series foundation model (Chronos), used zero-shot.

Requires the optional `torch` and `chronos-forecasting` dependencies.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .metrics import evaluate_forecast, summarize_folds


def load_pipeline():
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        config.CHRONOS_MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
    )
    print("Chronos pipeline loaded.")
    return pipeline


def run_walk_forward(df: pd.DataFrame, folds: list, pipeline, out_dir: str = config.RESULTS_DIR):
    import torch

    os.makedirs(out_dir, exist_ok=True)
    fold_results, predictions = [], []

    for train_end_time, forecast_idx in folds:
        history = df.loc[:train_end_time, config.TARGET].iloc[-config.CHRONOS_CONTEXT_HOURS:]
        context = torch.tensor(history.values, dtype=torch.float32)

        forecast_samples = pipeline.predict(context, prediction_length=config.HORIZON, num_samples=config.CHRONOS_NUM_SAMPLES)
        samples = forecast_samples[0].numpy()
        y_pred = np.median(samples, axis=0)
        lower, upper = np.quantile(samples, [0.1, 0.9], axis=0)

        y_true = df.loc[forecast_idx, config.TARGET].values
        fold_results.append(evaluate_forecast(y_true, y_pred))
        predictions.append((forecast_idx, y_pred, lower, upper))

    summary = summarize_folds(fold_results, "Chronos (zero-shot)")
    summary.to_csv(os.path.join(out_dir, "chronos_summary.csv"), index=False)

    _plot_example_fold(df, predictions, out_dir)
    return fold_results, predictions, summary


def _plot_example_fold(df, predictions, out_dir, example_fold_idx=5):
    example_fold_idx = min(example_fold_idx, len(predictions) - 1)
    forecast_idx, y_pred, lower, upper = predictions[example_fold_idx]
    y_true = df.loc[forecast_idx, config.TARGET]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_idx, y_true.values, label="Actual", color="black", linewidth=2)
    ax.plot(forecast_idx, y_pred, label="Chronos forecast (median)", color="purple", linestyle="--")
    ax.fill_between(forecast_idx, lower, upper, color="purple", alpha=0.15, label="80% interval")
    ax.set_title(f"Chronos (zero-shot) Example Fold Forecast\n({forecast_idx.min()} to {forecast_idx.max()})")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "18_chronos_example_fold.png"), dpi=150)
    plt.close(fig)
