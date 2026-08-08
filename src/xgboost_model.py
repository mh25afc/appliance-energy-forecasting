"""Part 6: Feature-based ML model (XGBoost), direct multi-horizon approach.

A fresh model is trained per fold, restricted to rows whose target time is
already known by that fold's train_end_time -- this is what keeps the
walk-forward evaluation leakage-safe (see the forecast-origin discussion in
the report for why this matters).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import config
from .metrics import evaluate_forecast, summarize_folds


def run_walk_forward(direct_dataset: pd.DataFrame, feature_table: pd.DataFrame, df: pd.DataFrame,
                      folds: list, feature_cols: list, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    target_time = direct_dataset.index + pd.to_timedelta(direct_dataset["horizon"], unit="h")

    fold_results, predictions = [], []
    for train_end_time, forecast_idx in folds:
        train_mask = target_time <= train_end_time
        train_data = direct_dataset.loc[train_mask]

        model = XGBRegressor(**config.XGB_PARAMS)
        model.fit(train_data[feature_cols], train_data["target"])

        origin_features = feature_table.loc[[train_end_time] * config.HORIZON].reset_index(drop=True)
        origin_features["horizon"] = np.arange(1, config.HORIZON + 1)
        y_pred = model.predict(origin_features[feature_cols])

        y_true = df.loc[forecast_idx, config.TARGET].values
        fold_results.append(evaluate_forecast(y_true, y_pred))
        predictions.append((forecast_idx, y_pred))

    summary = summarize_folds(fold_results, "XGBoost")
    summary.to_csv(os.path.join(out_dir, "ml_summary.csv"), index=False)

    _plot_example_fold(df, predictions, out_dir)
    return fold_results, predictions, summary


def fit_reference_model(direct_dataset: pd.DataFrame, train_max_time, feature_cols: list) -> XGBRegressor:
    """One model fit on all available training data, purely for feature-importance
    interpretability (not used in the walk-forward evaluation itself)."""
    target_time = direct_dataset.index + pd.to_timedelta(direct_dataset["horizon"], unit="h")
    mask = target_time <= train_max_time
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(direct_dataset.loc[mask, feature_cols], direct_dataset.loc[mask, "target"])
    return model


def plot_feature_importance(model: XGBRegressor, feature_cols: list, out_dir: str = config.RESULTS_DIR) -> pd.Series:
    os.makedirs(out_dir, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    importances.plot(kind="barh", ax=ax, color="darkgreen")
    ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "16_xgboost_feature_importance.png"), dpi=150)
    plt.close(fig)
    return importances


def _plot_example_fold(df, predictions, out_dir, example_fold_idx=5):
    example_fold_idx = min(example_fold_idx, len(predictions) - 1)
    forecast_idx, y_pred = predictions[example_fold_idx]
    y_true = df.loc[forecast_idx, config.TARGET]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_idx, y_true.values, label="Actual", color="black", linewidth=2)
    ax.plot(forecast_idx, y_pred, label="XGBoost forecast", color="forestgreen", linestyle="--")
    ax.set_title(f"XGBoost Example Fold Forecast\n({forecast_idx.min()} to {forecast_idx.max()})")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "17_xgboost_example_fold.png"), dpi=150)
    plt.close(fig)
