"""Forecast accuracy metrics used throughout this pipeline.

This project uses RMSE / MAE / MAPE (rather than sMAPE/MASE), matching the
metric choices made in the original notebook.
"""

import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_forecast(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


def summarize_folds(fold_metrics: list, model_name: str):
    """Turn a list of per-fold {'RMSE','MAE','MAPE'} dicts into one mean+/-std summary row."""
    import pandas as pd
    fold_df = pd.DataFrame(fold_metrics)
    return pd.DataFrame([{
        "Model": model_name,
        "RMSE_mean": fold_df["RMSE"].mean(), "RMSE_std": fold_df["RMSE"].std(),
        "MAE_mean": fold_df["MAE"].mean(), "MAE_std": fold_df["MAE"].std(),
        "MAPE_mean": fold_df["MAPE"].mean(), "MAPE_std": fold_df["MAPE"].std(),
    }])
