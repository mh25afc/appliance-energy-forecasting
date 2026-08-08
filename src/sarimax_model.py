"""Part 4: SARIMAX - AIC grid search, residual diagnostics, walk-forward forecasting."""

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import config
from .metrics import evaluate_forecast, summarize_folds

warnings.filterwarnings("ignore")


def run_aic_grid_search(train: pd.DataFrame, out_dir: str = config.RESULTS_DIR) -> pd.DataFrame:
    """AIC grid search over (p, d, q) with a fixed seasonal order, on a recent
    window of training data for speed. Cached to sarimax_grid_search.csv."""
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.path.join(out_dir, "sarimax_grid_search.csv")
    all_combos = list(itertools.product(config.SARIMAX_P_RANGE, config.SARIMAX_D_RANGE, config.SARIMAX_Q_RANGE))

    if os.path.exists(checkpoint):
        cached = pd.read_csv(checkpoint)
        if len(cached) > 0:
            return cached

    search_data = train[config.TARGET].iloc[-config.SARIMAX_SEARCH_WINDOW_DAYS * 24:]
    results = []
    for p, d, q in all_combos:
        try:
            model = SARIMAX(search_data, order=(p, d, q), seasonal_order=config.SEASONAL_ORDER,
                             enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=30, method="lbfgs")
            results.append({"p": p, "d": d, "q": q, "AIC": fit.aic})
        except Exception:
            continue

    grid_results = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    grid_results.to_csv(checkpoint, index=False)
    print(f"Fit {len(grid_results)} / {len(all_combos)} combinations successfully.")
    return grid_results


def select_best_order(grid_results: pd.DataFrame):
    best_row = grid_results.iloc[0]
    return (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))


def fit_final_model(train: pd.DataFrame, order):
    model = SARIMAX(train[config.TARGET], order=order, seasonal_order=config.SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=100, method="lbfgs")


def plot_residual_diagnostics(fitted_result, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    residuals = fitted_result.resid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(residuals.dropna(), lags=72, ax=axes[0])
    axes[0].set_title("ACF of SARIMAX Residuals")
    axes[1].hist(residuals.dropna(), bins=50, color="steelblue", edgecolor="white")
    axes[1].set_title("Distribution of SARIMAX Residuals")
    axes[1].set_xlabel("Residual (Wh)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "12_sarimax_residual_diagnostics.png"), dpi=150)
    plt.close(fig)


def walk_forward_forecast(df: pd.DataFrame, folds: list, fitted_result, out_dir: str = config.RESULTS_DIR):
    """Feeds each fold's actuals back into the fitted state-space model (no
    re-optimisation of parameters) and forecasts the next 24 hours."""
    os.makedirs(out_dir, exist_ok=True)
    fold_results, predictions = [], []
    current_fit = fitted_result

    for train_end_time, forecast_idx in folds:
        forecast_res = current_fit.get_forecast(steps=config.HORIZON)
        y_pred = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=0.05)

        y_true = df.loc[forecast_idx, config.TARGET].values
        fold_results.append(evaluate_forecast(y_true, y_pred))
        predictions.append((forecast_idx, y_pred, conf_int))

        new_obs = df.loc[forecast_idx, config.TARGET]
        current_fit = current_fit.append(new_obs, refit=False)

    summary = summarize_folds(fold_results, "SARIMAX")
    summary.to_csv(os.path.join(out_dir, "sarimax_summary.csv"), index=False)

    _plot_example_fold(df, predictions, out_dir)
    return fold_results, predictions, summary


def _plot_example_fold(df, predictions, out_dir, example_fold_idx=5):
    example_fold_idx = min(example_fold_idx, len(predictions) - 1)
    forecast_idx, y_pred, conf_int = predictions[example_fold_idx]
    y_true = df.loc[forecast_idx, config.TARGET]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_idx, y_true.values, label="Actual", color="black", linewidth=2)
    ax.plot(forecast_idx, y_pred, label="SARIMAX forecast", color="darkgreen", linestyle="--")
    ax.fill_between(forecast_idx, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color="darkgreen", alpha=0.15, label="95% CI")
    ax.set_title(f"SARIMAX Example Fold Forecast with 95% CI\n({forecast_idx.min()} to {forecast_idx.max()})")
    ax.set_ylabel("Appliances (Wh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "14_sarimax_example_fold.png"), dpi=150)
    plt.close(fig)
