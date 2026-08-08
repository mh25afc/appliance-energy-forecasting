"""Part 8: Consolidated evaluation across all 8 models."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

ADVANCED_MODEL_NAMES = ["SARIMAX", "XGBoost", "Chronos (zero-shot)"]


def build_final_comparison(*summaries: pd.DataFrame, out_dir: str = config.RESULTS_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    final = pd.concat(summaries, ignore_index=True).sort_values("RMSE_mean").reset_index(drop=True)
    final.insert(0, "Rank (by RMSE)", range(1, len(final) + 1))
    final.to_csv(os.path.join(out_dir, "final_model_comparison.csv"), index=False)
    return final


def plot_metric_comparison(final_comparison: pd.DataFrame, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = [("RMSE_mean", "RMSE_std", "RMSE (Wh)"),
                        ("MAE_mean", "MAE_std", "MAE (Wh)"),
                        ("MAPE_mean", "MAPE_std", "MAPE (%)")]
    plot_order = final_comparison.sort_values("RMSE_mean")

    for ax, (mean_col, std_col, label) in zip(axes, metrics_to_plot):
        colors = ["crimson" if m in ADVANCED_MODEL_NAMES else "steelblue" for m in plot_order["Model"]]
        ax.barh(plot_order["Model"], plot_order[mean_col], xerr=plot_order[std_col], color=colors, capsize=3)
        ax.set_xlabel(label)
        ax.set_title(f"{label} mean +/- std across folds")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "19_final_model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_per_fold_boxplots(benchmark_fold_results: dict, sarimax_fold_results: list,
                            xgb_fold_results: list, chronos_fold_results: list,
                            final_comparison: pd.DataFrame, out_dir: str = config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for model_name, fold_list in benchmark_fold_results.items():
        for i, m in enumerate(fold_list):
            records.append({"Model": model_name, "Fold": i, **m})
    for model_name, fold_list in [("SARIMAX", sarimax_fold_results), ("XGBoost", xgb_fold_results),
                                   ("Chronos (zero-shot)", chronos_fold_results)]:
        for i, m in enumerate(fold_list):
            records.append({"Model": model_name, "Fold": i, **m})

    all_folds_long = pd.DataFrame(records)
    model_order = final_comparison.sort_values("RMSE_mean")["Model"].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE"]):
        sns.boxplot(data=all_folds_long, x="Model", y=metric, order=model_order, ax=ax)
        ax.set_title(f"{metric} Distribution Across Folds")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "20_per_fold_boxplots.png"), dpi=150)
    plt.close(fig)


def improvement_over_strongest_benchmark(benchmark_summary: pd.DataFrame, final_comparison: pd.DataFrame,
                                           out_dir: str = config.RESULTS_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    best_rmse = benchmark_summary["RMSE_mean"].min()
    best_mae = benchmark_summary["MAE_mean"].min()
    best_mape = benchmark_summary["MAPE_mean"].min()

    advanced = final_comparison[final_comparison["Model"].isin(ADVANCED_MODEL_NAMES)].copy()
    advanced["RMSE_improvement_%"] = (1 - advanced["RMSE_mean"] / best_rmse) * 100
    advanced["MAE_improvement_%"] = (1 - advanced["MAE_mean"] / best_mae) * 100
    advanced["MAPE_improvement_%"] = (1 - advanced["MAPE_mean"] / best_mape) * 100

    result = advanced[["Model", "RMSE_improvement_%", "MAE_improvement_%", "MAPE_improvement_%"]]
    result.to_csv(os.path.join(out_dir, "improvement_over_strongest_benchmark.csv"), index=False)
    return result
