#!/usr/bin/env python3
"""End-to-end forecasting pipeline for the Energy Demand Forecasting project.

Reproduces Parts 1-8 of the assignment notebook: data prep & EDA -> problem
definition -> benchmarks -> SARIMAX -> feature engineering (direct
multi-horizon) -> XGBoost -> Chronos (optional) -> consolidated evaluation.

Usage:
    python run_pipeline.py                  # full pipeline, including Chronos
    python run_pipeline.py --skip-chronos    # skip the foundation model

Note: the XGBoost stage retrains a fresh model at every one of the 14 folds
(the "direct" multi-horizon approach used in the original notebook), so it
is noticeably slower than a single-fit approach -- this is expected.

All intermediate CSVs and figures are written to results/.
"""

import argparse

from src import config, data_prep, problem_definition, benchmarks, sarimax_model
from src import feature_engineering, xgboost_model, evaluation


def main(skip_chronos: bool = False):
    print("=== Part 1: Data preparation and EDA ===")
    raw = data_prep.load_raw_data()
    print(data_prep.data_quality_report(raw))
    df = data_prep.resample_hourly(raw)
    data_prep.run_eda(df)

    print("\n=== Part 2: Forecasting problem definition ===")
    train, test, split_point = problem_definition.train_test_split(df)
    folds = problem_definition.get_rolling_folds(df, test_start=test.index.min())
    problem_definition.plot_train_test_split(train, test)
    print(f"Train: {len(train)}h, Test: {len(test)}h, Folds: {len(folds)}")

    print("\n=== Part 3: Benchmark models ===")
    benchmark_fold_results, benchmark_predictions, benchmark_summary = benchmarks.run_benchmarks(df, folds)
    print(benchmark_summary)

    print("\n=== Part 4: SARIMAX ===")
    grid_results = sarimax_model.run_aic_grid_search(train)
    best_order = sarimax_model.select_best_order(grid_results)
    print(f"Selected SARIMAX order: {best_order} x {config.SEASONAL_ORDER}")
    sarimax_fit = sarimax_model.fit_final_model(train, best_order)
    sarimax_model.plot_residual_diagnostics(sarimax_fit)
    sarimax_fold_results, sarimax_predictions, sarimax_summary = sarimax_model.walk_forward_forecast(df, folds, sarimax_fit)
    print(sarimax_summary)

    print("\n=== Part 5: Feature engineering (direct multi-horizon dataset) ===")
    feature_table = feature_engineering.build_feature_table(df)
    feature_engineering.plot_feature_correlation(feature_table, df[config.TARGET])
    direct_dataset = feature_engineering.build_direct_dataset(feature_table, df[config.TARGET])
    feature_cols = [c for c in direct_dataset.columns if c != "target"]
    print(f"Direct dataset shape: {direct_dataset.shape}, {len(feature_cols)} features")

    print("\n=== Part 6: XGBoost (direct multi-horizon, retrained per fold) ===")
    xgb_fold_results, xgb_predictions, xgb_summary = xgboost_model.run_walk_forward(
        direct_dataset, feature_table, df, folds, feature_cols
    )
    ref_model = xgboost_model.fit_reference_model(direct_dataset, train.index.max(), feature_cols)
    xgboost_model.plot_feature_importance(ref_model, feature_cols)
    print(xgb_summary)

    summaries = [benchmark_summary, sarimax_summary, xgb_summary]
    chronos_fold_results = []

    if not skip_chronos:
        print("\n=== Part 7: Chronos (zero-shot foundation model) ===")
        try:
            from src import chronos_model
            pipeline = chronos_model.load_pipeline()
            chronos_fold_results, chronos_predictions, chronos_summary = chronos_model.run_walk_forward(df, folds, pipeline)
            print(chronos_summary)
            summaries.append(chronos_summary)
        except ImportError:
            print("torch / chronos-forecasting not installed - skipping Part 7. "
                  "Install with `pip install torch chronos-forecasting` to include it.")

    print("\n=== Part 8: Consolidated evaluation ===")
    final_comparison = evaluation.build_final_comparison(*summaries)
    print(final_comparison)
    evaluation.plot_metric_comparison(final_comparison)
    if chronos_fold_results:
        evaluation.plot_per_fold_boxplots(
            benchmark_fold_results, sarimax_fold_results, xgb_fold_results, chronos_fold_results, final_comparison
        )
    improvement = evaluation.improvement_over_strongest_benchmark(benchmark_summary, final_comparison)
    print(improvement)

    print(f"\nDone. All figures and CSVs saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-chronos", action="store_true",
                         help="Skip the Chronos foundation-model stage (Part 7).")
    args = parser.parse_args()
    main(skip_chronos=args.skip_chronos)
