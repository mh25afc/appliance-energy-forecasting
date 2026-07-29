# Household Appliance Energy Demand Forecasting

A comparative study of four categories of 24-hour-ahead time-series forecasters 
simple statistical benchmarks, a classical seasonal model (SARIMAX), a feature-based
machine-learning model (XGBoost), and a pretrained time-series foundation model
(Chronos, used zero-shot) on the UCI Appliance Energy Prediction dataset.

## Project overview

The goal of this project is not to fit a single model, but to rigorously compare eight
forecasters (5 benchmarks + SARIMAX + XGBoost + Chronos) under identical conditions:
the same 24-hour forecast horizon, the same 14 rolling-origin test folds, and the same
three error metrics (RMSE, MAE, MAPE). The full analysis data preparation, exploratory
analysis, model development, evaluation, and critical discussion is contained in a
single, self-contained Jupyter notebook.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebook/
│   └── Energy_Demand_Forecasting.ipynb      # Full analysis: EDA through final model comparison
├── data/
│   └── energydata_complete.csv      # Raw 10-minute resolution dataset
├── report/
│   └── Forecasting Household Appliance Energy Demand.pdf  # Written report
└── figures/
    └── ...                           # Key result figures, exported from the notebook
```

## Dataset

**Source:** [UCI Machine Learning Repository Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of
energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.

10-minute resolution sensor and weather data from a low-energy house in Belgium,
January–May 2016. Resampled to hourly resolution for this study (3,290 hourly
observations, no missing values or timestamp gaps).

## Clone the repository

```bash
git clone https://github.com/mh25afc/appliance-energy-forecasting.git
cd https://github.com/mh25afc/appliance-energy-forecasting
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note on Part 7 (Chronos):** the foundation-model section downloads pretrained weights
(~150–200MB) from Hugging Face on first run and requires an active internet connection.
All other sections run fully offline once the dataset is in place.

## Running the analysis

```bash
jupyter notebook notebook/Energy_Demand_Forecasting.ipynb
```

Approximate runtime: ~2 minutes for Parts 1–6 and 8–9 (EDA, benchmarks, SARIMAX,
XGBoost, evaluation, analysis questions); Part 7 (Chronos) adds a few more minutes on
CPU for model download and zero-shot inference across all folds.

## Methodology summary

| Stage | Approach |
|---|---|
| **Target / horizon** | `Appliances` (Wh), 24 hours ahead, hourly resolution |
| **Evaluation** | 14 non-overlapping 24-hour rolling-origin folds over the final 14 test days |
| **Benchmarks** | Mean, Naive, Daily Seasonal Naive, Weekly Seasonal Naive, Drift |
| **SARIMAX** | AIC grid search over 147 non-seasonal orders + seasonal grid at daily period (s=24) |
| **XGBoost** | Direct multi-horizon model with lag, rolling-window, calendar, and current-weather features |
| **Chronos** | Zero-shot forecasting with `chronos-t5-small`, no training on this dataset |

## Results

Final comparison across all 8 models, mean ± std over 14 rolling-origin folds:

| Rank | Model | RMSE (Wh) | MAE (Wh) | MAPE (%) |
|---|---|---|---|---|
| 1 | SARIMAX(2,1,6)×(1,0,1,24) | 56.70 | 37.59 | 34.85 |
| 2 | XGBoost (direct multi-horizon) | 61.02 | 44.04 | 43.73 |
| 3 | Chronos-T5-small (zero-shot) | 61.71 | 36.02 | **23.26** |
| 4 | Mean | 66.91 | 50.26 | 53.70 |
| 5 | Weekly Seasonal Naive | 69.23 | 43.46 | 37.35 |
| 6 | Daily Seasonal Naive | 76.30 | 48.31 | 43.33 |
| 7 | Naive | 97.77 | 85.55 | 112.91 |
| 8 | Drift | 97.98 | 85.80 | 113.31 |

**Headline finding:** Chronos, used purely zero-shot with no training on this dataset,
achieves the best typical-case accuracy (MAE, MAPE) of all eight models including two
that were fit directly to this exact series while SARIMAX achieves the best worst-case
control (RMSE). Full discussion, critical analysis, and the six required analysis
questions are answered in both the notebook (Parts 8–9) and the written report.

## Report

See [`report/Forecasting Household Appliance Energy Demand.pdf`]
for the full written report, including methodology, results, figures, critical
discussion, answers to the six analysis questions, future improvements, and references.

## References

- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*, 785–794.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

