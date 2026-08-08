"""Shared configuration constants used by every pipeline stage."""

import os

DATA_PATH = os.environ.get("ENERGY_DATA_PATH", "data/energydata_complete.csv")
RESULTS_DIR = "results"

TARGET = "Appliances"
HORIZON = 24
TEST_DAYS = 14

# --- SARIMAX (Part 4) ---
SEASONAL_ORDER = (1, 0, 1, 24)
SARIMAX_P_RANGE = range(0, 7)
SARIMAX_D_RANGE = range(0, 3)
SARIMAX_Q_RANGE = range(0, 7)
SARIMAX_SEARCH_WINDOW_DAYS = 21

# --- Feature engineering (Part 5) ---
LAG_HOURS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (24, 168)
WEATHER_COLS = ["T_out", "RH_out", "Windspeed", "Visibility", "Press_mm_hg", "T1", "RH_1", "lights"]

# --- XGBoost (Part 6) ---
XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

# --- Chronos (Part 7) ---
CHRONOS_MODEL_NAME = "amazon/chronos-t5-small"
CHRONOS_CONTEXT_HOURS = 512
CHRONOS_NUM_SAMPLES = 20
