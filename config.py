"""Settings for Solar Predict JA."""

# --- The measured system (NREL PVDAQ) ---
SYSTEM_ID = 1332
SYSTEM_NAME = "NREL Parking Garage, Golden, CO"
SIZE_KW = 1153.488
LAT, LON = 39.7388, -105.1732
LOCAL_TZ = "America/Denver"

# --- Window = 2021-01-01 to 2022-01-18 (Actual Weather Fallback: Upper Bound, Not a Forecast) ---
START, END = "2021-01-01", "2022-01-18"

# --- Files ---
RAW_FOLDER = "data/raw"
S3_SYSTEM_PREFIX = None        # leave None: modules/download.py finds the folder itself
DEMO_FOLDER = "data/demo"
DAILY_KWH_FILE = "data/demo/daily_kwh.csv"
FORECAST_FILE = "data/demo/forecast_daily.csv"
JOINED_FILE = "data/demo/joined_daily.csv"
EVALUATION_FILE = "data/demo/evaluation_baselines.csv"
MODEL_FILE = "models/random_forest.joblib"

# --- TODO: check these against the header of one daily PVDAQ file ---
TIME_COL = "measured_on"
POWER_COLS = ["metered_ac_power__2638"]      # AC power column(s) to add together
POWER_UNIT = "W"               # "W" or "kW" (the unit check in prepare.py tells you)
DATA_TZ = "Etc/GMT+7"          # fixed UTC-7; use "America/Denver" if the peak-time check says so

# --- Method settings (from the trainer's locked scope) ---
MIN_COMPLETENESS = 0.9         # keep days with at least 90% of readings
PERFORMANCE_RATIO = 0.8        # physics formula
TRAIN_FRACTION = 0.75          # earlier 75% train, latest 25% test, never shuffled

# --- Week 2 (M2): model, outputs ---
ALTITUDE_M = 1770                                  # from the PVDAQ system metadata
CLEARSKY_FILE = "data/demo/clearsky_daily.csv"     # cached; delete to recompute with pvlib
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
MODEL_INFO_FILE = "models/model_info.json"
EVAL_ALL_FILE = "data/demo/evaluation_all.csv"
SKILL_FILE = "data/demo/skill_scores.csv"
PRED_FILE = "data/demo/test_predictions.csv"
IMPORTANCE_FILE = "data/demo/feature_importance.csv"
FIG_FOLDER = "docs/figures"
SUMMARY_FILE = "docs/results_summary.md"
