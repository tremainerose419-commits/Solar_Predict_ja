"""Join measured kWh with forecasts, build the two baselines and score them."""
import numpy as np
import pandas as pd

import config

PERFORMANCE_RATIO = config.PERFORMANCE_RATIO


def build_table(daily, weather, size_kw):
    """One row per calendar day: measured kWh, forecast features, baselines.

    Missing days are kept as gaps (not skipped), so 'yesterday' really means
    the previous calendar day.
    """
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    kwh = daily["kwh"].reindex(full)
    df = pd.DataFrame({"kwh": kwh}).join(weather, how="left")
    df["persistence"] = df["kwh"].shift(1)  # tomorrow = today
    df["physics"] = size_kw * df["forecast_irr_kwh_m2"] * PERFORMANCE_RATIO
    df.index.name = "date"
    return df.dropna()  # score every method on exactly the same days


def chronological_split(df, train_fraction=config.TRAIN_FRACTION):
    """Earlier 75% of days for training, latest 25% for testing. Never shuffled."""
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


def scores(actual, predicted):
    err = actual - predicted
    return {
        "MAE_kWh": err.abs().mean(),
        "RMSE_kWh": float(np.sqrt((err ** 2).mean())),
        "MAE_pct_of_avg": 100 * err.abs().mean() / actual.mean(),
    }


def evaluation_table(test, methods=("persistence", "physics")):
    rows = {m: scores(test["kwh"], test[m]) for m in methods}
    table = pd.DataFrame(rows).T
    return table.round(1)


def skill(model_mae, baseline_mae):
    """skill = 1 - model MAE / baseline MAE. Positive: the model beats the baseline."""
    return 1 - model_mae / baseline_mae


def fit_performance_ratio(train, size_kw):
    """Performance ratio that best matches the TRAINING days: total kWh / (size x total irradiance).
    Fitted on training days only, so the test days stay unseen."""
    return train["kwh"].sum() / (size_kw * train["forecast_irr_kwh_m2"].sum())


def add_calibrated_physics(df, size_kw, ratio):
    df = df.copy()
    df["physics_calibrated"] = size_kw * df["forecast_irr_kwh_m2"] * ratio
    return df
