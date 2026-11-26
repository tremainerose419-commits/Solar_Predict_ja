"""Random Forest: choose inputs, fit, predict, save."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from modules import features


def new_model():
    return RandomForestRegressor(**config.RF_PARAMS)


def fit(train, cols):
    return new_model().fit(train[cols], train["kwh"])


def mae(actual, predicted):
    return float((actual - predicted).abs().mean())


def choose_feature_set(train, val_fraction=0.25):
    """Pick the input set using only TRAINING days (the last 25% of them as a check set).

    This keeps the test days untouched, so the final test score stays honest.
    Returns (name, validation MAE for each set, columns).
    """
    cut = int(len(train) * (1 - val_fraction))
    fit_part, val_part = train.iloc[:cut], train.iloc[cut:]
    candidates = {"without_day_of_year": features.FEATURES_BASE,
                  "with_day_of_year": features.FEATURES_WITH_DOY}
    if "peak_kw_7d" in train.columns:
        candidates["with_peak_power"] = features.FEATURES_WITH_PEAK
    scores = {}
    for name, cols in candidates.items():
        m = fit(fit_part, cols)
        scores[name] = mae(val_part["kwh"], pd.Series(m.predict(val_part[cols]), index=val_part.index))
    best = min(scores, key=scores.get)
    return best, scores, candidates[best]


def importances(model, cols):
    return (pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def save(model, cols, info):
    Path(config.MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols}, config.MODEL_FILE)
    Path(config.MODEL_INFO_FILE).write_text(json.dumps(info, indent=2, default=str))


def load():
    bundle = joblib.load(config.MODEL_FILE)
    return bundle["model"], bundle["features"]
