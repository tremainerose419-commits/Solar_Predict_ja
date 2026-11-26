"""M2: build inputs, train the Random Forest, score it against the baselines.

Run from the project folder (after modules.prepare):   python -m modules.train
"""
from pathlib import Path

import pandas as pd

import config
from modules import baselines, features, model

METHODS = ["persistence", "physics", "physics_calibrated", "random_forest"]


def run(daily, wx, clearsky, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    table = features.build_model_table(daily, wx, clearsky, config.SIZE_KW)
    train, test = baselines.chronological_split(table, config.TRAIN_FRACTION)
    say(f"{len(table)} scored days: {len(train)} training ({train.index.min().date()} to "
        f"{train.index.max().date()}), {len(test)} test ({test.index.min().date()} to "
        f"{test.index.max().date()})")

    # physics formula with a performance ratio fitted on TRAINING days only (a fairer baseline)
    ratio = baselines.fit_performance_ratio(train, config.SIZE_KW)
    train = baselines.add_calibrated_physics(train, config.SIZE_KW, ratio)
    test = baselines.add_calibrated_physics(test, config.SIZE_KW, ratio)

    # choose the input set using training days only, then fit on all training days
    name, val_scores, cols = model.choose_feature_set(train)
    say("Check on training days (MAE, kWh):", {k: round(v, 1) for k, v in val_scores.items()},
        "-> using", name)
    rf = model.fit(train, cols)
    test = test.copy()
    test["random_forest"] = rf.predict(test[cols])

    # the same model with the other input set, only for transparency
    other = features.FEATURES_WITH_DOY if name == "without_day_of_year" else features.FEATURES_BASE
    other_mae = model.mae(test["kwh"], pd.Series(model.fit(train, other).predict(test[other]),
                                                 index=test.index))

    ev = baselines.evaluation_table(test, METHODS)
    mae_rf = ev.loc["random_forest", "MAE_kWh"]
    skill = pd.DataFrame({
        "baseline_MAE_kWh": [ev.loc[b, "MAE_kWh"] for b in METHODS[:3]],
        "skill": [round(baselines.skill(mae_rf, ev.loc[b, "MAE_kWh"]), 3) for b in METHODS[:3]],
    }, index=METHODS[:3])
    skill.index.name = "baseline"
    imp = model.importances(rf, cols)

    info = {
        "feature_set": name, "features": cols, "validation_mae_kwh": val_scores,
        "test_mae_other_feature_set": round(other_mae, 1),
        "train_days": len(train), "test_days": len(test),
        "train_period": [str(train.index.min().date()), str(train.index.max().date())],
        "test_period": [str(test.index.min().date()), str(test.index.max().date())],
        "average_test_kwh": round(float(test["kwh"].mean()), 1),
        "fitted_performance_ratio": round(float(ratio), 3),
        "test_mae_kwh": float(mae_rf), "rf_params": config.RF_PARAMS,
    }
    return dict(table=table, train=train, test=test, rf=rf, cols=cols, ev=ev,
                skill=skill, imp=imp, info=info)


def main():
    daily = pd.read_csv(config.DAILY_KWH_FILE, index_col="date", parse_dates=True)
    wx = pd.read_csv(config.FORECAST_FILE, index_col="date", parse_dates=True)
    clearsky = features.load_clearsky()
    print(f"Clear-sky sunlight (kWh/m2/day): min {clearsky.min():.1f}, max {clearsky.max():.1f} "
          "(expect roughly 3 to 9)")
    r = run(daily, wx, clearsky)

    Path(config.DEMO_FOLDER).mkdir(parents=True, exist_ok=True)
    r["ev"].to_csv(config.EVAL_ALL_FILE)
    r["skill"].to_csv(config.SKILL_FILE)
    r["imp"].to_csv(config.IMPORTANCE_FILE, index=False)
    r["test"][["kwh"] + METHODS].round(1).to_csv(config.PRED_FILE)
    r["table"].to_csv(config.JOINED_FILE)
    model.save(r["rf"], r["cols"], r["info"])

    print("\nEvaluation table (test days only):")
    print(r["ev"])
    print("\nSkill of the Random Forest (1 - model MAE / baseline MAE; positive = beats it):")
    print(r["skill"])
    print("\nWhat the model relies on most:")
    print(r["imp"].round(3).to_string(index=False))
    print(f"\nSaved model to {config.MODEL_FILE}")


if __name__ == "__main__":
    main()
