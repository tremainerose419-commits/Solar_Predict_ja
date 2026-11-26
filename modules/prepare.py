"""M0 + M1: daily kWh, five sample rows, forecasts, two baselines scored.

Run from the project folder:  python -m modules.prepare
Settings live in config.py.
"""
from pathlib import Path

import pandas as pd

import config
from modules import baselines, pv_data, weather


def main():
    Path(config.DEMO_FOLDER).mkdir(parents=True, exist_ok=True)

    # 1. Load the daily files in the window
    files = pv_data.files_in_window(config.RAW_FOLDER, config.START, config.END)
    print(f"{len(files)} files in the window")
    if not files:
        raise SystemExit("\nNo data files to read.\n"
                         + pv_data.explain_no_files(config.RAW_FOLDER, config.START, config.END))
    raw = pv_data.load_power(files, config.TIME_COL, config.POWER_COLS)

    # 2. Checks first: units and timezone
    scale = 1000 if config.POWER_UNIT.upper() == "W" else 1
    peak_kw = raw[config.POWER_COLS].sum(axis=1).max() / scale
    print(f"Highest reading = {peak_kw:.0f} kW (system size is {config.SIZE_KW} kW; "
          "should be close, not 1000x off)")
    print("Median clock time of the middle of the production hours, by month "
          "(about 12.0 to 12.3 all year = fixed standard time; about 13.0 in summer = daylight saving):")
    print(pv_data.solar_noon_by_month(raw, config.TIME_COL, config.POWER_COLS))

    # 3. Daily kWh and the five sample rows for the trainer
    daily = pv_data.daily_kwh(raw, config.TIME_COL, config.POWER_COLS, config.POWER_UNIT,
                              config.DATA_TZ, config.LOCAL_TZ, config.MIN_COMPLETENESS)
    daily.to_csv(config.DAILY_KWH_FILE)
    print("\nFive sample rows of daily kWh:")
    print(daily.head(5).round(1))

    # 4. Archived day-ahead forecasts (downloaded once, then saved)
    wfile = Path(config.FORECAST_FILE)
    if not wfile.exists():
        hourly = weather.download_hourly(config.LAT, config.LON, config.START, config.END,
                                         config.LOCAL_TZ)
        weather.daily_features(hourly).to_csv(wfile)
    wx = pd.read_csv(wfile, index_col="date", parse_dates=True)

    # 5. Join, split by date, score the two baselines on the test days
    table = baselines.build_table(daily, wx, config.SIZE_KW)
    train, test = baselines.chronological_split(table, config.TRAIN_FRACTION)
    print(f"\n{len(train)} training days ({train.index.min().date()} to {train.index.max().date()}), "
          f"{len(test)} test days ({test.index.min().date()} to {test.index.max().date()})")

    # Optional third baseline: the physics formula with its performance ratio fitted on TRAINING days only
    ratio = baselines.fit_performance_ratio(train, config.SIZE_KW)
    print(f"\nPhysics formula check: the trainer's ratio is {config.PERFORMANCE_RATIO}; "
          f"the ratio that fits the training days is {ratio:.2f}")
    train = baselines.add_calibrated_physics(train, config.SIZE_KW, ratio)
    test = baselines.add_calibrated_physics(test, config.SIZE_KW, ratio)
    table = baselines.add_calibrated_physics(table, config.SIZE_KW, ratio)
    ev = baselines.evaluation_table(test, ("persistence", "physics", "physics_calibrated"))
    print("\nEvaluation table (test days only):")
    print(ev)
    ev.to_csv(config.EVALUATION_FILE)
    table.to_csv(config.JOINED_FILE)


if __name__ == "__main__":
    main()
