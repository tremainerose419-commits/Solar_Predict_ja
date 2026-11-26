"""Small checks for each part. Run from the project folder:  python tests/test_modules.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from modules import baselines, benchmark, features, model, pv_data, weather


def test_mae_hand_worked_example():
    actual = pd.Series([300, 320, 310])
    predicted = pd.Series([290, 330, 330])
    assert round(baselines.scores(actual, predicted)["MAE_kWh"], 2) == 13.33


def test_skill_formula():
    assert round(baselines.skill(35.8, 45.4), 2) == 0.21   # model 35.8 vs formula 45.4
    assert round(baselines.skill(35.8, 75.2), 2) == 0.52   # model 35.8 vs persistence 75.2
    assert baselines.skill(50, 50) == 0


def test_split_is_chronological_and_not_shuffled():
    idx = pd.date_range("2021-01-01", periods=100)
    df = pd.DataFrame({"kwh": range(100)}, index=idx)
    train, test = baselines.chronological_split(df, 0.75)
    assert len(train) == 75 and len(test) == 25
    assert train.index.max() < test.index.min()


def test_persistence_uses_previous_calendar_day():
    days = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-04"])  # 3 Jan missing
    daily = pd.DataFrame({"kwh": [100.0, 200.0, 400.0]}, index=days)
    wx = pd.DataFrame({"forecast_irr_kwh_m2": 5.0, "forecast_cloud_pct": 0.0,
                       "forecast_temp_c": 20.0}, index=pd.date_range("2021-01-01", periods=4))
    table = baselines.build_table(daily, wx, size_kw=100)
    assert table.loc["2021-01-02", "persistence"] == 100.0
    assert "2021-01-04" not in table.index.astype(str)   # yesterday (3 Jan) is missing, so dropped
    assert table.loc["2021-01-02", "physics"] == 100 * 5.0 * 0.8


def test_daily_kwh_units_and_completeness():
    t = pd.date_range("2021-06-01", periods=96, freq="15min")          # a full day, 15-min
    full = pd.DataFrame({"measured_on": t, "ac_power": 500_000.0})       # 500 kW as watts
    kwh = pv_data.daily_kwh(full, "measured_on", ["ac_power"], "W", "America/Denver")
    assert abs(kwh["kwh"].iloc[0] - 500 * 24) < 1e-6                     # 500 kW x 24 h
    half = full.iloc[:40]                                               # under 90% -> dropped
    assert len(pv_data.daily_kwh(half, "measured_on", ["ac_power"], "W", "America/Denver")) == 0


def test_solar_noon_not_fooled_by_clipping():
    t = pd.date_range("2021-07-08", periods=5760, freq="15s")
    hr = t.hour + t.minute / 60
    p = np.clip(1_000_000 * np.clip(np.sin((hr - 6) / 12 * np.pi), 0, None), 0, 500_000)
    df = pd.DataFrame({"measured_on": t, "ac_power": p})
    mid = pv_data.solar_noon_by_month(df, "measured_on", ["ac_power"])
    assert abs(mid.iloc[0] - 12.0) < 0.1


def test_fitted_performance_ratio():
    train = pd.DataFrame({"kwh": [400.0, 800.0], "forecast_irr_kwh_m2": [5.0, 10.0]})
    assert abs(baselines.fit_performance_ratio(train, size_kw=100) - 0.8) < 1e-9


def test_weather_daily_features():
    t = pd.date_range("2021-06-01", periods=24, freq="h")
    hourly = pd.DataFrame({"time": t, "shortwave_radiation": 500.0,
                           "cloud_cover": 40.0, "temperature_2m": 20.0})
    out = weather.daily_features(hourly)
    assert out["forecast_irr_kwh_m2"].iloc[0] == 12.0   # 24 x 500 W/m2 = 12 kWh/m2


def test_fiscal_year_and_hours():
    assert benchmark.get_jamaica_fiscal_year(pd.Timestamp("2021-04-01")) == "FY2021/22"
    assert benchmark.get_jamaica_fiscal_year(pd.Timestamp("2022-03-31")) == "FY2021/22"
    assert benchmark.hours_in_fiscal_year("FY2021/22") == 8760
    assert benchmark.hours_in_fiscal_year("FY2023/24") == 8784      # contains 29 Feb 2024


def test_benchmark_skips_part_years_and_computes_cf():
    days = pd.date_range("2021-04-01", "2022-03-31")
    full = pd.DataFrame({"timestamp": days, "net_export_kwh": 640.0})
    out = benchmark.evaluate_platform_vs_our(full, 100.0, "content_solar")
    assert len(out) == 1 and abs(out["User_Capacity_Factor_%"].iloc[0] - 26.67) < 0.01
    assert len(benchmark.evaluate_platform_vs_our(full.iloc[:100], 100.0)) == 0   # too few days


def _synthetic_inputs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2021-01-01", periods=n)
    doy = days.dayofyear.to_numpy()
    clear = 6 + 2.5 * np.sin(2 * np.pi * (doy - 80) / 365)
    idx = rng.uniform(0.3, 1.0, n)
    irr = clear * idx
    kwh = 400 * irr + rng.normal(0, 60, n)
    daily = pd.DataFrame({"kwh": kwh, "completeness": 1.0}, index=pd.Index(days, name="date"))
    wx = pd.DataFrame({"forecast_irr_kwh_m2": irr * rng.normal(1, 0.1, n),
                       "forecast_cloud_pct": (1 - idx) * 100, "forecast_temp_c": 15 + 10 * np.sin(doy / 58)},
                      index=pd.Index(days, name="date"))
    cs = pd.Series(clear, index=pd.Index(days, name="date"), name="clearsky_kwh_m2")
    return daily, wx, cs


def test_inputs_never_use_the_day_itself():
    daily, wx, cs = _synthetic_inputs()
    a = features.build_model_table(daily, wx, cs, 1000.0)
    d2 = daily.copy()
    day = a.index[50]
    d2.loc[day, "kwh"] *= 3                                   # change ONLY that day's measured output
    b = features.build_model_table(d2, wx, cs, 1000.0)
    assert a.loc[day, "ratio_7d"] == b.loc[day, "ratio_7d"]
    assert a.loc[day, "yesterday_kwh"] == b.loc[day, "yesterday_kwh"]
    nxt = day + pd.Timedelta(days=1)
    assert a.loc[nxt, "yesterday_kwh"] != b.loc[nxt, "yesterday_kwh"]   # tomorrow's input does change


def test_model_beats_persistence_on_learnable_data():
    daily, wx, cs = _synthetic_inputs()
    table = features.build_model_table(daily, wx, cs, 1000.0)
    train, test = baselines.chronological_split(table, 0.75)
    name, scores, cols = model.choose_feature_set(train)
    rf = model.fit(train, cols)
    pred = pd.Series(rf.predict(test[cols]), index=test.index)
    assert model.mae(test["kwh"], pred) < model.mae(test["kwh"], test["persistence"])
    assert "kwh" not in cols


def test_peak_inputs_use_only_previous_days_and_help_when_plant_changes():
    rng = np.random.default_rng(5)
    n = 400
    days = pd.date_range("2021-01-01", periods=n)
    cap = np.repeat(rng.choice([250.0, 500.0, 700.0], size=n // 50 + 1), 50)[:n]   # equipment on/off in blocks
    irr = rng.uniform(3, 7, n)
    kwh = cap * irr * 0.6 + rng.normal(0, 40, n)
    idx = pd.Index(days, name="date")
    daily = pd.DataFrame({"kwh": kwh, "completeness": 1.0, "peak_kw": cap * rng.uniform(0.95, 1.0, n)}, index=idx)
    wx = pd.DataFrame({"forecast_irr_kwh_m2": irr, "forecast_cloud_pct": 50.0, "forecast_temp_c": 15.0}, index=idx)
    cs = pd.Series(7.0, index=idx, name="clearsky_kwh_m2")
    t = features.build_model_table(daily, wx, cs, 1000.0)
    # changing one day's peak must not change that day's own inputs
    d2 = daily.copy(); day = t.index[60]; d2.loc[day, "peak_kw"] *= 5
    t2 = features.build_model_table(d2, wx, cs, 1000.0)
    assert t.loc[day, "peak_kw_yesterday"] == t2.loc[day, "peak_kw_yesterday"]
    assert t.loc[day, "peak_kw_7d"] == t2.loc[day, "peak_kw_7d"]
    # and the peak inputs should reduce the test error when the plant's capacity changes
    train, test = baselines.chronological_split(t, 0.75)
    base = model.fit(train, features.FEATURES_BASE)
    peak = model.fit(train, features.FEATURES_WITH_PEAK)
    mae_base = model.mae(test["kwh"], pd.Series(base.predict(test[features.FEATURES_BASE]), index=test.index))
    mae_peak = model.mae(test["kwh"], pd.Series(peak.predict(test[features.FEATURES_WITH_PEAK]), index=test.index))
    assert mae_peak < mae_base


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(tests)} checks passed")
