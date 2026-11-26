"""Model inputs (features).

Rule for every input: would we know it on the day BEFORE the day we predict?
- forecast_*            : weather forecast issued before the day (known)
- clearsky_kwh_m2       : astronomy, known years ahead
- yesterday_kwh         : measured output of the previous day (known)
- ratio_7d              : measured / formula output over the previous 7 days (known)
- peak_kw_*             : highest power of the previous day / previous 7 days (known)
- day_of_year           : a calendar fact
Nothing from the day being predicted (its measured kWh) is ever an input.
"""
from pathlib import Path

import pandas as pd

import config

FEATURES_BASE = ["forecast_irr_kwh_m2", "forecast_cloud_pct", "forecast_temp_c",
                 "clearsky_kwh_m2", "yesterday_kwh", "ratio_7d"]
FEATURES_WITH_DOY = FEATURES_BASE + ["day_of_year"]
FEATURES_WITH_PEAK = FEATURES_BASE + ["peak_kw_yesterday", "peak_kw_7d"]


def clearsky_daily_kwh_m2(lat, lon, start, end, tz, altitude):
    """Best-case (no clouds) sunlight per day in kWh/m2, calculated with pvlib.

    Needs `pip install pvlib`. Sanity check: roughly 3 to 5 in winter and 7 to 9 in summer here.
    """
    from pvlib.location import Location

    loc = Location(lat, lon, tz=tz, altitude=altitude)
    times = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                          freq="15min", tz=tz, inclusive="left")
    ghi = loc.get_clearsky(times)["ghi"]                 # W/m2 every 15 minutes
    daily = ghi.resample("D").sum() * 0.25 / 1000.0      # kWh/m2 per day
    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"
    daily.name = "clearsky_kwh_m2"
    return daily


def load_clearsky():
    """Use the saved file if it exists, otherwise calculate it with pvlib and save it."""
    f = Path(config.CLEARSKY_FILE)
    if f.exists():
        return pd.read_csv(f, index_col="date", parse_dates=True)["clearsky_kwh_m2"]
    daily = clearsky_daily_kwh_m2(config.LAT, config.LON, config.START, config.END,
                                  config.LOCAL_TZ, config.ALTITUDE_M)
    daily.to_csv(f)
    return daily


def build_model_table(daily, weather, clearsky, size_kw, performance_ratio=None):
    """One row per calendar day with the target, all inputs and the baselines.

    Missing days stay as gaps, so 'yesterday' and 'the last 7 days' always mean real
    calendar days. Rows with any missing input are dropped, so every method is scored
    on exactly the same days.
    """
    pr = config.PERFORMANCE_RATIO if performance_ratio is None else performance_ratio
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    df = pd.DataFrame({"kwh": daily["kwh"].reindex(full)})
    df = df.join(weather, how="left").join(clearsky.rename("clearsky_kwh_m2"), how="left")

    df["yesterday_kwh"] = df["kwh"].shift(1)
    if "peak_kw" in daily.columns:
        # how much of the plant was working: highest power of the PREVIOUS day / previous 7 days
        peak = daily["peak_kw"].reindex(full)
        df["peak_kw_yesterday"] = peak.shift(1)
        df["peak_kw_7d"] = peak.shift(1).rolling(7, min_periods=3).max()
    df["physics"] = size_kw * df["forecast_irr_kwh_m2"] * pr
    df["persistence"] = df["yesterday_kwh"]

    # actual / formula over the PREVIOUS 7 days (captures dust, faults, clipping)
    both = df["kwh"].notna() & df["physics"].notna()
    actual_7 = df["kwh"].where(both).shift(1).rolling(7, min_periods=3).sum()
    formula_7 = df["physics"].where(both).shift(1).rolling(7, min_periods=3).sum()
    df["ratio_7d"] = actual_7 / formula_7

    df["day_of_year"] = df.index.dayofyear
    df.index.name = "date"
    return df.dropna()
