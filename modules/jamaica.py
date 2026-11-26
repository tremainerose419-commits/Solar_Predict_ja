"""Jamaica demo page (M4): SIMULATED daily output for a fictional farm, from Jamaican weather.

This is a formula applied to weather, NOT measured output. It illustrates the method on the
Jamaica page (label it "not validated"). It must never be used as the target to train or
test the Random Forest (the circular-baseline trap your trainer warned about).
"""
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"                  # ACTUAL weather
FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"    # archived FORECASTS

MJ_PER_KWH = 3.6


def fetch_daily_ghi(lat, lon, start_date, end_date, forecast=True, tz="America/Jamaica"):
    """Daily total sunlight (MJ/m2) for a Jamaican site, saved by the caller to CSV.

    forecast=True uses archived day-ahead style forecasts (right for the demo's 'Tomorrow' idea).
    forecast=False uses actual past weather (an upper bound, not a forecast).
    """
    r = requests.get(FORECAST_URL if forecast else ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
        "daily": "shortwave_radiation_sum", "timezone": tz}, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]),
                         "ghi_mj_m2": d["shortwave_radiation_sum"]})


def simulate_daily_kwh(ghi_df, dc_kw, performance_ratio=0.8):
    """Simulated kWh/day = DC size (kWp) x peak sun hours x performance ratio.

    1 kWh/m2 = 3.6 MJ/m2, so peak sun hours = MJ/m2 / 3.6.
    Uses the same 0.8 ratio as the physics baseline. The ratio is NOT tuned to hit a
    benchmark: tuning to a target and then comparing with that target proves nothing.
    """
    df = ghi_df.copy()
    df["peak_sun_hours"] = df["ghi_mj_m2"] / MJ_PER_KWH
    df["simulated_kwh"] = (dc_kw * df["peak_sun_hours"] * performance_ratio).round(2)
    return df


def capacity_factor(df, ac_kw, kwh_col="simulated_kwh"):
    """Capacity factor = energy / (AC capacity x hours covered by the days present)."""
    hours = 24 * df[kwh_col].notna().sum()
    return df[kwh_col].sum() / (ac_kw * hours)


if __name__ == "__main__":
    # Fictional 20 MW AC farm, Clarendon (approximate coordinates; check before using).
    ghi = fetch_daily_ghi(17.97, -77.24, "2023-04-01", "2024-03-31", forecast=False)
    sim = simulate_daily_kwh(ghi, dc_kw=20_000 * 1.3)   # 1.3 = assumed DC/AC ratio
    print(sim.head())
    print(f"Simulated capacity factor: {capacity_factor(sim, ac_kw=20_000):.1%}")
