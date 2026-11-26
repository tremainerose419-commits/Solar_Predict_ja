"""Archived forecasts from the Open-Meteo Historical Forecast API, as daily features."""
import pandas as pd
import requests

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def download_hourly(lat, lon, start, end, tz="America/Denver"):
    """One-off download of hourly forecast values for the site (save to CSV afterwards)."""
    r = requests.get(URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
        "timezone": tz,
    }, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def daily_features(hourly):
    """Hourly forecast -> one row per day.

    forecast_irr_kwh_m2: shortwave radiation (W/m2, hourly) summed and divided
    by 1000 = kWh/m2 per day (the 'hours of full sun' number).
    """
    day = hourly["time"].dt.normalize()
    g = hourly.groupby(day)
    out = pd.DataFrame({
        "forecast_irr_kwh_m2": g["shortwave_radiation"].sum() / 1000.0,
        "forecast_cloud_pct": g["cloud_cover"].mean(),
        "forecast_temp_c": g["temperature_2m"].mean(),
    })
    out.index.name = "date"
    return out
