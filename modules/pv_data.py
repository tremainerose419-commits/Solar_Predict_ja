"""Load PVDAQ daily files and turn them into daily kWh (measured output)."""
import re
from pathlib import Path

import pandas as pd


def files_in_window(folder, start, end):
    """Pick the daily CSV files whose date in the file name is inside the window.

    PVDAQ stores one file per day. Files with no date in the name are kept
    (with a warning) so nothing is lost silently.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    keep, undated = [], []
    for f in sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".csv", ".parquet")):
        m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", f.name)
        if not m:
            undated.append(f)
            continue
        day = pd.Timestamp("-".join(m.groups()))
        if start - pd.Timedelta(days=1) <= day <= end + pd.Timedelta(days=1):
            keep.append(f)
    if undated:
        print(f"WARNING: {len(undated)} files have no date in the name; including them.")
    return keep + undated


def load_power(files, time_col, power_cols):
    """Read the timestamp and power columns from every file into one table."""
    cols = [time_col] + list(power_cols)
    parts = [
        pd.read_parquet(f, columns=cols) if Path(f).suffix.lower() == ".parquet"
        else pd.read_csv(f, usecols=cols)
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.drop_duplicates(subset=time_col).sort_values(time_col).reset_index(drop=True)


def solar_noon_by_month(df, time_col, power_cols, threshold=0.05):
    """Median clock time of the MIDDLE of the production hours, for each month.

    Uses the midpoint between the first and last time power is above 5% of the
    day's maximum. Unlike 'time of the peak', this is not fooled by output that
    is clipped at a ceiling (a flat top makes the peak time jump around).

    Solar noon at Golden, CO is close to 12:00 Mountain STANDARD time all year.
    - Midpoint near 12.0-12.3 all year -> timestamps are fixed standard time.
    - Midpoint near 13.0-13.3 in summer -> timestamps follow daylight saving time.
    """
    power = df[power_cols].sum(axis=1).clip(lower=0)
    t = df[time_col]
    day = t.dt.normalize()
    day_max = power.groupby(day).transform("max")
    on = (power > threshold * day_max) & (day_max > 0)
    clock = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    first = clock[on].groupby(day[on]).min()
    last = clock[on].groupby(day[on]).max()
    mid = (first + last) / 2
    return mid.groupby(mid.index.month).median().round(2)


def daily_kwh(df, time_col, power_cols, power_unit="W", data_tz="Etc/GMT+7",
              local_tz="America/Denver", min_completeness=0.9):
    """Sum power readings into daily kWh and keep only well-covered days.

    power_unit: "W" or "kW". Check against the system size (1153.488 kW).
    data_tz:    the timezone the timestamps are really in. "Etc/GMT+7" means
                fixed UTC-7 (Mountain standard time all year). Use
                "America/Denver" if the check shows they follow daylight saving.
    """
    power = df[power_cols].sum(axis=1, min_count=1)
    if power_unit.upper() == "W":
        power = power / 1000.0
    ts = df[time_col].dt.tz_localize(data_tz, ambiguous="NaT", nonexistent="shift_forward")
    ts = ts.dt.tz_convert(local_tz)

    interval_h = ts.diff().dropna().median().total_seconds() / 3600
    energy = power.clip(lower=0) * interval_h  # kW x hours = kWh; negative night values -> 0
    day = ts.dt.tz_localize(None).dt.normalize()

    out = pd.DataFrame({
        "kwh": energy.groupby(day).sum(),
        "readings": power.notna().groupby(day).sum(),
        "peak_kw": power.clip(lower=0).groupby(day).max(),
    })
    out["completeness"] = out["readings"] / (24 / interval_h)
    total = len(out)
    out = out[out["completeness"] >= min_completeness]
    print(f"Interval {interval_h * 3600:.0f} seconds; kept {len(out)} of {total} days "
          f"(>= {min_completeness:.0%} of readings).")
    out.index.name = "date"
    return out[["kwh", "completeness", "peak_kw"]]


def explain_no_files(folder, start, end):
    """Say what is actually in the folder, so the cause of '0 files' is obvious."""
    root = Path(folder)
    if not root.exists():
        return f"The folder '{root.resolve()}' does not exist. Create it and put the PVDAQ files in it."
    everything = [p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if not everything:
        return (f"The folder '{root.resolve()}' is empty. Download the system's daily files "
                "into it first (see README).")
    kinds = sorted({p.suffix.lower() or "(no extension)" for p in everything})
    sample = ", ".join(p.name for p in everything[:5])
    return (f"Found {len(everything)} files in '{root.resolve()}' (types: {', '.join(kinds)}; "
            f"e.g. {sample}), but none are .csv/.parquet files dated {start} to {end}. "
            "Check the file types and that the dates in the file names fall inside the window.")
