"""Diagnostic: which meters and inverters were producing, day by day?

If parts of the plant are off for weeks at a time, daily kWh moves with equipment
availability, not just weather. That would explain a weak weather-to-output link.

Run from the project folder:   python -m modules.diagnose
Saves docs/figures/5_equipment_by_day.png and data/demo/equipment_by_day.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from modules import pv_data

# Columns seen in the system 1332 files (only those present are used).
CANDIDATES = ["metered_ac_power__2638", "meter1_ac_power__2639", "meter2_ac_power__2640",
              "meter3_ac_power__2641", "inv1_ac_power__2642", "inv2_ac_power__2646"]


def daily_max_power(files, time_col, unit):
    """Highest reading per day for each candidate column, in kW (one file at a time)."""
    rows = []
    for f in files:
        keep = lambda c: c == time_col or c in CANDIDATES
        df = (pd.read_parquet(f) if Path(f).suffix.lower() == ".parquet"
              else pd.read_csv(f, usecols=keep))
        cols = [c for c in CANDIDATES if c in df.columns]
        t = pd.to_datetime(df[time_col])
        day = t.dt.normalize()
        m = df[cols].clip(lower=0).groupby(day).max()
        rows.append(m)
    out = pd.concat(rows).groupby(level=0).max().sort_index()
    return out / (1000.0 if unit.upper() == "W" else 1.0)


def main():
    files = pv_data.files_in_window(config.RAW_FOLDER, config.START, config.END)
    if not files:
        raise SystemExit(pv_data.explain_no_files(config.RAW_FOLDER, config.START, config.END))
    print(f"Reading {len(files)} files (about a minute)...")
    m = daily_max_power(files, config.TIME_COL, config.POWER_UNIT)
    m.index.name = "date"
    m.to_csv(Path(config.DEMO_FOLDER) / "equipment_by_day.csv")

    print("\nMedian of each day's highest reading (kW), by month:")
    print(m.groupby(m.index.month).median().round(0).to_string())
    print("\nDays on which each column never exceeded 5 kW (equipment off or not reporting):")
    print((m < 5).sum().to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    for c in m.columns:
        ax.plot(m.index, m[c], lw=1.2, label=c.split("__")[0])
    ax.set_ylabel("Highest reading of the day (kW)")
    ax.set_title("Which meters and inverters were producing, day by day")
    ax.legend(ncol=3, fontsize=8, loc="upper center")
    ax.set_ylim(top=m.max().max() * 1.2)
    fig.autofmt_xdate(); fig.tight_layout()
    Path(config.FIG_FOLDER).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(config.FIG_FOLDER) / "5_equipment_by_day.png", dpi=160)
    print("\nChart saved: docs/figures/5_equipment_by_day.png")


if __name__ == "__main__":
    main()