"""Compare a solar farm's annual output with published Jamaican utility-scale solar records.

USE: Jamaica demo page only (M4). This is annual totals, not daily output, so it cannot
train or test the Random Forest. The forecasting model stays on the PVDAQ data.

IMPORTANT: the OUR_BENCHMARKS numbers were supplied by the group and have NOT been checked
against the original OUR annual reports. Before showing them to anyone, open the reports,
confirm each figure and page number, and cite them in docs/data_notes.md.
Known problem: for FY2023/24 the Content Solar capacity factor matches 8,760 hours while the
Eight Rivers one matches 8,784 hours (leap year). One of the two is wrong.
Public sources agree only roughly: Eight Rivers about 26% and Content Solar about 25%
capacity factor, and Eight Rivers about 80 to 82 GWh per year.
"""
import numpy as np
import pandas as pd

OUR_BENCHMARKS = {   # UNVERIFIED, see the note above
    "FY2021/22": {"content_solar_mwh": 44372.14, "content_solar_cf": 0.2533,
                  "eight_rivers_mwh": 82410.00, "eight_rivers_cf": 0.2543},
    "FY2022/23": {"content_solar_mwh": 45798.20, "content_solar_cf": 0.2614,
                  "eight_rivers_mwh": 84150.00, "eight_rivers_cf": 0.2596},
    "FY2023/24": {"content_solar_mwh": 45751.92, "content_solar_cf": 0.2611,
                  "eight_rivers_mwh": 84620.00, "eight_rivers_cf": 0.2604},
}


def get_jamaica_fiscal_year(dt: pd.Timestamp) -> str:
    """Maps a timestamp to Jamaica's fiscal year (April 1 to March 31)."""
    if dt.month >= 4:
        return f"FY{dt.year}/{str(dt.year + 1)[-2:]}"
    return f"FY{dt.year - 1}/{str(dt.year)[-2:]}"


def hours_in_fiscal_year(fy_label: str) -> int:
    """Exact hours in the April 1 to March 31 period (accounts for leap years)."""
    start_year = int(fy_label[2:6])
    start = pd.Timestamp(year=start_year, month=4, day=1)
    end = pd.Timestamp(year=start_year + 1, month=4, day=1)
    return int((end - start).total_seconds() // 3600)


def evaluate_platform_vs_our(user_telemetry_df, system_ac_capacity_kw,
                             benchmark_plant="content_solar", min_coverage=0.95):
    """Compare a farm's annual energy and capacity factor with the published benchmark.

    user_telemetry_df: columns ['timestamp', 'net_export_kwh'] (one row per day)
    system_ac_capacity_kw: rated AC capacity in kW
    benchmark_plant: 'content_solar' (20 MW) or 'eight_rivers' (37 MW)
    min_coverage: a fiscal year is compared only if at least this share of its days is present,
                  otherwise a part-year would look like a very poor capacity factor.
    """
    df = user_telemetry_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fiscal_year"] = df["timestamp"].apply(get_jamaica_fiscal_year)
    grouped = (df.groupby("fiscal_year")
               .agg(user_kwh=("net_export_kwh", "sum"),
                    days=("timestamp", lambda s: s.dt.normalize().nunique()))
               .reset_index())
    grouped["user_mwh"] = grouped["user_kwh"] / 1000.0

    results, system_mw = [], system_ac_capacity_kw / 1000.0
    for _, row in grouped.iterrows():
        fy = row["fiscal_year"]
        if fy not in OUR_BENCHMARKS:
            continue
        hrs = hours_in_fiscal_year(fy)
        coverage = row["days"] / (hrs / 24)
        if coverage < min_coverage:
            print(f"Skipping {fy}: only {coverage:.0%} of its days are present.")
            continue
        user_cf = row["user_mwh"] / (system_mw * hrs)
        our_cf = OUR_BENCHMARKS[fy][f"{benchmark_plant}_cf"]
        expected = system_mw * hrs * our_cf
        results.append({
            "Fiscal_Year": fy,
            "User_MWh": round(row["user_mwh"], 4),
            "Expected_at_benchmark_MWh": round(expected, 4),
            "MWh_Difference": round(row["user_mwh"] - expected, 4),
            "User_Capacity_Factor_%": round(user_cf * 100, 2),
            "Benchmark_Capacity_Factor_%": round(our_cf * 100, 2),
            "Difference_vs_benchmark_%": round((user_cf - our_cf) / our_cf * 100, 2),
            "Status": "ABOVE benchmark" if user_cf >= our_cf else "BELOW benchmark",
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # SIMULATED example only (random numbers, not real data): a 100 kW system in Kingston.
    dates = pd.date_range("2021-04-01", "2024-03-31", freq="D")
    np.random.seed(42)
    sample = pd.DataFrame({"timestamp": dates,
                           "net_export_kwh": np.random.normal(640, 75, len(dates)).clip(min=100)})
    print("SIMULATED DATA, for testing the code only")
    print(evaluate_platform_vs_our(sample, 100.0, "content_solar").to_string(index=False))
