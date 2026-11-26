# Solar Predict JA

Forecasts a solar system's next-day energy (kWh) from the weather forecast, for a utility planner.
It is tested on real measured output (NREL PVDAQ system 1332, Golden, Colorado) to see whether a
Random Forest beats two simple baselines: yesterday's output and the standard solar formula.

Group 2, Associate of Science in AI, Amber National AI Lab. Project: Smart Caribbean Utilities, Project E (solar).

## Result (81 held-out test days, 7 Oct 2021 to 17 Jan 2022)

| Method | MAE (kWh/day) | RMSE (kWh/day) |
|---|---|---|
| Yesterday's output (persistence) | 707.7 | 1,012.1 |
| Physics formula (0.8) | 737.8 | 976.8 |
| Physics formula (fitted ratio) | 917.2 | 1,071.9 |
| **Random Forest** | **687.0** | **846.1** |

The Random Forest is slightly better on average error (skill +0.03 against persistence, +0.07 against the formula)
and clearly better on large misses (RMSE). The test period is short, so read this as "slightly better", not proven.
The model also uses recent peak power as an input, because the plant's equipment changed during the year
(see `docs/data_notes.md`).

## Install
    pip install -r requirements.txt

## Run
1. `python -m modules.download`   (downloads the daily files for system 1332 into `data/raw/`; not stored in git)
2. `python tests/test_modules.py` (13 checks)
3. `python -m modules.prepare`    (daily kWh, forecasts, baselines)
4. `python -m modules.train`      (Random Forest, skill scores, saved model)
5. `python -m modules.report`     (charts in `docs/figures`, summary in `docs/results_summary.md`)
6. `python -m modules.diagnose`   (which meters and inverters were producing, optional)
7. `streamlit run app.py`         (Tomorrow, Past performance, Model check, Jamaica demo)

The results in `data/demo` and the trained model are included, so the app runs without step 1.

## Data sources
- NREL PVDAQ (OEDI): https://data.openei.org/submissions/4568
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api

## Not doing
Battery sizing, off-grid design, JPS bills and tariffs, several sites, hourly forecasts,
live inverter data, homeowner accounts.

## Limits
See `docs/data_notes.md`. The Jamaica page is a simulation and is not validated.
