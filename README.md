# Solar Predict JA

Forecasts a solar system's next-day energy (kWh) from the weather forecast for a utility
planner, and tests on real measured output (NREL PVDAQ system 1332) whether it beats yesterday's
output and the standard solar formula.

## Install
    pip install -r requirements.txt

## Run
1. Download the daily files for system 1332 into `data/raw/` (see `docs/data_notes.md`).
2. Check the TODO lines in `config.py`.
3. `python -m modules.prepare`   (daily kWh, forecasts, baselines)
   `python -m modules.train`     (Random Forest, skill scores, saved model)
   `python -m modules.report`    (charts in docs/figures, summary in docs/results_summary.md)
4. `python tests/test_modules.py`   (checks)
5. `streamlit run app.py`   (after the app pages are built)

## Data sources
- NREL PVDAQ (OEDI): https://data.openei.org/submissions/4568
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api

## Not doing
Battery sizing, off-grid design, JPS bills and tariffs, several sites, hourly forecasts,
live inverter data, homeowner accounts.

## Limits
See `docs/data_notes.md`. The Jamaica page is not validated.
