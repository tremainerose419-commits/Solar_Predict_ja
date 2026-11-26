# Data notes

## Measured output
- Source: NREL PVDAQ public datasets (OEDI data lake), system 1332, NREL Parking Garage, Golden, CO
- Listed size: 1,153.488 kW. Coordinates: 39.7388, -105.1732
- PVDAQ record: 2013-03-29 to 2022-01-18. Window used: 2021-01-01 to 2022-01-18
- Files: 355 daily files downloaded for the window (`python -m modules.download`)
- Column used: `metered_ac_power__2638` (the site total). Individual meters and inverters are never added to it, to avoid counting energy twice
- Units: watts. Checked from a sunny-day peak of about 500,700 (impossible as kW for this plant) and the highest reading in the window, 1,084 kW, which is close to the listed 1,153 kW
- Readings: every 15 seconds. Daily kWh = power x time, negative night readings set to zero
- Timestamps: fixed UTC-7 clock (no daylight saving), set as `Etc/GMT+7`. Checked with the midpoint of production hours by month
- Known issue: from June the production midpoint shifts by about 2 hours. Cause unknown. Daily totals are unaffected because there is no production at midnight
- Completeness rule: keep days with at least 90% of readings. 354 days kept
- Scored days: 324 (243 train, 81 test) after dropping days missing any input
- Train period: 2021-01-04 to 2021-10-06. Test period: 2021-10-07 to 2022-01-17
- Array type (fixed or tracking): not checked

## Plant equipment varied during the year
The site peak depended on which inverters and meters were producing (median of each day's highest reading):

| Months | Producing | Site peak (kW) |
|---|---|---|
| Jan, Feb | Both inverters | about 660 to 720 |
| Mar to Aug | Inverter 2 only | about 290 to 485 |
| Sep | Everything | about 705 |
| Oct, Nov | Inverter 1 only | about 450 to 520 |
| Dec | Both inverters | about 545 |

We do not know why (fault, maintenance or reconfiguration). Because of this, the model also uses recent peak power
(yesterday's peak and the highest peak of the last 7 days) as inputs. Both use earlier days only.

## Weather forecasts
- Source: Open-Meteo Historical Forecast API, same coordinates, hourly, saved to `data/demo/forecast_daily.csv`
- Variables: shortwave radiation (summed to kWh/m2/day), cloud cover, temperature
- Clear-sky sunlight (pvlib, altitude 1,770 m) is calculated from the sun's position, so it is known in advance
- Limit: archived forecast model outputs, not strictly forecasts issued 24 hours ahead. Actual weather is never used as an input

## Method
- Target: measured kWh, never a formula. Inputs are known the day before (a test proves a day's own values never enter its inputs)
- Split: earlier 75% of days train, latest 25% test, never shuffled. The input set was chosen using the last 25% of the training days only
- Baselines: persistence; physics formula (size x forecast sunlight x 0.8); physics formula with a ratio fitted on training days (0.42)
- Model: Random Forest, 300 trees, minimum 5 days per leaf, fixed random seed
- Scores: MAE, RMSE, MAE as a percentage of the average day, and skill = 1 - model MAE / baseline MAE

## Disclosure
- The first model scored worse than the baselines (MAE 915.8 kWh against 707.7 for persistence). We then diagnosed the equipment changes above and added the peak-power inputs and the fitted-ratio baseline. We saw the first test results before making that change. The change came from diagnosing the data, not from searching for a score
- Final test result: Random Forest MAE 687.0 kWh against 707.7 (persistence) and 737.8 (physics, 0.8). The gain is small on 81 days, so we do not claim the model is proven better

## Limits
- Short window (about one year); test days fall in autumn and winter, when the equipment mix differs from most training days
- The model relies mostly on the plant's recent state (yesterday's output and peak power), not on weather alone
- US data proves the method; the Jamaica page is a labelled illustration, simulated from weather and not validated
- Jamaica benchmark figures were supplied by the group and are unverified until checked against the OUR annual reports
