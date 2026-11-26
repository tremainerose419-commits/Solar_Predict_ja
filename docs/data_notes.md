# Data notes

## Measured output
- Source: NREL PVDAQ public datasets (OEDI data lake), system 1332, NREL Parking Garage, Golden, CO
- Size: 1,153.488 kW. Coordinates: 39.7388, -105.1732
- PVDAQ record: 2013-03-29 to 2022-01-18. Window used: 2021-01-01 to 2022-01-18
- Columns used: [ fill in: timestamp column, power column(s) ]
- Units: [ W or kW, and how you checked ]
- Timestamps: [ fixed UTC-7 or daylight saving, and what the peak-time check showed ]
- Completeness rule: keep days with at least 90% of readings. Days kept: [ n of n ]
- Array type (fixed or tracking): [ check ]

## Weather forecasts
- Source: Open-Meteo Historical Forecast API, same coordinates, hourly, saved to data/demo/forecast_daily.csv
- Variables: shortwave radiation (summed to kWh/m2/day), cloud cover, temperature
- Limit: archived forecast model outputs, not strictly forecasts issued 24 hours ahead

## Method
- Target: measured kWh, never a formula. Inputs: forecasts only
- Split: earlier 75% of days train, latest 25% test, never shuffled
- Baselines: persistence; physics formula (size x forecast irradiance x 0.8)

## Limits
- Short window (about one year); test days fall in autumn and winter
- US data proves the method; the Jamaica page is a labelled illustration
