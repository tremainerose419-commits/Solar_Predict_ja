# Results summary (written from your real numbers)

## The data
- 324 scored days: 243 for training (2021-01-04 to 2021-10-06) and 81 for testing (2021-10-07 to 2022-01-17), split by date and never shuffled.
- The system makes about 2238 kWh on an average test day.

## Results on the test days (days the model never saw)

| Method | MAE (kWh/day) | RMSE (kWh/day) | MAE as % of average |
|---|---|---|---|
| Yesterday's output | 707.7 | 1012.1 | 31.6% |
| Formula (0.8 ratio) | 737.8 | 976.8 | 33.0% |
| Formula (fitted ratio) | 917.2 | 1071.9 | 41.0% |
| Random Forest (AI) | 687.0 | 846.1 | 30.7% |

## Skill: does the AI beat the simple methods?

- Against **Yesterday's output** (error 708 kWh): the AI makes **3% fewer errors** (skill +0.03).
- Against **Formula (0.8 ratio)** (error 738 kWh): the AI makes **7% fewer errors** (skill +0.07).
- Against **Formula (fitted ratio)** (error 917 kWh): the AI makes **25% fewer errors** (skill +0.25).

## Say it in plain words

> "On days the model had never seen, our forecast was off by 687 kWh on average, about 31% of a normal day's output (2238 kWh)."

## What the model relies on most

- yesterday_kwh: 0.41
- forecast_cloud_pct: 0.19
- peak_kw_yesterday: 0.16

Input set used: **with peak power** (chosen using training days only). On the test days the other input set scored 916 kWh, shown for transparency.

## Limits to state

- About one year of data, so each season is seen once; the test days fall in autumn and winter.
- Weather inputs are archived forecast model outputs, not forecasts issued exactly 24 hours ahead.
- One system in Colorado. This proves the method, not that it works in Jamaica.
- RMSE (846 kWh) above MAE (687 kWh) means a few days are missed by much more than average.
