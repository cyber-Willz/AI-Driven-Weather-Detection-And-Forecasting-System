# Data sources (downloaded from public GitHub raw URLs)

| File | Source | Content |
|---|---|---|
| data/seattle-weather.csv | https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv | Seattle-Tacoma daily 2012-01-01..2015-12-31 (1461 days): precipitation, temp_max, temp_min, wind, condition label (drizzle/fog/rain/snow/sun). Complete, no gaps. |
| data/melbourne-daily-min-temperatures.csv | https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv | Melbourne daily minimum temperature 1981-1990 (3650 rows; the two leap days are absent and are linearly interpolated by the loader). |
| data/vega-weather.csv | https://raw.githubusercontent.com/vega/vega-datasets/main/data/weather.csv | Multi-station file; the New York rows are used (2012-01-01..2015-12-31, 1461 days, same variables and condition labels as Seattle; the Seattle rows duplicate seattle-weather.csv). |
| data/eu-weather-prediction-dataset.csv | https://raw.githubusercontent.com/florian-huber/weather_prediction_dataset/main/dataset/weather_prediction_dataset.csv (also Zenodo doi:10.5281/zenodo.7525955) | 18 European ECA&D stations, daily 2000-01-01..2010-01-01 (3654 days), MIT-licensed repo. Underlying data: ECA&D (Klein Tank et al. 2002, Int. J. Climatol. 22, 1441-1453; http://www.ecad.eu, non-commercial research/education use). Units: precipitation in 10 mm (converted to mm by the loader). Invalid entries (≤5% per column) were mean-imputed by the dataset authors. |
| data/storms.csv | https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/dplyr/storms.csv | NOAA HURDAT2-derived Atlantic best tracks, 1975-2024, 20,778 records (693 named systems), position, max wind (kt), pressure (mb), status. Distributed with the R package dplyr (MIT); underlying data NOAA/NHC, public domain. Only synoptic times (00/06/12/18 UTC) are used. |

Open-Meteo / NOAA API hosts were not reachable from the build sandbox (egress allow-list), so these two
static public files were used. Check each source's license before redistributing.
