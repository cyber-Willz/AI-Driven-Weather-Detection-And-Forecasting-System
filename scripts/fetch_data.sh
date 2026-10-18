#!/bin/sh
# Re-download the public datasets used for evaluation (copies are already bundled in data/).
set -e
cd "$(dirname "$0")/.."
curl -fsSL -o data/seattle-weather.csv https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv
curl -fsSL -o data/melbourne-daily-min-temperatures.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
curl -fsSL -o data/vega-weather.csv https://raw.githubusercontent.com/vega/vega-datasets/main/data/weather.csv
curl -fsSL -o data/eu-weather-prediction-dataset.csv https://raw.githubusercontent.com/florian-huber/weather_prediction_dataset/main/dataset/weather_prediction_dataset.csv
curl -fsSL -o data/storms.csv https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/dplyr/storms.csv
