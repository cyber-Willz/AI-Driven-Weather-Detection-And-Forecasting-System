# Ablation: neighbour-station + pressure inputs (EU, 8 stations, train 2000-07, test 2008-09)

Same code, seeds and splits; only the extra inputs differ. Skill = 1 - MAE_model / MAE_climatology; mean over stations.

| Metric | Lead | Baseline (own station) | + neighbours/pressure | Change |
|---|---|---|---|---|
| tmax skill | 1d | +0.315 | +0.385 | +0.070 |
| tmax skill | 2d | +0.134 | +0.196 | +0.062 |
| tmax skill | 3d | +0.069 | +0.105 | +0.036 |
| tmax skill | 7d | -0.009 | -0.003 | +0.006 |
| tmin skill | 1d | +0.302 | +0.364 | +0.062 |
| tmin skill | 2d | +0.112 | +0.181 | +0.069 |
| tmin skill | 3d | +0.049 | +0.095 | +0.046 |
| tmin skill | 7d | -0.020 | -0.010 | +0.010 |
| tmax MAE (°C) | 1d | 2.023 | 1.811 | -0.212 |
| tmax MAE (°C) | 2d | 2.575 | 2.392 | -0.184 |
| tmax MAE (°C) | 3d | 2.770 | 2.665 | -0.105 |
| wind MAE (m/s) | 1d | 1.009 | 0.977 | -0.032 |
| wind MAE (m/s) | 2d | 1.141 | 1.125 | -0.016 |
| wind MAE (m/s) | 3d | 1.167 | 1.158 | -0.009 |
| rain Brier (clim 0.238) | 1d | 0.213 | 0.198 | -0.015 |
| rain Brier (clim 0.238) | 2d | 0.235 | 0.224 | -0.011 |
| rain Brier (clim 0.238) | 3d | 0.238 | 0.233 | -0.005 |
| rain AUC | 1d | 0.700 | 0.744 | +0.044 |
| rain AUC | 2d | 0.602 | 0.656 | +0.055 |
| rain AUC | 3d | 0.567 | 0.607 | +0.039 |
| fault AUC at 2σ | 1d | 0.968 | 0.980 | +0.012 |

Stations improved at 1-day tmax MAE: 8 / 8 ; tmin: 8 / 8 ; rain Brier 1d: 8 / 8
