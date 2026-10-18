# AI weather system — hold-out evaluation on public datasets

Config: horizons=7, ensemble=5, hidden=[32, 16], epochs<=200, seed=42

### Seattle-Tacoma daily 2012-2015 — test year 2014 (train 2012-13)

Train: 2012-01-01 .. 2013-12-31 (731 days) · Test: 2014-01-01 .. 2014-12-31 (365 days) · gap-filled days in source: 0

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmax_c | 1d | 2.188 | 2.167 | 2.933 | 2.276 | 2.797 | 3.696 | 2.955 | +0.427 |
| tmin_c | 1d | 1.540 | 1.397 | 2.179 | 1.463 | 2.036 | 2.896 | 1.912 | +0.506 |
| wind_ms | 1d | 1.011 | 1.032 | 1.069 | 1.181 | 1.300 | 1.399 | 1.516 | +0.137 |
| precip_mm | 1d | 3.293 | 3.390 | 3.488 | 4.245 | 6.760 | 7.010 | 8.294 | +0.070 |
| tmax_c | 2d | 2.706 | 2.724 | 2.933 | 3.070 | 3.483 | 3.696 | 4.034 | +0.112 |
| tmin_c | 2d | 1.916 | 1.889 | 2.179 | 1.970 | 2.589 | 2.896 | 2.679 | +0.201 |
| wind_ms | 2d | 1.074 | 1.097 | 1.069 | 1.353 | 1.393 | 1.399 | 1.804 | +0.009 |
| precip_mm | 2d | 3.425 | 3.621 | 3.488 | 4.686 | 6.939 | 7.010 | 8.772 | +0.020 |
| tmax_c | 3d | 2.858 | 2.955 | 2.933 | 3.570 | 3.658 | 3.696 | 4.657 | +0.021 |
| tmin_c | 3d | 2.090 | 2.131 | 2.179 | 2.315 | 2.785 | 2.896 | 3.204 | +0.075 |
| wind_ms | 3d | 1.085 | 1.121 | 1.069 | 1.434 | 1.420 | 1.399 | 1.887 | -0.029 |
| precip_mm | 3d | 3.470 | 3.720 | 3.488 | 4.733 | 7.009 | 7.010 | 8.652 | +0.000 |
| tmax_c | 4d | 2.907 | 3.061 | 2.933 | 3.888 | 3.685 | 3.696 | 4.945 | +0.006 |
| tmin_c | 4d | 2.181 | 2.212 | 2.179 | 2.620 | 2.898 | 2.896 | 3.583 | -0.001 |
| wind_ms | 4d | 1.084 | 1.149 | 1.069 | 1.429 | 1.414 | 1.399 | 1.909 | -0.022 |
| precip_mm | 4d | 3.465 | 3.658 | 3.488 | 5.200 | 7.000 | 7.010 | 9.344 | +0.003 |
| tmax_c | 5d | 2.914 | 3.107 | 2.933 | 3.951 | 3.700 | 3.696 | 5.012 | -0.003 |
| tmin_c | 5d | 2.188 | 2.229 | 2.179 | 2.761 | 2.896 | 2.896 | 3.813 | -0.000 |
| wind_ms | 5d | 1.090 | 1.192 | 1.069 | 1.495 | 1.416 | 1.399 | 1.958 | -0.024 |
| precip_mm | 5d | 3.481 | 3.647 | 3.488 | 5.120 | 7.012 | 7.010 | 9.079 | -0.001 |
| tmax_c | 6d | 2.956 | 3.131 | 2.933 | 3.961 | 3.728 | 3.696 | 5.034 | -0.018 |
| tmin_c | 6d | 2.200 | 2.220 | 2.179 | 2.874 | 2.933 | 2.896 | 3.990 | -0.026 |
| wind_ms | 6d | 1.075 | 1.169 | 1.069 | 1.489 | 1.412 | 1.399 | 1.957 | -0.018 |
| precip_mm | 6d | 3.503 | 3.706 | 3.488 | 5.094 | 7.009 | 7.010 | 9.140 | +0.000 |
| tmax_c | 7d | 2.924 | 3.155 | 2.933 | 3.967 | 3.652 | 3.696 | 5.093 | +0.023 |
| tmin_c | 7d | 2.185 | 2.239 | 2.179 | 2.977 | 2.896 | 2.896 | 4.173 | -0.000 |
| wind_ms | 7d | 1.068 | 1.141 | 1.069 | 1.486 | 1.398 | 1.399 | 1.976 | +0.002 |
| precip_mm | 7d | 3.506 | 3.686 | 3.488 | 5.267 | 6.968 | 7.010 | 9.414 | +0.012 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Rain / no-rain probability (precip > 0)

| Lead | Brier AI (blend) | Brier monthly clim. | Brier persistence | AUC AI (blend) | AUC clim. | Log-loss AI (blend) |
|---|---|---|---|---|---|---|
| 1d | 0.189 | 0.222 | 0.260 | 0.773 | 0.671 | 0.562 |
| 2d | 0.216 | 0.222 | 0.351 | 0.700 | 0.671 | 0.623 |
| 3d | 0.228 | 0.222 | 0.367 | 0.650 | 0.671 | 0.647 |
| 4d | 0.221 | 0.222 | 0.392 | 0.670 | 0.671 | 0.631 |
| 5d | 0.222 | 0.222 | 0.436 | 0.654 | 0.671 | 0.633 |
| 6d | 0.224 | 0.222 | 0.447 | 0.648 | 0.671 | 0.639 |
| 7d | 0.222 | 0.222 | 0.414 | 0.659 | 0.671 | 0.633 |

#### Next-condition forecast (drizzle/fog/rain/snow/sun)

| Lead | Acc AI-MLP | Acc persistence | Acc majority | Macro-F1 AI-MLP | Macro-F1 persistence |
|---|---|---|---|---|---|
| 1d | 0.668 | 0.671 | 0.405 | 0.349 | 0.439 |
| 2d | 0.641 | 0.578 | 0.405 | 0.334 | 0.342 |
| 3d | 0.616 | 0.562 | 0.405 | 0.323 | 0.333 |
| 4d | 0.592 | 0.529 | 0.405 | 0.311 | 0.300 |
| 5d | 0.575 | 0.488 | 0.405 | 0.302 | 0.269 |
| 6d | 0.586 | 0.466 | 0.405 | 0.305 | 0.250 |
| 7d | 0.597 | 0.493 | 0.405 | 0.311 | 0.289 |

#### Current-condition detection from same-day readings

Accuracy: AI-MLP **0.868**, majority-class 0.405, rule (precip>0 → rain else sun) 0.918 · Macro-F1: AI-MLP 0.658, rule 0.481

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| drizzle | 0.000 | 0.000 | 0.000 | 0 |
| fog | 0.000 | 0.000 | 0.000 | 28 |
| rain | 0.886 | 1.000 | 0.940 | 148 |
| snow | 0.667 | 1.000 | 0.800 | 2 |
| sun | 0.888 | 0.893 | 0.891 | 187 |

Confusion matrix (rows = truth, cols = prediction; order drizzle, fog, rain, snow, sun):

```
   0    0    0    0    0
   4    0    3    0   21
   0    0  148    0    0
   0    0    0    2    0
   3    0   16    1  167
```

#### Sensor-fault / anomaly detection on `tmax_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.43, σ(climatology anomaly)=3.41

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.926 | 0.833 | 0.673 / 0.446 | 0.437 / 0.185 |
| 3σ | 0.988 | 0.954 | 0.788 / 0.876 | 0.675 / 0.494 |
| 4σ | 0.999 | 0.991 | 0.814 / 0.987 | 0.780 / 0.816 |
| 6σ | 1.000 | 1.000 | 0.814 / 1.000 | 0.807 / 0.993 |

### Seattle-Tacoma daily 2012-2015 — test year 2015 (train 2012-14)

Train: 2012-01-01 .. 2014-12-31 (1096 days) · Test: 2015-01-01 .. 2015-12-31 (365 days) · gap-filled days in source: 0

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmax_c | 1d | 2.271 | 2.186 | 3.124 | 2.240 | 2.876 | 3.912 | 2.907 | +0.459 |
| tmin_c | 1d | 1.397 | 1.383 | 2.086 | 1.492 | 1.761 | 2.671 | 1.956 | +0.566 |
| wind_ms | 1d | 1.008 | 1.008 | 1.068 | 1.108 | 1.313 | 1.379 | 1.452 | +0.094 |
| precip_mm | 1d | 3.145 | 3.124 | 3.272 | 3.799 | 7.340 | 7.777 | 8.331 | +0.109 |
| tmax_c | 2d | 2.858 | 2.797 | 3.124 | 3.132 | 3.548 | 3.912 | 3.929 | +0.177 |
| tmin_c | 2d | 1.759 | 1.755 | 2.086 | 2.025 | 2.194 | 2.671 | 2.598 | +0.325 |
| wind_ms | 2d | 1.110 | 1.163 | 1.068 | 1.377 | 1.432 | 1.379 | 1.762 | -0.077 |
| precip_mm | 2d | 3.261 | 3.276 | 3.272 | 4.384 | 7.658 | 7.777 | 9.687 | +0.030 |
| tmax_c | 3d | 3.042 | 3.018 | 3.124 | 3.473 | 3.791 | 3.912 | 4.424 | +0.061 |
| tmin_c | 3d | 1.917 | 1.923 | 2.086 | 2.337 | 2.400 | 2.671 | 3.003 | +0.192 |
| wind_ms | 3d | 1.096 | 1.173 | 1.068 | 1.356 | 1.422 | 1.379 | 1.798 | -0.064 |
| precip_mm | 3d | 3.243 | 3.303 | 3.272 | 4.734 | 7.720 | 7.777 | 10.109 | +0.014 |
| tmax_c | 4d | 3.044 | 3.076 | 3.124 | 3.650 | 3.825 | 3.912 | 4.700 | +0.044 |
| tmin_c | 4d | 1.972 | 1.952 | 2.086 | 2.533 | 2.476 | 2.671 | 3.149 | +0.141 |
| wind_ms | 4d | 1.112 | 1.172 | 1.068 | 1.364 | 1.449 | 1.379 | 1.845 | -0.103 |
| precip_mm | 4d | 3.259 | 3.292 | 3.272 | 4.915 | 7.770 | 7.777 | 10.393 | +0.002 |
| tmax_c | 5d | 3.082 | 3.113 | 3.124 | 3.696 | 3.847 | 3.912 | 4.796 | +0.033 |
| tmin_c | 5d | 2.021 | 1.990 | 2.086 | 2.572 | 2.551 | 2.671 | 3.224 | +0.088 |
| wind_ms | 5d | 1.135 | 1.198 | 1.068 | 1.402 | 1.470 | 1.379 | 1.870 | -0.137 |
| precip_mm | 5d | 3.257 | 3.281 | 3.272 | 4.951 | 7.752 | 7.777 | 10.534 | +0.006 |
| tmax_c | 6d | 3.124 | 3.182 | 3.124 | 3.736 | 3.897 | 3.912 | 4.743 | +0.007 |
| tmin_c | 6d | 2.039 | 2.010 | 2.086 | 2.511 | 2.600 | 2.671 | 3.264 | +0.052 |
| wind_ms | 6d | 1.119 | 1.194 | 1.068 | 1.357 | 1.457 | 1.379 | 1.819 | -0.115 |
| precip_mm | 6d | 3.244 | 3.291 | 3.272 | 5.075 | 7.757 | 7.777 | 10.624 | +0.005 |
| tmax_c | 7d | 3.141 | 3.189 | 3.124 | 3.740 | 3.922 | 3.912 | 4.706 | -0.006 |
| tmin_c | 7d | 2.028 | 2.062 | 2.086 | 2.559 | 2.582 | 2.671 | 3.261 | +0.066 |
| wind_ms | 7d | 1.121 | 1.206 | 1.068 | 1.405 | 1.446 | 1.379 | 1.831 | -0.099 |
| precip_mm | 7d | 3.253 | 3.273 | 3.272 | 4.962 | 7.740 | 7.777 | 10.606 | +0.009 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Rain / no-rain probability (precip > 0)

| Lead | Brier AI (blend) | Brier monthly clim. | Brier persistence | AUC AI (blend) | AUC clim. | Log-loss AI (blend) |
|---|---|---|---|---|---|---|
| 1d | 0.190 | 0.203 | 0.296 | 0.766 | 0.734 | 0.563 |
| 2d | 0.207 | 0.203 | 0.356 | 0.719 | 0.734 | 0.602 |
| 3d | 0.203 | 0.203 | 0.384 | 0.728 | 0.734 | 0.595 |
| 4d | 0.203 | 0.203 | 0.384 | 0.747 | 0.734 | 0.593 |
| 5d | 0.204 | 0.203 | 0.395 | 0.736 | 0.734 | 0.596 |
| 6d | 0.203 | 0.203 | 0.384 | 0.739 | 0.734 | 0.591 |
| 7d | 0.202 | 0.203 | 0.348 | 0.744 | 0.734 | 0.589 |

#### Next-condition forecast (drizzle/fog/rain/snow/sun)

| Lead | Acc AI-MLP | Acc persistence | Acc majority | Macro-F1 AI-MLP | Macro-F1 persistence |
|---|---|---|---|---|---|
| 1d | 0.625 | 0.600 | 0.395 | 0.339 | 0.442 |
| 2d | 0.600 | 0.523 | 0.395 | 0.325 | 0.387 |
| 3d | 0.603 | 0.488 | 0.395 | 0.327 | 0.353 |
| 4d | 0.586 | 0.493 | 0.395 | 0.318 | 0.359 |
| 5d | 0.595 | 0.479 | 0.395 | 0.321 | 0.330 |
| 6d | 0.589 | 0.501 | 0.395 | 0.317 | 0.336 |
| 7d | 0.564 | 0.532 | 0.395 | 0.303 | 0.354 |

#### Current-condition detection from same-day readings

Accuracy: AI-MLP **0.825**, majority-class 0.395, rule (precip>0 → rain else sun) 0.838 · Macro-F1: AI-MLP 0.455, rule 0.461

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| drizzle | 0.000 | 0.000 | 0.000 | 7 |
| fog | 0.000 | 0.000 | 0.000 | 52 |
| rain | 0.979 | 0.979 | 0.979 | 144 |
| snow | 0.000 | 0.000 | 0.000 | 0 |
| sun | 0.734 | 0.988 | 0.842 | 162 |

Confusion matrix (rows = truth, cols = prediction; order drizzle, fog, rain, snow, sun):

```
   0    0    0    0    7
   0    0    1    0   51
   0    0  141    3    0
   0    0    0    0    0
   0    0    2    0  160
```

#### Sensor-fault / anomaly detection on `tmax_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.63, σ(climatology anomaly)=3.48

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.923 | 0.814 | 0.740 / 0.342 | 0.967 / 0.143 |
| 3σ | 0.989 | 0.934 | 0.871 / 0.807 | 1.000 / 0.495 |
| 4σ | 0.999 | 0.985 | 0.895 / 0.983 | 1.000 / 0.825 |
| 6σ | 1.000 | 1.000 | 0.891 / 1.000 | 1.000 / 1.000 |

### New York daily 2012-2015 — test year 2014 (train 2012-13)

Train: 2012-01-01 .. 2013-12-31 (731 days) · Test: 2014-01-01 .. 2014-12-31 (365 days) · gap-filled days in source: 0

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmax_c | 1d | 2.840 | 2.787 | 3.404 | 3.092 | 3.584 | 4.344 | 4.114 | +0.319 |
| tmin_c | 1d | 2.266 | 2.131 | 2.959 | 2.318 | 3.067 | 3.990 | 3.182 | +0.409 |
| wind_ms | 1d | 1.372 | 1.379 | 1.460 | 1.579 | 1.795 | 1.859 | 2.105 | +0.068 |
| precip_mm | 1d | 3.791 | 3.779 | 3.800 | 6.083 | 11.337 | 11.370 | 15.371 | +0.006 |
| tmax_c | 2d | 3.402 | 3.446 | 3.404 | 4.103 | 4.305 | 4.344 | 5.389 | +0.018 |
| tmin_c | 2d | 2.798 | 2.868 | 2.959 | 3.262 | 3.783 | 3.990 | 4.279 | +0.101 |
| wind_ms | 2d | 1.465 | 1.512 | 1.460 | 2.036 | 1.891 | 1.859 | 2.612 | -0.034 |
| precip_mm | 2d | 3.831 | 3.845 | 3.800 | 6.544 | 11.386 | 11.370 | 15.984 | -0.003 |
| tmax_c | 3d | 3.414 | 3.552 | 3.404 | 4.583 | 4.369 | 4.344 | 5.780 | -0.012 |
| tmin_c | 3d | 2.878 | 3.097 | 2.959 | 3.524 | 3.872 | 3.990 | 4.552 | +0.058 |
| wind_ms | 3d | 1.465 | 1.516 | 1.460 | 2.014 | 1.886 | 1.859 | 2.618 | -0.029 |
| precip_mm | 3d | 3.835 | 3.857 | 3.800 | 6.608 | 11.387 | 11.370 | 15.894 | -0.003 |
| tmax_c | 4d | 3.432 | 3.637 | 3.404 | 4.744 | 4.380 | 4.344 | 6.054 | -0.016 |
| tmin_c | 4d | 2.925 | 3.205 | 2.959 | 3.746 | 3.942 | 3.990 | 4.821 | +0.024 |
| wind_ms | 4d | 1.459 | 1.508 | 1.460 | 2.003 | 1.877 | 1.859 | 2.574 | -0.019 |
| precip_mm | 4d | 3.840 | 3.862 | 3.800 | 6.349 | 11.362 | 11.370 | 15.533 | +0.001 |
| tmax_c | 5d | 3.384 | 3.645 | 3.404 | 4.730 | 4.340 | 4.344 | 6.005 | +0.002 |
| tmin_c | 5d | 2.921 | 3.163 | 2.959 | 3.721 | 3.956 | 3.990 | 4.899 | +0.017 |
| wind_ms | 5d | 1.452 | 1.502 | 1.460 | 2.005 | 1.878 | 1.859 | 2.598 | -0.020 |
| precip_mm | 5d | 3.850 | 3.888 | 3.800 | 6.550 | 11.371 | 11.370 | 16.125 | -0.000 |
| tmax_c | 6d | 3.418 | 3.574 | 3.404 | 4.517 | 4.379 | 4.344 | 5.740 | -0.016 |
| tmin_c | 6d | 2.907 | 3.135 | 2.959 | 3.714 | 3.953 | 3.990 | 4.926 | +0.019 |
| wind_ms | 6d | 1.450 | 1.507 | 1.460 | 2.035 | 1.880 | 1.859 | 2.609 | -0.022 |
| precip_mm | 6d | 3.854 | 3.893 | 3.800 | 6.603 | 11.359 | 11.370 | 15.989 | +0.002 |
| tmax_c | 7d | 3.386 | 3.598 | 3.404 | 4.357 | 4.340 | 4.344 | 5.638 | +0.002 |
| tmin_c | 7d | 2.927 | 3.153 | 2.959 | 3.840 | 3.968 | 3.990 | 5.037 | +0.011 |
| wind_ms | 7d | 1.437 | 1.518 | 1.460 | 2.092 | 1.860 | 1.859 | 2.707 | -0.000 |
| precip_mm | 7d | 3.817 | 3.855 | 3.800 | 6.368 | 11.352 | 11.370 | 15.957 | +0.003 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Rain / no-rain probability (precip > 0)

| Lead | Brier AI (blend) | Brier monthly clim. | Brier persistence | AUC AI (blend) | AUC clim. | Log-loss AI (blend) |
|---|---|---|---|---|---|---|
| 1d | 0.222 | 0.224 | 0.373 | 0.571 | 0.538 | 0.635 |
| 2d | 0.224 | 0.224 | 0.510 | 0.538 | 0.538 | 0.641 |
| 3d | 0.224 | 0.224 | 0.496 | 0.538 | 0.538 | 0.641 |
| 4d | 0.224 | 0.224 | 0.449 | 0.538 | 0.538 | 0.641 |
| 5d | 0.224 | 0.224 | 0.466 | 0.538 | 0.538 | 0.641 |
| 6d | 0.224 | 0.224 | 0.449 | 0.538 | 0.538 | 0.641 |
| 7d | 0.224 | 0.224 | 0.392 | 0.538 | 0.538 | 0.641 |

#### Next-condition forecast (drizzle/fog/rain/snow/sun)

| Lead | Acc AI-MLP | Acc persistence | Acc majority | Macro-F1 AI-MLP | Macro-F1 persistence |
|---|---|---|---|---|---|
| 1d | 0.567 | 0.567 | 0.622 | 0.251 | 0.282 |
| 2d | 0.504 | 0.438 | 0.622 | 0.195 | 0.184 |
| 3d | 0.504 | 0.430 | 0.622 | 0.199 | 0.207 |
| 4d | 0.515 | 0.466 | 0.622 | 0.228 | 0.234 |
| 5d | 0.515 | 0.460 | 0.622 | 0.213 | 0.213 |
| 6d | 0.482 | 0.463 | 0.622 | 0.207 | 0.201 |
| 7d | 0.477 | 0.512 | 0.622 | 0.203 | 0.240 |

#### Current-condition detection from same-day readings

Accuracy: AI-MLP **0.923**, majority-class 0.622, rule (precip>0 → rain else sun) 0.901 · Macro-F1: AI-MLP 0.536, rule 0.375

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| drizzle | 0.000 | 0.000 | 0.000 | 3 |
| fog | 0.000 | 0.000 | 0.000 | 13 |
| rain | 0.970 | 0.951 | 0.960 | 102 |
| snow | 0.667 | 0.900 | 0.766 | 20 |
| sun | 0.933 | 0.978 | 0.955 | 227 |

Confusion matrix (rows = truth, cols = prediction; order drizzle, fog, rain, snow, sun):

```
   0    0    0    0    3
   0    0    0    0   13
   0    0   97    5    0
   0    0    2   18    0
   0    0    1    4  222
```

#### Sensor-fault / anomaly detection on `tmax_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.84, σ(climatology anomaly)=3.85

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.868 | 0.823 | 0.577 / 0.391 | 0.393 / 0.188 |
| 3σ | 0.977 | 0.940 | 0.742 / 0.813 | 0.635 / 0.514 |
| 4σ | 0.999 | 0.991 | 0.771 / 0.979 | 0.745 / 0.838 |
| 6σ | 1.000 | 1.000 | 0.777 / 1.000 | 0.776 / 0.987 |

### New York daily 2012-2015 — test year 2015 (train 2012-14)

Train: 2012-01-01 .. 2014-12-31 (1096 days) · Test: 2015-01-01 .. 2015-12-31 (365 days) · gap-filled days in source: 0

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmax_c | 1d | 2.607 | 2.435 | 3.520 | 2.929 | 3.309 | 4.504 | 3.812 | +0.460 |
| tmin_c | 1d | 2.223 | 2.044 | 3.187 | 2.332 | 2.955 | 4.315 | 3.123 | +0.531 |
| wind_ms | 1d | 1.320 | 1.323 | 1.387 | 1.670 | 1.701 | 1.748 | 2.178 | +0.053 |
| precip_mm | 1d | 2.964 | 3.001 | 3.004 | 4.510 | 7.749 | 7.786 | 10.633 | +0.009 |
| tmax_c | 2d | 3.099 | 3.044 | 3.520 | 3.559 | 4.006 | 4.504 | 4.600 | +0.209 |
| tmin_c | 2d | 2.693 | 2.640 | 3.187 | 3.181 | 3.611 | 4.315 | 4.118 | +0.300 |
| wind_ms | 2d | 1.383 | 1.404 | 1.387 | 1.873 | 1.741 | 1.748 | 2.395 | +0.008 |
| precip_mm | 2d | 3.049 | 3.073 | 3.004 | 4.731 | 7.788 | 7.786 | 10.723 | -0.001 |
| tmax_c | 3d | 3.233 | 3.186 | 3.520 | 3.808 | 4.144 | 4.504 | 4.935 | +0.154 |
| tmin_c | 3d | 2.878 | 2.810 | 3.187 | 3.366 | 3.878 | 4.315 | 4.335 | +0.192 |
| wind_ms | 3d | 1.399 | 1.419 | 1.387 | 1.822 | 1.769 | 1.748 | 2.377 | -0.025 |
| precip_mm | 3d | 3.019 | 3.040 | 3.004 | 4.882 | 7.795 | 7.786 | 10.988 | -0.002 |
| tmax_c | 4d | 3.306 | 3.209 | 3.520 | 4.088 | 4.268 | 4.504 | 5.228 | +0.102 |
| tmin_c | 4d | 2.962 | 2.921 | 3.187 | 3.555 | 3.935 | 4.315 | 4.544 | +0.169 |
| wind_ms | 4d | 1.398 | 1.432 | 1.387 | 1.868 | 1.774 | 1.748 | 2.362 | -0.030 |
| precip_mm | 4d | 3.006 | 3.029 | 3.004 | 4.889 | 7.787 | 7.786 | 10.988 | -0.000 |
| tmax_c | 5d | 3.362 | 3.314 | 3.520 | 4.218 | 4.348 | 4.504 | 5.333 | +0.068 |
| tmin_c | 5d | 3.041 | 3.017 | 3.187 | 3.723 | 4.069 | 4.315 | 4.710 | +0.111 |
| wind_ms | 5d | 1.400 | 1.440 | 1.387 | 1.941 | 1.770 | 1.748 | 2.443 | -0.026 |
| precip_mm | 5d | 3.005 | 3.045 | 3.004 | 4.913 | 7.790 | 7.786 | 11.026 | -0.001 |
| tmax_c | 6d | 3.375 | 3.360 | 3.520 | 4.135 | 4.326 | 4.504 | 5.242 | +0.078 |
| tmin_c | 6d | 3.039 | 3.082 | 3.187 | 3.776 | 4.059 | 4.315 | 4.782 | +0.115 |
| wind_ms | 6d | 1.388 | 1.395 | 1.387 | 1.921 | 1.760 | 1.748 | 2.462 | -0.014 |
| precip_mm | 6d | 3.016 | 3.051 | 3.004 | 4.294 | 7.787 | 7.786 | 10.099 | -0.000 |
| tmax_c | 7d | 3.421 | 3.391 | 3.520 | 4.044 | 4.361 | 4.504 | 5.058 | +0.062 |
| tmin_c | 7d | 3.072 | 3.098 | 3.187 | 3.739 | 4.149 | 4.315 | 4.688 | +0.076 |
| wind_ms | 7d | 1.387 | 1.385 | 1.387 | 1.852 | 1.752 | 1.748 | 2.416 | -0.005 |
| precip_mm | 7d | 3.002 | 3.016 | 3.004 | 4.634 | 7.774 | 7.786 | 10.744 | +0.003 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Rain / no-rain probability (precip > 0)

| Lead | Brier AI (blend) | Brier monthly clim. | Brier persistence | AUC AI (blend) | AUC clim. | Log-loss AI (blend) |
|---|---|---|---|---|---|---|
| 1d | 0.207 | 0.210 | 0.321 | 0.598 | 0.569 | 0.604 |
| 2d | 0.212 | 0.210 | 0.405 | 0.558 | 0.569 | 0.614 |
| 3d | 0.210 | 0.210 | 0.386 | 0.569 | 0.569 | 0.611 |
| 4d | 0.210 | 0.210 | 0.408 | 0.569 | 0.569 | 0.611 |
| 5d | 0.210 | 0.210 | 0.427 | 0.569 | 0.569 | 0.611 |
| 6d | 0.210 | 0.210 | 0.375 | 0.569 | 0.569 | 0.611 |
| 7d | 0.210 | 0.210 | 0.370 | 0.569 | 0.569 | 0.611 |

#### Next-condition forecast (drizzle/fog/rain/snow/sun)

| Lead | Acc AI-MLP | Acc persistence | Acc majority | Macro-F1 AI-MLP | Macro-F1 persistence |
|---|---|---|---|---|---|
| 1d | 0.581 | 0.562 | 0.600 | 0.260 | 0.331 |
| 2d | 0.523 | 0.468 | 0.600 | 0.226 | 0.282 |
| 3d | 0.545 | 0.468 | 0.600 | 0.196 | 0.282 |
| 4d | 0.540 | 0.458 | 0.600 | 0.193 | 0.258 |
| 5d | 0.573 | 0.436 | 0.600 | 0.222 | 0.217 |
| 6d | 0.581 | 0.485 | 0.600 | 0.223 | 0.274 |
| 7d | 0.559 | 0.496 | 0.600 | 0.204 | 0.278 |

#### Current-condition detection from same-day readings

Accuracy: AI-MLP **0.893**, majority-class 0.600, rule (precip>0 → rain else sun) 0.849 · Macro-F1: AI-MLP 0.572, rule 0.366

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| drizzle | 0.500 | 0.043 | 0.080 | 23 |
| fog | 0.000 | 0.000 | 0.000 | 12 |
| rain | 0.978 | 0.978 | 0.978 | 91 |
| snow | 0.895 | 0.850 | 0.872 | 20 |
| sun | 0.866 | 1.000 | 0.928 | 219 |

Confusion matrix (rows = truth, cols = prediction; order drizzle, fog, rain, snow, sun):

```
   1    0    0    0   22
   1    0    0    0   11
   0    0   89    2    0
   0    0    2   17    1
   0    0    0    0  219
```

#### Sensor-fault / anomaly detection on `tmax_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.82, σ(climatology anomaly)=3.98

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.910 | 0.801 | 0.646 / 0.430 | 0.479 / 0.170 |
| 3σ | 0.989 | 0.947 | 0.784 / 0.839 | 0.746 / 0.527 |
| 4σ | 1.000 | 0.987 | 0.812 / 0.986 | 0.821 / 0.803 |
| 6σ | 1.000 | 1.000 | 0.815 / 1.000 | 0.849 / 0.996 |

### Melbourne daily min temperature 1981-1990 — test year 1989 (train 1981-88)

Train: 1981-01-01 .. 1988-12-31 (2922 days) · Test: 1989-01-01 .. 1989-12-31 (365 days) · gap-filled days in source: 2

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmin_c | 1d | 1.667 | 1.704 | 2.027 | 1.880 | 2.069 | 2.557 | 2.375 | +0.345 |
| tmin_c | 2d | 1.950 | 1.946 | 2.027 | 2.427 | 2.464 | 2.557 | 3.062 | +0.072 |
| tmin_c | 3d | 1.985 | 1.998 | 2.027 | 2.602 | 2.498 | 2.557 | 3.305 | +0.046 |
| tmin_c | 4d | 2.006 | 2.007 | 2.027 | 2.742 | 2.523 | 2.557 | 3.441 | +0.027 |
| tmin_c | 5d | 2.030 | 2.019 | 2.027 | 2.849 | 2.547 | 2.557 | 3.495 | +0.008 |
| tmin_c | 6d | 2.001 | 2.023 | 2.027 | 2.804 | 2.533 | 2.557 | 3.466 | +0.019 |
| tmin_c | 7d | 1.992 | 2.024 | 2.027 | 2.753 | 2.506 | 2.557 | 3.431 | +0.040 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Sensor-fault / anomaly detection on `tmin_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.35, σ(climatology anomaly)=2.79

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.951 | 0.891 | 0.768 / 0.214 | 0.630 / 0.134 |
| 3σ | 0.996 | 0.983 | 0.921 / 0.737 | 0.891 / 0.529 |
| 4σ | 1.000 | 0.999 | 0.943 / 0.987 | 0.935 / 0.857 |
| 6σ | 1.000 | 1.000 | 0.942 / 1.000 | 0.942 / 1.000 |

### Melbourne daily min temperature 1981-1990 — test year 1990 (train 1981-89)

Train: 1981-01-01 .. 1989-12-31 (3287 days) · Test: 1990-01-01 .. 1990-12-31 (365 days) · gap-filled days in source: 2

#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)

| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |
|---|---|---|---|---|---|---|---|---|---|
| tmin_c | 1d | 1.699 | 1.720 | 1.956 | 2.025 | 2.192 | 2.542 | 2.582 | +0.257 |
| tmin_c | 2d | 1.949 | 1.947 | 1.956 | 2.628 | 2.533 | 2.542 | 3.373 | +0.007 |
| tmin_c | 3d | 1.984 | 1.964 | 1.956 | 2.835 | 2.576 | 2.542 | 3.585 | -0.027 |
| tmin_c | 4d | 1.979 | 1.958 | 1.956 | 2.767 | 2.566 | 2.542 | 3.500 | -0.019 |
| tmin_c | 5d | 1.979 | 1.955 | 1.956 | 2.767 | 2.558 | 2.542 | 3.473 | -0.012 |
| tmin_c | 6d | 2.003 | 1.955 | 1.956 | 2.836 | 2.591 | 2.542 | 3.529 | -0.039 |
| tmin_c | 7d | 1.985 | 1.960 | 1.956 | 2.776 | 2.575 | 2.542 | 3.557 | -0.026 |

*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*

#### Sensor-fault / anomaly detection on `tmin_c` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)

σ(model residual)=2.22, σ(climatology anomaly)=2.77

| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |
|---|---|---|---|---|
| 2σ | 0.930 | 0.897 | 0.967 / 0.239 | 0.933 / 0.124 |
| 3σ | 0.990 | 0.976 | 1.000 / 0.774 | 1.000 / 0.515 |
| 4σ | 1.000 | 0.997 | 1.000 / 0.970 | 1.000 / 0.846 |
| 6σ | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

