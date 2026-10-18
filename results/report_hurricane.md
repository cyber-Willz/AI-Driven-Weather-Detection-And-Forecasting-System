# AI weather system — tropical-cyclone track & intensity forecasting

Data: NOAA HURDAT2-derived best tracks (dplyr `storms`), Atlantic, 6-hourly, 656 storm segments, 1975–2024.
Forecast origins: tropical/subtropical-cyclone fixes with ≥24 h of history, verified against the best track at the lead time (storm must still exist then). Splits are by storm season; validation = most recent 20% of training storms (architecture and blend weights are chosen there, never on test data).

Config: ensemble=3, hidden grid=[[32, 16], [64, 32]], epochs<=120, seed=7

## Split: train ≤2009 (399 storms) → test 2010–2014 (81 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP vs poly ridge (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1454 | 90 | 66 | 66 | 7.2 | 6.0 | 0.50 / 0.75 |
| 24h | 1380 | 231 | 174 | 174 | 12.3 | 10.0 | 0.50 / 0.50 |
| 36h | 1279 | 408 | 308 | 304 | 16.1 | 13.0 | 0.50 / 0.50 |
| 48h | 1170 | 601 | 450 | 437 | 18.9 | 14.9 | 0.50 / 0.25 |
| 72h | 978 | 1011 | 733 | 714 | 22.3 | 16.0 | 0.50 / 0.50 |

## Split: train ≤2014 (480 storms) → test 2015–2019 (77 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP vs poly ridge (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1603 | 87 | 68 | 68 | 7.2 | 5.9 | 0.25 / 0.75 |
| 24h | 1542 | 221 | 180 | 181 | 12.5 | 10.2 | 0.25 / 0.75 |
| 36h | 1464 | 390 | 319 | 323 | 16.4 | 13.3 | 0.25 / 1.00 |
| 48h | 1361 | 571 | 464 | 476 | 19.4 | 15.7 | 0.25 / 0.50 |
| 72h | 1145 | 968 | 756 | 772 | 24.2 | 19.1 | 0.25 / 0.50 |

## Split: train ≤2019 (557 storms) → test 2020–2024 (99 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP vs poly ridge (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1835 | 87 | 68 | 68 | 8.0 | 6.5 | 0.50 / 0.75 |
| 24h | 1741 | 220 | 175 | 174 | 13.6 | 10.7 | 0.75 / 0.75 |
| 36h | 1624 | 379 | 304 | 300 | 17.6 | 13.4 | 0.50 / 1.00 |
| 48h | 1498 | 550 | 432 | 428 | 20.5 | 15.5 | 0.75 / 0.50 |
| 72h | 1240 | 917 | 676 | 658 | 23.9 | 16.4 | 0.75 / 0.50 |

## Pooled over all test periods (95% CI from a storm-level bootstrap, 1000 resamples)

### Track error (km, lower is better)

| Lead | n | Persistence | Const.-velocity | Ridge (CLIPER-like) | Poly ridge | MLP | **AI blend** | AI − CV (km) [95% CI] | Skill vs CV | AI − plain ridge (km) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 12h | 4892 | 252 | 88 | 67 | 68 | 69 | **68** | -20.4 [-22.3, -18.6] | +23.2% | +0.2 [-0.4, +0.9] |
| 24h | 4663 | 505 | 224 | 176 | 177 | 179 | **176** | -47.5 [-53.0, -41.5] | +21.2% | -0.1 [-2.3, +2.3] |
| 36h | 4367 | 750 | 391 | 310 | 310 | 314 | **309** | -82.3 [-93.0, -71.7] | +21.0% | -1.5 [-5.7, +3.0] |
| 48h | 4029 | 983 | 572 | 448 | 447 | 454 | **447** | -125.2 [-145.2, -107.2] | +21.9% | -1.2 [-9.2, +7.1] |
| 72h | 3363 | 1413 | 962 | 720 | 713 | 726 | **713** | -248.6 [-281.5, -210.5] | +25.8% | -6.8 [-23.4, +9.1] |

### Intensity error (kt MAE of maximum sustained wind, lower is better)

| Lead | n | Persistence | Ridge | Poly ridge | MLP | **AI blend** | AI − persistence (kt) [95% CI] | Skill vs persistence |
|---|---|---|---|---|---|---|---|---|
| 12h | 4892 | 7.49 | 6.57 | 6.38 | 6.21 | **6.16** | -1.33 [-1.59, -1.08] | +17.8% |
| 24h | 4663 | 12.87 | 11.61 | 10.85 | 10.45 | **10.34** | -2.54 [-3.01, -2.03] | +19.7% |
| 36h | 4367 | 16.77 | 15.07 | 13.81 | 13.34 | **13.26** | -3.51 [-4.22, -2.76] | +20.9% |
| 48h | 4029 | 19.68 | 17.12 | 15.68 | 15.77 | **15.40** | -4.28 [-5.36, -3.30] | +21.7% |
| 72h | 3363 | 23.54 | 18.99 | 17.36 | 17.86 | **17.22** | -6.32 [-7.85, -4.66] | +26.8% |

### Rapid intensification (≥30 kt gain in 24 h)

n=4663, events=247 (5.3%). Brier: AI 0.0452 vs climatology 0.0502 (skill +9.9%); AUC 0.823. At p≥0.2: hits 78/247 (POD 0.32), false alarms 219 (precision 0.26).
