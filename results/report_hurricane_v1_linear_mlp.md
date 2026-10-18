# AI weather system — tropical-cyclone track & intensity forecasting

Data: NOAA HURDAT2-derived best tracks (dplyr `storms`), Atlantic, 6-hourly, 656 storm segments, 1975–2024.
Forecast origins: tropical/subtropical-cyclone fixes with ≥24 h of history, verified against the best track at the lead time (storm must still exist then). Splits are by storm season; validation = most recent 20% of training storms (architecture and blend weights are chosen there, never on test data).

Config: ensemble=3, hidden grid=[[32, 16], [64, 32]], epochs<=120, seed=7

## Split: train ≤2009 (399 storms) → test 2010–2014 (81 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1454 | 90 | 66 | 66 | 7.2 | 6.0 | 0.50 / 1.00 |
| 24h | 1380 | 231 | 174 | 175 | 12.3 | 10.1 | 0.50 / 1.00 |
| 36h | 1279 | 408 | 308 | 309 | 16.1 | 13.3 | 0.50 / 1.00 |
| 48h | 1170 | 601 | 450 | 449 | 18.9 | 15.2 | 0.75 / 1.00 |
| 72h | 978 | 1011 | 733 | 735 | 22.3 | 16.2 | 0.75 / 1.00 |

## Split: train ≤2014 (480 storms) → test 2015–2019 (77 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1603 | 87 | 68 | 67 | 7.2 | 6.0 | 0.50 / 0.75 |
| 24h | 1542 | 221 | 180 | 179 | 12.5 | 10.3 | 0.50 / 1.00 |
| 36h | 1464 | 390 | 319 | 315 | 16.4 | 13.3 | 0.50 / 1.00 |
| 48h | 1361 | 571 | 464 | 465 | 19.4 | 16.3 | 0.50 / 0.75 |
| 72h | 1145 | 968 | 756 | 755 | 24.2 | 19.8 | 0.50 / 1.00 |

## Split: train ≤2019 (557 storms) → test 2020–2024 (99 storms)

| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP (track / int.) |
|---|---|---|---|---|---|---|---|
| 12h | 1835 | 87 | 68 | 68 | 8.0 | 6.6 | 0.50 / 0.75 |
| 24h | 1741 | 220 | 175 | 173 | 13.6 | 10.8 | 0.50 / 1.00 |
| 36h | 1624 | 379 | 304 | 302 | 17.6 | 13.4 | 0.50 / 1.00 |
| 48h | 1498 | 550 | 432 | 429 | 20.5 | 16.0 | 0.25 / 1.00 |
| 72h | 1240 | 917 | 676 | 670 | 23.9 | 17.4 | 0.25 / 1.00 |

## Pooled over all test periods (95% CI from a storm-level bootstrap, 1000 resamples)

### Track error (km, lower is better)

| Lead | n | Persistence | Const.-velocity | Ridge (CLIPER-like) | MLP | **AI blend** | AI − CV (km) [95% CI] | Skill vs CV | AI − ridge (km) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 12h | 4892 | 252 | 88 | 67 | 69 | **67** | -20.7 [-22.6, -18.8] | +23.5% | -0.0 [-0.4, +0.3] |
| 24h | 4663 | 505 | 224 | 176 | 179 | **176** | -48.0 [-53.0, -42.9] | +21.4% | -0.6 [-1.7, +0.4] |
| 36h | 4367 | 750 | 391 | 310 | 314 | **309** | -82.6 [-92.9, -73.1] | +21.1% | -1.8 [-4.0, +0.6] |
| 48h | 4029 | 983 | 572 | 448 | 454 | **447** | -124.8 [-141.5, -108.5] | +21.8% | -0.9 [-4.4, +2.4] |
| 72h | 3363 | 1413 | 962 | 720 | 726 | **718** | -244.1 [-277.7, -210.9] | +25.4% | -2.3 [-9.8, +5.1] |

### Intensity error (kt MAE of maximum sustained wind, lower is better)

| Lead | n | Persistence | Ridge | MLP | **AI blend** | AI − persistence (kt) [95% CI] | Skill vs persistence |
|---|---|---|---|---|---|---|---|
| 12h | 4892 | 7.49 | 6.57 | 6.21 | **6.22** | -1.28 [-1.53, -1.03] | +17.0% |
| 24h | 4663 | 12.87 | 11.61 | 10.45 | **10.45** | -2.42 [-2.94, -1.89] | +18.8% |
| 36h | 4367 | 16.77 | 15.07 | 13.34 | **13.34** | -3.43 [-4.18, -2.63] | +20.4% |
| 48h | 4029 | 19.68 | 17.12 | 15.77 | **15.85** | -3.83 [-4.93, -2.80] | +19.4% |
| 72h | 3363 | 23.54 | 18.99 | 17.86 | **17.86** | -5.68 [-7.22, -4.01] | +24.1% |

### Rapid intensification (≥30 kt gain in 24 h)

n=4663, events=247 (5.3%). Brier: AI 0.0452 vs climatology 0.0502 (skill +9.9%); AUC 0.823. At p≥0.2: hits 78/247 (POD 0.32), false alarms 219 (precision 0.26).
