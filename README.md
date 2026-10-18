# ai_weather — AI-driven weather detection & forecasting (Rust, zero dependencies)

Builds with `rustc 1.75` (same MSRV as your workspaces); only `std`.

## What it does
| Capability | Method | Module |
|---|---|---|
| Multi-horizon forecast (1–7 d) of every channel (tmax, tmin, wind, precip) | Seasonal harmonic climatology + MLP-ensemble predicting *anomalies* from lagged anomalies, rolling means, wet-spell stats, target-day season; ridge baseline on same features | `models.rs`, `features.rs`, `climatology.rs` |
| Rain probability | MLP-ensemble softmax, shrunk toward monthly climatology with a validation-chosen weight | `models.rs` |
| Next-day condition forecast | MLP-ensemble softmax (sun/fog/drizzle/rain/snow) | `models.rs` |
| Current-condition **detection** | MLP-ensemble on same-day readings + season (no future info) | `detect.rs` |
| Sensor-fault / extreme **anomaly detection** | z-score of observation vs 1-day-ahead forecast (compared with climatology z-score) | `quality.rs` |

Leakage control: strictly chronological splits; climatology, scalers, early stopping and blend weights use
training data only; a forecast at origin `t` uses data ≤ `t`.

## Run
```sh
cargo test --release                                  # 14 tests (unit + end-to-end hold-out checks on Seattle, New York, EU stations and hurricanes)
cargo run --release -- evaluate [--quick]             # writes results/report.md (~1 min)
cargo run --release -- evaluate-eu                    # 8 EU stations with neighbour inputs (default), ~5 min
cargo run --release -- evaluate-eu --no-neighbors     # own-station baseline (for the ablation)
cargo run --release -- hurricane                      # track/intensity/RI on NOAA best tracks, ~6 min, writes results/report_hurricane.md
cargo run --release -- forecast --dataset seattle     # 7-day forecast after the last record
```

## Headline results (full numbers: `results/report.md`)
Hold-out years, trained only on earlier years (Seattle and New York 2014, 2015; Melbourne 1989, 1990).
* **1-day temperature**: MSE skill vs climatology +0.43…+0.57 (Seattle tmax/tmin), +0.26…+0.35 (Melbourne tmin); MAE ≈ 1.4–2.3 °C. Roughly on par with persistence/ridge at 1 day (ridge is sometimes marginally better).
* **Skill decays quickly**: ≈ +0.1–0.3 at 2 d, ≈ 0 by 4–7 d. Station-only history cannot see synoptic weather; real multi-day skill needs NWP/satellite/multi-station inputs.
* **Rain probability**: 1-day Brier 0.189/0.190 vs monthly climatology 0.222/0.203; beyond ~2 days it reverts to climatology (no skill).
* **Condition detection**: accuracy 0.87 / 0.83 (majority 0.40). A trivial "precip>0 → rain else sun" rule is as good or better on accuracy (0.92 / 0.84); the model's gain is macro-F1 in 2014 (0.66 vs 0.48, it finds snow). Fog/drizzle are essentially undetectable from these four readings.
* **Fault detection**: forecast-residual detector AUC 0.99+ at 3–4σ faults vs 0.95–0.99 for climatology z; clearly better at 2σ (≈0.92–0.95 vs 0.81–0.90).

### Third dataset added: New York (2012–2015, vega-datasets `weather.csv`)
* 1-day temperature skill vs climatology: +0.32/+0.41 (tmax/tmin, 2014), +0.46/+0.53 (2015). Unlike Seattle, some skill persists to 5–7 days in 2015 (+0.06 to +0.10) but is ≈0 in 2014.
* Rain probability: essentially **no skill** (Brier 0.222 vs 0.224 in 2014; 0.207 vs 0.210 in 2015; AUC ≈ 0.57–0.60). Treat as climatology.
* Next-day condition: does not beat the majority class (0.57–0.58 vs 0.60–0.62).
* Current-condition detection: accuracy 0.92 / 0.89 vs the simple precip rule 0.90 / 0.85, macro-F1 0.54 / 0.57 vs 0.38 / 0.37 (finds snow, F1 0.77–0.87); fog and drizzle still undetected.
* Fault detection (tmax): AUC 0.87–0.91 vs 0.80–0.82 for climatology at 2σ; ≥0.98 at 3σ+.

### Hurricanes: track, intensity and rapid-intensification forecasting (`hurricane.rs`)
Data: NOAA HURDAT2-derived Atlantic best tracks (dplyr `storms` table via Rdatasets, 1975–2024, 6-hourly; NOAA's own
hosts were not reachable). 656 storm segments; origins are tropical/subtropical-cyclone fixes with ≥24 h history.
Leads 12/24/36/48/72 h. Three consecutive test periods (2010–14, 2015–19, 2020–24), each trained only on earlier
seasons; MLP architecture and ridge/MLP blend weights are chosen on the most recent 20% of *training storms*.
Pooled results (n = 3.4k–4.9k forecasts per lead; 95% CIs from a storm-level bootstrap; `results/report_hurricane.md`):

| Lead | Track error km: persistence / const.-velocity / **AI** | AI vs const.-velocity | Intensity MAE kt: persistence / **AI** | AI vs persistence |
|---|---|---|---|---|
| 12 h | 252 / 88 / **68** | −23% | 7.49 / **6.16** | −18% |
| 24 h | 505 / 224 / **176** | −21% | 12.87 / **10.34** | −20% |
| 48 h | 983 / 572 / **447** | −22% | 19.68 / **15.40** | −22% |
| 72 h | 1413 / 962 / **713** | −26% | 23.54 / **17.22** | −27% |

* **Track**: the whole gain over constant-velocity extrapolation comes from a linear (CLIPER-style) regression on position,
  motion, intensity and season. The MLP and polynomial-ridge variants add **nothing significant** on track (all CIs for
  AI − plain ridge span zero). Further track skill needs environmental steering information (NWP/reanalysis), which best-track data lacks.
* **Intensity**: nonlinear models help: AI beats the linear model by ~10% (24 h: 10.34 vs 11.61 kt) and persistence by 18–27%.
  The polynomial-ridge/MLP blend improved on the first version (MLP + linear: 10.45 kt at 24 h, 17.86 at 72 h; `results/report_hurricane_v1_linear_mlp.md`) by 1–4%; this small gain was not separately CI-tested.
* **Rapid intensification (≥30 kt/24 h, 5.3% of cases)**: AUC 0.82, Brier skill +9.9% vs climatology; at p ≥ 0.2 it catches 32% of events with 26% precision.
  Useful for ranking risk, weak as a categorical alarm.
* These are statistical-only errors on all eligible fixes; they are **not comparable to official NHC forecasts** (which use dynamical models
  and a stricter verification sample). "CLIPER-like" means my ridge regression, not NHC's CLIPER5.

### Improvement: neighbour-station and pressure inputs (multi-station data only)
When a dataset has several stations, `evaluate-eu` now feeds the forecaster every other station's temperature
anomaly (today, yesterday) and pressure anomaly + 1-day tendency, plus the station's own pressure tendency
(`Dataset::aux`, `features.rs`). These are inputs only, never targets. Motivation: weather moves between
stations, which a single-station model cannot see. Measured effect (same seeds/splits, mean of 8 stations,
`results/neighbor_ablation.md`):

| Metric | Own station only | + neighbours/pressure |
|---|---|---|
| tmax skill vs climatology, 1d / 2d / 3d | +0.315 / +0.134 / +0.069 | **+0.385 / +0.196 / +0.105** |
| tmin skill, 1d / 2d / 3d | +0.302 / +0.112 / +0.049 | **+0.364 / +0.181 / +0.095** |
| tmax MAE 1d (°C) | 2.02 | **1.81** |
| rain Brier 1d / 2d (climatology 0.238) | 0.213 / 0.235 | **0.198 / 0.224** |
| rain AUC 1d | 0.700 | **0.744** |
| fault AUC at 2σ | 0.968 | **0.980** |

All 8 stations improve at 1-day tmax, tmin and rain Brier. Day-7 skill stays ≈ 0; wind improves only marginally
(MAE 1.009 → 0.977 m/s). Single-station datasets (Seattle, New York, Melbourne) are unchanged.
Caveat: the feature family was chosen after a quick prototype on the same 2008–09 test years (no tuning of
individual features), so treat the size of the gain as slightly optimistic; confirm on other years/regions.

### Fourth dataset: EU multi-station (ECA&D via florian-huber/weather_prediction_dataset)
Eight stations (De Bilt, Düsseldorf, München, Maastricht, Oslo, Ljubljana, Perpignan, Malmö), daily 2000–2010,
trained on 2000–2007 and tested on 2008–2009 (`results/report_eu.md` = with neighbour inputs; `results/report_eu_baseline.md` = own-station only). The figures in this paragraph are the own-station baseline. No condition labels, so only
forecasting, rain probability and fault detection apply.
* **Temperature**: 1-day MAE skill vs climatology +0.30 (tmax) / +0.30 (tmin) on average (range +0.15 Perpignan … +0.39 München); 1-day MAE 1.5–2.4 °C. Beats persistence at every station, by a clear margin at München (tmax 2.42 vs 2.92). **No skill by day 7** (mean −0.01 / −0.02).
* **Rain probability (1 day)**: Brier better than monthly climatology at every station (mean improvement 0.025; AUC 0.70 vs 0.56). Reverts to climatology at longer leads.
* **Fault detection**: forecast-residual AUC at 2σ 0.97 vs 0.87 for climatology z (0.92 vs 0.85 at Perpignan); saturates at ≥0.99 for 4σ.
* Caveat: the dataset authors replaced ≤5% invalid entries per column with column means; this cannot be isolated, so a small fraction of values are artificial.

