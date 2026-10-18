use ai_weather::climatology::Climatology;
use ai_weather::data::*;
use ai_weather::eval::{evaluate_split, Split};
use ai_weather::linalg::*;
use ai_weather::metrics::*;
use ai_weather::mlp::*;
use ai_weather::models::TrainCfg;
use ai_weather::rng::Rng;

#[test]
fn date_roundtrip_and_doy() {
    assert_eq!(Date::new(1970, 1, 1).ordinal(), 0);
    assert_eq!(Date::new(2000, 3, 1).ordinal() - Date::new(2000, 2, 28).ordinal(), 2); // leap year
    for n in [-1000i64, 0, 365, 11_000, 15_000, 20_000] {
        assert_eq!(Date::from_ordinal(n).ordinal(), n);
    }
    assert_eq!(Date::new(2012, 12, 31).doy(), 366);
    assert_eq!(Date::parse("\"2015-07-04\""), Some(Date::new(2015, 7, 4)));
}

#[test]
fn ridge_recovers_linear_map() {
    let mut rng = Rng::new(1);
    let (mut x, mut y) = (Mat::default(), Mat::default());
    for _ in 0..200 {
        let (a, b) = (rng.normal(), rng.normal());
        x.push(&[a, b]);
        y.push(&[1.0 + 2.0 * a - 3.0 * b]);
    }
    let w = ridge_fit(&x, &y, 1e-9);
    assert!((w.at(0, 0) - 1.0).abs() < 1e-6 && (w.at(1, 0) - 2.0).abs() < 1e-6 && (w.at(2, 0) + 3.0).abs() < 1e-6);
}

#[test]
fn mlp_learns_nonlinear_function() {
    let mut rng = Rng::new(3);
    let (mut x, mut y) = (Mat::default(), Mat::default());
    for _ in 0..400 {
        let a = rng.f64() * 6.0 - 3.0;
        x.push(&[a]);
        y.push(&[(a).sin()]);
    }
    let cfg = MlpConfig { epochs: 300, ..MlpConfig::default() };
    let net = Mlp::fit(&x, &y, &Mat::default(), &Mat::default(), Loss::Mse, &cfg, 5);
    assert!(net.dataset_loss(&x, &y) < 0.02, "loss {}", net.dataset_loss(&x, &y));
}

#[test]
fn metrics_known_values() {
    assert!((auc(&[0.1, 0.4, 0.35, 0.8], &[false, false, true, true]) - 0.75).abs() < 1e-12);
    assert!((brier(&[1.0, 0.0], &[true, false])).abs() < 1e-12);
    let c = confusion(2, &[0, 0, 1, 1], &[0, 1, 1, 1]);
    assert_eq!(c, vec![vec![1, 1], vec![0, 2]]);
}

#[test]
fn climatology_recovers_seasonal_cycle() {
    let mut csv = String::from("Date,Temp\n");
    let start = Date::new(2000, 1, 1).ordinal();
    for i in 0..1500 {
        let d = Date::from_ordinal(start + i);
        let v = 15.0 + 8.0 * (2.0 * std::f64::consts::PI * (d.doy() as f64 - 1.0) / 365.25).sin();
        csv.push_str(&format!("{},{:.4}\n", d, v));
    }
    let ds = parse_single(&csv, "synthetic", "t", Kind::Temp).unwrap();
    let c = Climatology::fit(&ds, ds.len());
    assert!(c.anom_std[0] < 1e-3);
}

#[test]
fn leap_day_gap_is_filled() {
    let ds = parse_single("Date,Temp\n1984-02-28,10\n1984-03-01,12\n", "g", "t", Kind::Temp).unwrap();
    assert_eq!(ds.len(), 3);
    assert_eq!(ds.n_filled, 1);
    assert!((ds.ch[0].v[1] - 11.0).abs() < 1e-9);
}

#[test]
fn end_to_end_beats_baselines_on_seattle_holdout() {
    let ds = load_seattle("data/seattle-weather.csv").unwrap();
    let split = Split { name: "2015".into(), train_end: Date::new(2014, 12, 31), test_start: Date::new(2015, 1, 1), test_end: Date::new(2015, 12, 31) };
    let r = evaluate_split(&ds, &split, &TrainCfg::quick());
    // tmax_c, tmin_c: index 0, 1. [mlp, ridge, clim, persistence]
    for c in 0..2 {
        assert!(r.mae_h1[c][0] < r.mae_h1[c][2], "1-day MAE must beat climatology: {:?}", r.mae_h1[c]);
    }
    let b = r.brier_h1.unwrap();
    assert!(b[0] < b[1] && b[0] < b[2], "rain Brier {:?}", b);
    let (m, majority, _) = r.detect_acc.unwrap();
    assert!(m > majority + 0.2);
    let (fm, fc) = r.fault_auc_4sigma.unwrap();
    assert!(fm >= fc - 1e-9 && fm > 0.95);
}

#[test]
fn new_york_station_loads_and_generalises() {
    let ds = load_new_york("data/vega-weather.csv").unwrap();
    assert_eq!(ds.len(), 1461);
    assert_eq!(ds.n_filled, 0);
    let split = Split { name: "2015".into(), train_end: Date::new(2014, 12, 31), test_start: Date::new(2015, 1, 1), test_end: Date::new(2015, 12, 31) };
    let r = evaluate_split(&ds, &split, &TrainCfg::quick());
    for c in 0..2 {
        assert!(r.mae_h1[c][0] < r.mae_h1[c][2], "NYC 1-day MAE must beat climatology: {:?}", r.mae_h1[c]);
    }
}

#[test]
fn eca_wide_loader_converts_units_and_gen_on_eu_station() {
    let ds = load_eca_station("data/eu-weather-prediction-dataset.csv", "DE_BILT", false).unwrap();
    assert_eq!(ds.len(), 3654);
    assert!(ds.labels.is_none());
    // precipitation stored in 10 mm units must be converted: max daily value should be tens of mm, not < 10
    let pmax = ds.ch[3].v.iter().cloned().fold(0.0, f64::max);
    assert!(pmax > 15.0 && pmax < 200.0, "pmax {pmax}");
    let split = Split { name: "08-09".into(), train_end: Date::new(2007, 12, 31), test_start: Date::new(2008, 1, 1), test_end: Date::new(2009, 12, 31) };
    let r = evaluate_split(&ds, &split, &TrainCfg::quick());
    for c in 0..2 {
        assert!(r.mae_h1[c][0] < r.mae_h1[c][2], "EU 1-day MAE must beat climatology: {:?}", r.mae_h1[c]);
    }
}

#[test]
fn neighbour_inputs_attach_aligned_aux_series_and_run_end_to_end() {
    let own = load_eca_station("data/eu-weather-prediction-dataset.csv", "DE_BILT", false).unwrap();
    let nb = load_eca_station("data/eu-weather-prediction-dataset.csv", "DE_BILT", true).unwrap();
    assert!(own.aux.is_empty());
    assert_eq!(nb.len(), own.len());
    assert!(nb.aux.len() > 20, "expected neighbour temp+pressure series, got {}", nb.aux.len());
    assert!(nb.aux.iter().all(|a| a.v.len() == nb.len() && a.v.iter().all(|x| x.is_finite())));
    assert!(nb.aux.iter().any(|a| a.name == "DE_BILT_pressure(own)"));
    assert!(!nb.aux.iter().any(|a| a.name == "DE_BILT_temp_mean"), "target station must not appear as its own neighbour");
    let split = Split { name: "08-09".into(), train_end: Date::new(2007, 12, 31), test_start: Date::new(2008, 1, 1), test_end: Date::new(2009, 12, 31) };
    let r = evaluate_split(&nb, &split, &TrainCfg::quick());
    assert!(r.mae_h1[0][0] < r.mae_h1[0][2]);
}

use ai_weather::hurricane::{haversine_km, load_storms, parse_storms, HurCfg};

#[test]
fn haversine_known_distance() {
    // London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is about 344 km
    let d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522);
    assert!((d - 344.0).abs() < 4.0, "{d}");
    assert!(haversine_km(10.0, -50.0, 10.0, -50.0).abs() < 1e-9);
}

#[test]
fn storms_parser_filters_non_synoptic_and_splits_gaps() {
    let mut csv = String::from("rownames,name,year,month,day,hour,lat,long,status,category,wind,pressure,tropicalstorm_force_diameter,hurricane_force_diameter\n");
    let mut n = 0;
    for h in [0, 6, 12, 18, 24, 30, 36, 42, 48] {
        let (d, hh) = (1 + h / 24, h % 24);
        n += 1;
        csv.push_str(&format!("{n},Test,2001,8,{d},{hh},20.0,-50.0,hurricane,1,70,980,,\n"));
    }
    csv.push_str("99,Test,2001,8,2,3,20.0,-50.0,hurricane,1,70,980,,\n"); // 03 UTC: dropped
    let tr = parse_storms(&csv).unwrap();
    assert_eq!(tr.len(), 1);
    assert_eq!(tr[0].fixes.len(), 9);
}

#[test]
fn constant_velocity_baseline_is_exact_on_straight_track() {
    // a storm moving due east at constant speed: residual targets must be ~0 so the CV baseline has ~0 error
    let mut csv = String::from("rownames,name,year,month,day,hour,lat,long,status,category,wind,pressure,tropicalstorm_force_diameter,hurricane_force_diameter\n");
    for k in 0..30 {
        let h = 6 * k;
        csv.push_str(&format!("{},Line,2005,9,{},{},15.0,{:.4},tropical storm,,50,1000,,\n", k + 1, 1 + h / 24, h % 24, -40.0 + 0.1 * k as f64));
    }
    let tr = parse_storms(&csv).unwrap();
    let cfg = HurCfg::quick();
    // train and test on the same synthetic storm just to inspect the CV baseline error
    let all = vec![tr[0].clone(), tr[0].clone()];
    let out = {
        let mut a = all.clone();
        a[1].id = 1;
        a[1].year = 2006;
        a[0].year = 2005;
        ai_weather::hurricane::evaluate_split(&a, 2005, 2006, 2006, &cfg)
    };
    let cv24 = &out.recs[1].trk[1];
    assert!(!cv24.is_empty());
    assert!(cv24.iter().all(|e| *e < 1.0), "CV error on straight track should be ~0 km, got {:?}", &cv24[..3]);
}

#[test]
fn hurricane_quick_run_on_real_data_beats_baselines() {
    let tracks = load_storms("data/storms.csv").unwrap();
    assert!(tracks.len() > 500);
    let out = ai_weather::hurricane::evaluate_split(&tracks, 2014, 2015, 2019, &HurCfg::quick());
    let m = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let r24 = &out.recs[1]; // 24 h
    assert!(m(&r24.trk[1]) < m(&r24.trk[0]), "constant velocity must beat persistence");
    assert!(m(&r24.trk[4]) < 0.9 * m(&r24.trk[1]), "AI track must beat CV by >10%: {} vs {}", m(&r24.trk[4]), m(&r24.trk[1]));
    assert!(m(&r24.int[3]) < 0.9 * m(&r24.int[0]), "AI intensity must beat persistence by >10%");
    assert!(out.ri.p_ai.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
}
