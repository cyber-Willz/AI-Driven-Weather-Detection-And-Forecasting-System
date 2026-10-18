use ai_weather::climatology::Climatology;
use ai_weather::data::{load_eca_station, load_melbourne, load_new_york, load_seattle, Dataset, Date};
use ai_weather::detect::argmax;
use ai_weather::eval::{evaluate_split, Split};
use ai_weather::features::{Context, MIN_T};
use ai_weather::models::{Forecaster, TrainCfg};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn sp(name: &str, te: (i32, u32, u32), ts: (i32, u32, u32), tend: (i32, u32, u32)) -> Split {
    Split { name: name.into(), train_end: Date::new(te.0, te.1, te.2), test_start: Date::new(ts.0, ts.1, ts.2), test_end: Date::new(tend.0, tend.1, tend.2) }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    let quick = args.iter().any(|a| a == "--quick");
    let cfg = if quick { TrainCfg::quick() } else { TrainCfg::default() };
    let dir = arg(&args, "--data-dir").unwrap_or_else(|| "data".into());
    match cmd {
        "evaluate" => {
            let out = arg(&args, "--out").unwrap_or_else(|| "results/report.md".into());
            let sea = load_seattle(&format!("{dir}/seattle-weather.csv")).expect("seattle");
            let nyc = load_new_york(&format!("{dir}/vega-weather.csv")).expect("new york");
            let only = arg(&args, "--only");
            let mel = load_melbourne(&format!("{dir}/melbourne-daily-min-temperatures.csv")).expect("melbourne");
            let mut md = String::from("# AI weather system — hold-out evaluation on public datasets\n\n");
            md.push_str(&format!("Config: horizons={}, ensemble={}, hidden={:?}, epochs<={}, seed={}\n\n", cfg.horizons, cfg.ensemble, cfg.mlp.hidden, cfg.mlp.epochs, cfg.seed));
            for (ds, splits) in [
                (&sea, vec![sp("test year 2014 (train 2012-13)", (2013, 12, 31), (2014, 1, 1), (2014, 12, 31)), sp("test year 2015 (train 2012-14)", (2014, 12, 31), (2015, 1, 1), (2015, 12, 31))]),
                (&nyc, vec![sp("test year 2014 (train 2012-13)", (2013, 12, 31), (2014, 1, 1), (2014, 12, 31)), sp("test year 2015 (train 2012-14)", (2014, 12, 31), (2015, 1, 1), (2015, 12, 31))]),
                (&mel, vec![sp("test year 1989 (train 1981-88)", (1988, 12, 31), (1989, 1, 1), (1989, 12, 31)), sp("test year 1990 (train 1981-89)", (1989, 12, 31), (1990, 1, 1), (1990, 12, 31))]),
            ] {
                if let Some(o) = &only { if !ds.name.to_lowercase().contains(&o.to_lowercase()) { continue; } }
                for s in splits {
                    eprintln!("evaluating {} / {}", ds.name, s.name);
                    md.push_str(&evaluate_split(ds, &s, &cfg).md);
                }
            }
            if let Some(p) = std::path::Path::new(&out).parent() { let _ = std::fs::create_dir_all(p); }
            let out = if only.is_some() && arg(&args, "--out").is_none() { "results/report_subset.md".to_string() } else { out };
            std::fs::write(&out, &md).expect("write report");
            println!("{md}");
        }
        "evaluate-eu" => {
            let out = arg(&args, "--out").unwrap_or_else(|| "results/report_eu.md".into());
            let list = arg(&args, "--stations").unwrap_or_else(|| "DE_BILT,DUSSELDORF,MUENCHEN,MAASTRICHT,OSLO,LJUBLJANA,PERPIGNAN,MALMO".into());
            let path = format!("{dir}/eu-weather-prediction-dataset.csv");
            // Neighbour-station + pressure inputs are ON by default; --no-neighbors reproduces the baseline.
            let neighbors = !args.iter().any(|a| a == "--no-neighbors");
            let split = sp("train 2000-2007 / test 2008-2009", (2007, 12, 31), (2008, 1, 1), (2009, 12, 31));
            let mut md = format!("# AI weather system — EU multi-station hold-out evaluation (ECA&D)\n\nNeighbour-station inputs: {}\n\n", if neighbors { "ON" } else { "OFF" });
            md.push_str(&format!("Config: horizons={}, ensemble={}, hidden={:?}, epochs<={}, seed={}\n\n", cfg.horizons, cfg.ensemble, cfg.mlp.hidden, cfg.mlp.epochs, cfg.seed));
            let mut summary = String::from("## Summary (MAE in physical units; skill = 1 - MAE_model / MAE_climatology)\n\n| Station | tmax 1d MAE AI / clim / persist | tmax skill 1d → 7d | tmin 1d MAE AI / clim / persist | tmin skill 1d → 7d | Rain Brier 1d AI / clim | Fault AUC 4σ resid / clim |\n|---|---|---|---|---|---|---|\n");
            let mut detail = String::new();
            let (mut acc, mut cnt) = ([0.0f64; 6], 0.0f64);
            for st in list.split(',') {
                eprintln!("evaluating {st}");
                let ds = load_eca_station(&path, st, neighbors).expect("station");
                let r = evaluate_split(&ds, &split, &cfg);
                let (a, b) = (r.mae_h1[0], r.mae_h1[1]);
                let (a7, b7) = (r.mae_hmax[0], r.mae_hmax[1]);
                let sk = |m: [f64; 4]| 1.0 - m[0] / m[2];
                let br = r.brier_h1.unwrap_or([f64::NAN; 3]);
                let fa = r.fault_auc_4sigma.unwrap_or((f64::NAN, f64::NAN));
                summary.push_str(&format!("| {} | {:.2} / {:.2} / {:.2} | {:+.2} → {:+.2} | {:.2} / {:.2} / {:.2} | {:+.2} → {:+.2} | {:.3} / {:.3} | {:.3} / {:.3} |\n", st, a[0], a[2], a[3], sk(a), sk(a7), b[0], b[2], b[3], sk(b), sk(b7), br[0], br[1], fa.0, fa.1));
                for (i, v) in [sk(a), sk(a7), sk(b), sk(b7), br[1] - br[0], fa.0 - fa.1].iter().enumerate() { acc[i] += v; }
                cnt += 1.0;
                detail.push_str(&r.md);
            }
            summary.push_str(&format!("\n**Mean over {} stations:** tmax skill 1d {:+.3} → 7d {:+.3}; tmin skill 1d {:+.3} → 7d {:+.3}; rain Brier improvement over climatology (1d) {:+.4}; fault AUC gain (4σ) {:+.4}\n\n", cnt, acc[0] / cnt, acc[1] / cnt, acc[2] / cnt, acc[3] / cnt, acc[4] / cnt, acc[5] / cnt));
            md.push_str(&summary);
            md.push_str("---\n\n# Per-station detail\n\n");
            md.push_str(&detail);
            std::fs::write(&out, &md).expect("write");
            println!("{summary}");
        }
        "hurricane" => {
            use ai_weather::hurricane::{evaluate_all, load_storms, HurCfg};
            let out = arg(&args, "--out").unwrap_or_else(|| "results/report_hurricane.md".into());
            let tracks = load_storms(&format!("{dir}/storms.csv")).expect("storms");
            let hcfg = if quick { HurCfg::quick() } else { HurCfg::default() };
            let md = evaluate_all(&tracks, &hcfg, &[(2009, 2010, 2014), (2014, 2015, 2019), (2019, 2020, 2024)]);
            if let Some(p) = std::path::Path::new(&out).parent() { let _ = std::fs::create_dir_all(p); }
            std::fs::write(&out, &md).expect("write");
            println!("{md}");
        }
        "forecast" => {
            let which = arg(&args, "--dataset").unwrap_or_else(|| "seattle".into());
            let ds: Dataset = if which == "melbourne" { load_melbourne(&arg(&args, "--csv").unwrap_or(format!("{dir}/melbourne-daily-min-temperatures.csv"))).expect("load") } else { load_seattle(&arg(&args, "--csv").unwrap_or(format!("{dir}/seattle-weather.csv"))).expect("load") };
            let n = ds.len();
            let clim = Climatology::fit(&ds, n);
            let ctx = Context::new(&ds, &clim);
            let fc = Forecaster::train(&ds, &clim, &ctx, n - 1, &cfg);
            let t = n - 1;
            assert!(t >= MIN_T);
            println!("Forecast issued after {} ({})", ds.dates[t], ds.name);
            for h in 1..=cfg.horizons {
                let p = fc.predict(&ds, &ctx, t, h);
                let vals: Vec<String> = ds.ch.iter().enumerate().map(|(c, ch)| format!("{}={:.1}", ch.name, p.mlp[c])).collect();
                let rain = p.p_rain.map(|r| format!(" P(rain)={:.0}%", r * 100.0)).unwrap_or_default();
                let cond = p.cond_probs.as_ref().map(|c| format!(" likely={}", fc.label_names[argmax(c)])).unwrap_or_default();
                println!("{}  {}{}{}", ds.dates[t].plus(h as i64), vals.join(" "), rain, cond);
            }
        }
        _ => println!("usage: ai_weather hurricane [--quick] [--out results/report_hurricane.md]\n       ai_weather evaluate-eu [--stations A,B,..] [--out results/report_eu.md] [--no-neighbors]\n       ai_weather evaluate [--data-dir data] [--out results/report.md] [--quick]\n       ai_weather forecast [--dataset seattle|melbourne] [--csv path] [--quick]"),
    }
}
