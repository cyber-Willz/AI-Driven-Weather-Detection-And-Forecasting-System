//! Tropical-cyclone track and intensity forecasting on HURDAT2-derived best tracks
//! (dplyr `storms`: Atlantic, 1975-2024, 6-hourly).
//!
//! Leads 12/24/36/48/72 h. Track forecasts are residuals on top of a constant-velocity (12 h motion)
//! extrapolation; intensity forecasts are 6-hourly-wind changes. Baselines: persistence, constant
//! velocity, ridge regression (a CLIPER-style statistical model) and the MLP ensemble; the final "AI"
//! forecast blends ridge and MLP with weights chosen on a held-out validation set of storms.
//! Evaluation is chronological by storm season, with a storm-level bootstrap for uncertainty.

use crate::data::{split_csv_line, Date};
use crate::linalg::{ridge_fit, ridge_predict, std_dev, Mat, Standardizer};
use crate::metrics::{auc, brier};
use crate::mlp::{Ensemble, Loss, MlpConfig};
use crate::rng::Rng;
use std::collections::HashMap;
use std::fmt::Write;

pub const KM_PER_DEG: f64 = 111.195;
pub const LEAD_STEPS: [usize; 5] = [2, 4, 6, 8, 12]; // x 6 h = 12, 24, 36, 48, 72 h
pub const HIST: usize = 4; // 24 h of history required at the origin
pub const RI_STEPS: usize = 4; // rapid intensification: +30 kt within 24 h
pub const RI_KT: f64 = 30.0;

#[derive(Clone, Debug)]
pub struct Fix {
    pub t: i64, // hours since 1970-01-01
    pub lat: f64,
    pub lon: f64,
    pub wind: f64,
    pub pres: f64,
    pub status: u8, // 0 TD, 1 TS, 2 HU, 3 subtropical, 4 other (extratropical, low, wave, ...)
    pub doy: u32,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub id: usize,
    pub name: String,
    pub year: i32,
    pub fixes: Vec<Fix>,
}

fn status_code(s: &str) -> u8 {
    match s {
        "tropical depression" => 0,
        "tropical storm" => 1,
        "hurricane" => 2,
        "subtropical storm" | "subtropical depression" => 3,
        _ => 4,
    }
}

/// Parse the `storms.csv` table. Keeps synoptic times only (00/06/12/18 UTC) and splits a storm
/// wherever consecutive fixes are not exactly 6 h apart.
pub fn parse_storms(text: &str) -> Result<Vec<Track>, String> {
    let mut lines = text.lines();
    let header = split_csv_line(lines.next().ok_or("empty file")?);
    let col = |n: &str| header.iter().position(|h| h == n).ok_or_else(|| format!("column {n} not found"));
    let (cn, cy, cm, cd, ch) = (col("name")?, col("year")?, col("month")?, col("day")?, col("hour")?);
    let (clat, clon, cs, cw, cp) = (col("lat")?, col("long")?, col("status")?, col("wind")?, col("pressure")?);
    let mut groups: Vec<((String, i32), Vec<Fix>)> = Vec::new();
    let mut index: HashMap<(String, i32), usize> = HashMap::new();
    for (ln, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        let bad = |what: &str| format!("line {}: bad {what}", ln + 2);
        let hour: i64 = f[ch].parse().map_err(|_| bad("hour"))?;
        if hour % 6 != 0 {
            continue;
        }
        let year: i32 = f[cy].parse().map_err(|_| bad("year"))?;
        let date = Date::new(year, f[cm].parse().map_err(|_| bad("month"))?, f[cd].parse().map_err(|_| bad("day"))?);
        let fix = Fix {
            t: date.ordinal() * 24 + hour,
            lat: f[clat].parse().map_err(|_| bad("lat"))?,
            lon: f[clon].parse().map_err(|_| bad("long"))?,
            wind: f[cw].parse().map_err(|_| bad("wind"))?,
            pres: f[cp].parse().map_err(|_| bad("pressure"))?,
            status: status_code(&f[cs]),
            doy: date.doy(),
        };
        let key = (f[cn].clone(), year);
        let gi = *index.entry(key.clone()).or_insert_with(|| {
            groups.push((key.clone(), Vec::new()));
            groups.len() - 1
        });
        groups[gi].1.push(fix);
    }
    let mut tracks: Vec<Track> = Vec::new();
    for ((name, year), mut fixes) in groups {
        fixes.sort_by_key(|x| x.t);
        fixes.dedup_by_key(|x| x.t);
        let mut seg: Vec<Fix> = Vec::new();
        let flush = |seg: &mut Vec<Fix>, tracks: &mut Vec<Track>| {
            if seg.len() >= HIST + 3 {
                tracks.push(Track { id: 0, name: name.clone(), year, fixes: std::mem::take(seg) });
            } else {
                seg.clear();
            }
        };
        for fx in fixes {
            if let Some(last) = seg.last() {
                if fx.t - last.t != 6 {
                    flush(&mut seg, &mut tracks);
                }
            }
            seg.push(fx);
        }
        flush(&mut seg, &mut tracks);
    }
    tracks.sort_by_key(|t| t.fixes[0].t);
    for (i, t) in tracks.iter_mut().enumerate() {
        t.id = i;
    }
    if tracks.is_empty() {
        return Err("no tracks parsed".into());
    }
    Ok(tracks)
}

pub fn load_storms(path: &str) -> Result<Vec<Track>, String> {
    parse_storms(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
}

// ---------------------------------------------------------------- geometry

pub fn haversine_km(la1: f64, lo1: f64, la2: f64, lo2: f64) -> f64 {
    let (p1, p2) = (la1.to_radians(), la2.to_radians());
    let dl = (lo2 - lo1).to_radians();
    let dp = p2 - p1;
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * 6371.0 * a.sqrt().asin()
}

/// (east, north) km for a lat/lon change, using the cos(latitude) at `lat0`.
fn to_km(lat0: f64, dlat: f64, dlon: f64) -> (f64, f64) {
    (dlon * KM_PER_DEG * lat0.to_radians().cos(), dlat * KM_PER_DEG)
}
fn from_km(lat0: f64, lon0: f64, e: f64, n: f64) -> (f64, f64) {
    (lat0 + n / KM_PER_DEG, lon0 + e / (KM_PER_DEG * lat0.to_radians().cos().max(0.05)))
}

/// Mean motion (km/h east, north) over the `k` 6-hour steps ending at fix `i`.
fn velocity(tr: &Track, i: usize, k: usize) -> (f64, f64) {
    let (a, b) = (&tr.fixes[i - k], &tr.fixes[i]);
    let (e, n) = to_km(b.lat, b.lat - a.lat, b.lon - a.lon);
    (e / (6.0 * k as f64), n / (6.0 * k as f64))
}

pub fn features(tr: &Track, i: usize) -> Vec<f64> {
    let f = &tr.fixes[i];
    let w = std::f64::consts::TAU * (f.doy as f64 - 1.0) / 365.25;
    let (v6, v12, v24) = (velocity(tr, i, 1), velocity(tr, i, 2), velocity(tr, i, 4));
    let peak = tr.fixes[..=i].iter().map(|x| x.wind).fold(0.0, f64::max);
    let mut x = vec![
        f.lat, f.lon, f.wind, f.pres, w.sin(), w.cos(),
        v6.0, v6.1, v12.0, v12.1, v24.0, v24.1,
        f.wind - tr.fixes[i - 1].wind,
        f.wind - tr.fixes[i - 2].wind,
        f.wind - tr.fixes[i - 4].wind,
        f.pres - tr.fixes[i - 2].pres,
        f.pres - tr.fixes[i - 4].pres,
        (1.0 + 6.0 * i as f64).ln(),
        peak - f.wind,
    ];
    for s in 0..5u8 {
        x.push(if f.status == s { 1.0 } else { 0.0 });
    }
    x
}

// ---------------------------------------------------------------- datasets

#[derive(Clone, Debug)]
pub struct Meta {
    pub storm: usize,
    pub lat0: f64,
    pub lon0: f64,
    pub e_cv: f64,
    pub n_cv: f64,
    pub lat_t: f64,
    pub lon_t: f64,
    pub wind0: f64,
    pub wind_t: f64,
}

impl Meta {
    fn track_err(&self, re: f64, rn: f64) -> f64 {
        let (la, lo) = from_km(self.lat0, self.lon0, self.e_cv + re, self.n_cv + rn);
        haversine_km(la, lo, self.lat_t, self.lon_t)
    }
    fn int_err(&self, dw: f64) -> f64 {
        (self.wind0 + dw - self.wind_t).abs()
    }
}

struct LeadData {
    x: Mat,
    y: Mat, // residual targets: [east km, north km, wind kt]
    meta: Vec<Meta>,
}

fn build_lead(tracks: &[&Track], steps: usize) -> LeadData {
    let (mut x, mut y, mut meta) = (Mat::default(), Mat::default(), Vec::new());
    for tr in tracks {
        for i in HIST..tr.fixes.len() {
            if tr.fixes[i].status > 3 || i + steps >= tr.fixes.len() {
                continue;
            }
            let (o, t) = (&tr.fixes[i], &tr.fixes[i + steps]);
            let v12 = velocity(tr, i, 2);
            let hrs = 6.0 * steps as f64;
            let (e_cv, n_cv) = (v12.0 * hrs, v12.1 * hrs);
            let (e, n) = to_km(o.lat, t.lat - o.lat, t.lon - o.lon);
            x.push(&features(tr, i));
            y.push(&[e - e_cv, n - n_cv, t.wind - o.wind]);
            meta.push(Meta { storm: tr.id, lat0: o.lat, lon0: o.lon, e_cv, n_cv, lat_t: t.lat, lon_t: t.lon, wind0: o.wind, wind_t: t.wind });
        }
    }
    LeadData { x, y, meta }
}

// ---------------------------------------------------------------- models

#[derive(Clone, Debug)]
pub struct HurCfg {
    pub ensemble: usize,
    pub hidden_grid: Vec<Vec<usize>>,
    pub epochs: usize,
    pub patience: usize,
    pub lr: f64,
    pub batch: usize,
    pub weight_decay: f64,
    pub ridge_lambda: f64,
    pub seed: u64,
}

impl Default for HurCfg {
    fn default() -> Self {
        HurCfg { ensemble: 3, hidden_grid: vec![vec![32, 16], vec![64, 32]], epochs: 120, patience: 12, lr: 3e-3, batch: 64, weight_decay: 1e-2, ridge_lambda: 10.0, seed: 7 }
    }
}
impl HurCfg {
    pub fn quick() -> Self {
        HurCfg { ensemble: 1, hidden_grid: vec![vec![32, 16]], epochs: 20, patience: 5, ..HurCfg::default() }
    }
    fn mlp(&self, hidden: &[usize]) -> MlpConfig {
        MlpConfig { hidden: hidden.to_vec(), lr: self.lr, epochs: self.epochs, batch: self.batch, weight_decay: self.weight_decay, patience: self.patience }
    }
}

const BLEND_GRID: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Pairwise products (incl. squares) of the first 12 standardised features: position, intensity,
/// season and motion terms, as in classical CLIPER regressions.
fn expand(x: &[f64]) -> Vec<f64> {
    let mut o = x.to_vec();
    for i in 0..12 {
        for j in i..12 {
            o.push(x[i] * x[j]);
        }
    }
    o
}
fn expand_mat(m: &Mat) -> Mat {
    let mut o = Mat::default();
    for i in 0..m.r {
        o.push(&expand(m.row(i)));
    }
    o
}

struct LeadModel {
    xs: Standardizer,
    ysc: Vec<f64>,
    ridge: Mat,   // plain linear baseline (CLIPER-like)
    xs2: Standardizer,
    ridge2: Mat,  // ridge on polynomial/interaction features, lambda chosen on validation storms
    mlp: Ensemble,
    w_track: f64, // weight on the MLP in the final track forecast (rest: ridge2)
    w_int: f64,
}

fn scale_y(y: &Mat, sc: &[f64]) -> Mat {
    let mut o = Mat::with_cols(y.c);
    for i in 0..y.r {
        o.push(&y.row(i).iter().enumerate().map(|(j, v)| v / sc[j]).collect::<Vec<_>>());
    }
    o
}

fn val_mse(pred: impl Fn(usize) -> Vec<f64>, y: &Mat) -> f64 {
    (0..y.r).map(|i| pred(i).iter().zip(y.row(i)).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / 3.0).sum::<f64>() / y.r.max(1) as f64
}

impl LeadModel {
    fn train(tr: &LeadData, va: &LeadData, cfg: &HurCfg, seed: u64) -> LeadModel {
        let xs = Standardizer::fit(&tr.x);
        let (xtr, xva) = (xs.transform(&tr.x), xs.transform(&va.x));
        let ysc: Vec<f64> = (0..3)
            .map(|j| {
                let s = std_dev(&(0..tr.y.r).map(|i| tr.y.at(i, j)).collect::<Vec<_>>());
                if s < 1e-8 { 1.0 } else { s }
            })
            .collect();
        let (ytr, yva) = (scale_y(&tr.y, &ysc), scale_y(&va.y, &ysc));
        let ridge = ridge_fit(&xtr, &ytr, cfg.ridge_lambda);
        // polynomial ridge: lambda chosen by validation MSE
        let (etr0, eva0) = (expand_mat(&xtr), expand_mat(&xva));
        let xs2 = Standardizer::fit(&etr0);
        let (etr, eva) = (xs2.transform(&etr0), xs2.transform(&eva0));
        let mut best2: Option<(f64, Mat)> = None;
        for &l in &[30.0, 300.0, 3000.0] {
            let w = ridge_fit(&etr, &ytr, l);
            let mse = val_mse(|i| ridge_predict(&w, eva.row(i)), &yva);
            if best2.as_ref().map_or(true, |b| mse < b.0) {
                best2 = Some((mse, w));
            }
        }
        let ridge2 = best2.unwrap().1;
        // MLP architecture chosen by validation MSE of the ensemble mean
        let mut best: Option<(f64, Ensemble)> = None;
        for (k, h) in cfg.hidden_grid.iter().enumerate() {
            let e = Ensemble::fit(&xtr, &ytr, &xva, &yva, Loss::Mse, &cfg.mlp(h), cfg.ensemble, seed + 100 * k as u64);
            let mse = val_mse(|i| e.predict_row(xva.row(i)), &yva);
            if best.as_ref().map_or(true, |b| mse < b.0) {
                best = Some((mse, e));
            }
        }
        let mlp = best.unwrap().1;
        let mut m = LeadModel { xs, ysc, ridge, xs2, ridge2, mlp, w_track: 1.0, w_int: 1.0 };
        // blend weights (MLP vs polynomial ridge) chosen on validation storms
        let preds: Vec<[[f64; 3]; 3]> = (0..va.x.r).map(|i| m.predict_all(va.x.row(i))).collect();
        let (mut bt, mut bi) = (f64::INFINITY, f64::INFINITY);
        for &w in &BLEND_GRID {
            let (mut et, mut ei) = (0.0, 0.0);
            for (meta, p) in va.meta.iter().zip(&preds) {
                let bl = LeadModel::mix(&p[1], &p[2], w, w);
                et += meta.track_err(bl[0], bl[1]);
                ei += meta.int_err(bl[2]);
            }
            if et < bt - 1e-9 { bt = et; m.w_track = w; }
            if ei < bi - 1e-9 { bi = ei; m.w_int = w; }
        }
        m
    }

    fn mix(r2: &[f64; 3], a: &[f64; 3], wt: f64, wi: f64) -> [f64; 3] {
        [wt * a[0] + (1.0 - wt) * r2[0], wt * a[1] + (1.0 - wt) * r2[1], wi * a[2] + (1.0 - wi) * r2[2]]
    }

    /// [plain ridge, polynomial ridge, mlp] residual predictions in physical units [east km, north km, wind kt].
    fn predict_all(&self, x_raw: &[f64]) -> [[f64; 3]; 3] {
        let x = self.xs.transform_row(x_raw);
        let r = ridge_predict(&self.ridge, &x);
        let r2 = ridge_predict(&self.ridge2, &self.xs2.transform_row(&expand(&x)));
        let a = self.mlp.predict_row(&x);
        let mut out = [[0.0; 3]; 3];
        for j in 0..3 {
            out[0][j] = r[j] * self.ysc[j];
            out[1][j] = r2[j] * self.ysc[j];
            out[2][j] = a[j] * self.ysc[j];
        }
        out
    }
    fn blended(&self, p: &[[f64; 3]; 3]) -> [f64; 3] {
        LeadModel::mix(&p[1], &p[2], self.w_track, self.w_int)
    }
}

// ---------------------------------------------------------------- evaluation

#[derive(Default, Clone)]
pub struct LeadRecs {
    pub storm: Vec<usize>,
    /// track error km: persistence, constant-velocity, ridge, mlp, ai-blend, polynomial ridge
    pub trk: [Vec<f64>; 6],
    /// intensity abs error kt: persistence, ridge, mlp, ai-blend, polynomial ridge
    pub int: [Vec<f64>; 5],
    pub w_track: Vec<f64>,
    pub w_int: Vec<f64>,
}

#[derive(Default)]
pub struct RiRecs {
    pub p_ai: Vec<f64>,
    pub p_clim: Vec<f64>,
    pub y: Vec<bool>,
}

pub struct SplitOut {
    pub recs: Vec<LeadRecs>,
    pub ri: RiRecs,
    pub n_train_storms: usize,
    pub n_test_storms: usize,
}

pub fn evaluate_split(tracks: &[Track], train_end: i32, test_from: i32, test_to: i32, cfg: &HurCfg) -> SplitOut {
    let train: Vec<&Track> = tracks.iter().filter(|t| t.year <= train_end).collect();
    let test: Vec<&Track> = tracks.iter().filter(|t| t.year >= test_from && t.year <= test_to).collect();
    let nv = train.len() / 5;
    let (fit, val) = train.split_at(train.len() - nv); // validation = most recent 20% of storms
    let mut recs: Vec<LeadRecs> = Vec::new();
    for (li, &steps) in LEAD_STEPS.iter().enumerate() {
        let (dtr, dva, dte) = (build_lead(fit, steps), build_lead(val, steps), build_lead(&test, steps));
        let m = LeadModel::train(&dtr, &dva, cfg, cfg.seed + 1000 * li as u64);
        let mut r = LeadRecs::default();
        for i in 0..dte.x.r {
            let meta = &dte.meta[i];
            let p = m.predict_all(dte.x.row(i));
            let (rd, r2, ml) = (p[0], p[1], p[2]);
            let ai = m.blended(&p);
            r.storm.push(meta.storm);
            r.trk[0].push(haversine_km(meta.lat0, meta.lon0, meta.lat_t, meta.lon_t));
            r.trk[1].push(meta.track_err(0.0, 0.0));
            r.trk[2].push(meta.track_err(rd[0], rd[1]));
            r.trk[3].push(meta.track_err(ml[0], ml[1]));
            r.trk[4].push(meta.track_err(ai[0], ai[1]));
            r.trk[5].push(meta.track_err(r2[0], r2[1]));
            r.int[0].push(meta.int_err(0.0));
            r.int[1].push(meta.int_err(rd[2]));
            r.int[2].push(meta.int_err(ml[2]));
            r.int[3].push(meta.int_err(ai[2]));
            r.int[4].push(meta.int_err(r2[2]));
        }
        r.w_track.push(m.w_track);
        r.w_int.push(m.w_int);
        recs.push(r);
    }
    // ---- rapid-intensification probability (24 h) ----
    let build_ri = |ts: &[&Track]| {
        let (mut x, mut y) = (Mat::default(), Mat::default());
        let mut lab = Vec::new();
        for tr in ts {
            for i in HIST..tr.fixes.len() {
                if tr.fixes[i].status > 3 || i + RI_STEPS >= tr.fixes.len() {
                    continue;
                }
                let ri = tr.fixes[i + RI_STEPS].wind - tr.fixes[i].wind >= RI_KT;
                x.push(&features(tr, i));
                y.push(&if ri { [0.0, 1.0] } else { [1.0, 0.0] });
                lab.push(ri);
            }
        }
        (x, y, lab)
    };
    let (xf, yf, lf) = build_ri(fit);
    let (xv, yv, lv) = build_ri(val);
    let (xt, _yt, lt) = build_ri(&test);
    let xs = Standardizer::fit(&xf);
    let (xf, xv, xt) = (xs.transform(&xf), xs.transform(&xv), xs.transform(&xt));
    let ens = Ensemble::fit(&xf, &yf, &xv, &yv, Loss::SoftmaxCe, &cfg.mlp(&cfg.hidden_grid[0]), cfg.ensemble, cfg.seed + 9000);
    let base = lf.iter().filter(|&&b| b).count() as f64 / lf.len().max(1) as f64;
    let pv: Vec<f64> = (0..xv.r).map(|i| ens.predict_row(xv.row(i))[1]).collect();
    let (mut bw, mut bb) = (1.0, f64::INFINITY);
    for &w in &BLEND_GRID {
        let b = brier(&pv.iter().map(|p| w * p + (1.0 - w) * base).collect::<Vec<_>>(), &lv);
        if b < bb - 1e-12 { bb = b; bw = w; }
    }
    let ri = RiRecs {
        p_ai: (0..xt.r).map(|i| bw * ens.predict_row(xt.row(i))[1] + (1.0 - bw) * base).collect(),
        p_clim: vec![base; xt.r],
        y: lt,
    };
    SplitOut { recs, ri, n_train_storms: train.len(), n_test_storms: test.len() }
}

/// Storm-level bootstrap of mean(diff): returns (mean, lo95, hi95).
pub fn boot_ci(storm: &[usize], diff: &[f64], reps: usize, seed: u64) -> (f64, f64, f64) {
    let mut by: HashMap<usize, (f64, f64)> = HashMap::new();
    for (s, d) in storm.iter().zip(diff) {
        let e = by.entry(*s).or_insert((0.0, 0.0));
        e.0 += d;
        e.1 += 1.0;
    }
    let groups: Vec<(f64, f64)> = by.into_values().collect();
    let tot = groups.iter().map(|g| g.0).sum::<f64>() / groups.iter().map(|g| g.1).sum::<f64>();
    let mut rng = Rng::new(seed);
    let mut means: Vec<f64> = (0..reps)
        .map(|_| {
            let (mut s, mut c) = (0.0, 0.0);
            for _ in 0..groups.len() {
                let g = groups[rng.below(groups.len())];
                s += g.0;
                c += g.1;
            }
            s / c
        })
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (tot, means[(reps as f64 * 0.025) as usize], means[(reps as f64 * 0.975) as usize])
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Full chronological evaluation (three consecutive test periods) and markdown report.
pub fn evaluate_all(tracks: &[Track], cfg: &HurCfg, splits: &[(i32, i32, i32)]) -> String {
    let mut md = String::new();
    let mut pooled: Vec<LeadRecs> = vec![LeadRecs::default(); LEAD_STEPS.len()];
    let mut ri = RiRecs::default();
    let _ = writeln!(md, "# AI weather system — tropical-cyclone track & intensity forecasting\n");
    let _ = writeln!(md, "Data: NOAA HURDAT2-derived best tracks (dplyr `storms`), Atlantic, 6-hourly, {} storm segments, {}–{}.", tracks.len(), tracks.iter().map(|t| t.year).min().unwrap(), tracks.iter().map(|t| t.year).max().unwrap());
    let _ = writeln!(md, "Forecast origins: tropical/subtropical-cyclone fixes with ≥24 h of history, verified against the best track at the lead time (storm must still exist then). Splits are by storm season; validation = most recent 20% of training storms (architecture and blend weights are chosen there, never on test data).\n");
    let _ = writeln!(md, "Config: ensemble={}, hidden grid={:?}, epochs<={}, seed={}\n", cfg.ensemble, cfg.hidden_grid, cfg.epochs, cfg.seed);
    for &(train_end, a, b) in splits {
        eprintln!("split: train <= {train_end}, test {a}-{b}");
        let o = evaluate_split(tracks, train_end, a, b, cfg);
        let _ = writeln!(md, "## Split: train ≤{train_end} ({} storms) → test {a}–{b} ({} storms)\n", o.n_train_storms, o.n_test_storms);
        let _ = writeln!(md, "| Lead | n | Track km: CV | CLIPER-like ridge | AI blend | Intensity MAE kt: persist. | AI blend | blend weight on MLP vs poly ridge (track / int.) |\n|---|---|---|---|---|---|---|---|");
        for (li, r) in o.recs.iter().enumerate() {
            let _ = writeln!(md, "| {}h | {} | {:.0} | {:.0} | {:.0} | {:.1} | {:.1} | {:.2} / {:.2} |", LEAD_STEPS[li] * 6, r.storm.len(), mean(&r.trk[1]), mean(&r.trk[2]), mean(&r.trk[4]), mean(&r.int[0]), mean(&r.int[3]), r.w_track[0], r.w_int[0]);
            let p = &mut pooled[li];
            p.storm.extend(&r.storm);
            for k in 0..6 { p.trk[k].extend(&r.trk[k]); }
            for k in 0..5 { p.int[k].extend(&r.int[k]); }
        }
        let _ = writeln!(md);
        ri.p_ai.extend(&o.ri.p_ai);
        ri.p_clim.extend(&o.ri.p_clim);
        ri.y.extend(&o.ri.y);
    }
    let _ = writeln!(md, "## Pooled over all test periods (95% CI from a storm-level bootstrap, 1000 resamples)\n");
    let _ = writeln!(md, "### Track error (km, lower is better)\n");
    let _ = writeln!(md, "| Lead | n | Persistence | Const.-velocity | Ridge (CLIPER-like) | Poly ridge | MLP | **AI blend** | AI − CV (km) [95% CI] | Skill vs CV | AI − plain ridge (km) [95% CI] |\n|---|---|---|---|---|---|---|---|---|---|---|");
    for (li, p) in pooled.iter().enumerate() {
        let d_cv: Vec<f64> = p.trk[4].iter().zip(&p.trk[1]).map(|(a, b)| a - b).collect();
        let d_rg: Vec<f64> = p.trk[4].iter().zip(&p.trk[2]).map(|(a, b)| a - b).collect();
        let (m1, l1, h1) = boot_ci(&p.storm, &d_cv, 1000, 11 + li as u64);
        let (m2, l2, h2) = boot_ci(&p.storm, &d_rg, 1000, 21 + li as u64);
        let _ = writeln!(md, "| {}h | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | **{:.0}** | {:+.1} [{:+.1}, {:+.1}] | {:+.1}% | {:+.1} [{:+.1}, {:+.1}] |", LEAD_STEPS[li] * 6, p.storm.len(), mean(&p.trk[0]), mean(&p.trk[1]), mean(&p.trk[2]), mean(&p.trk[5]), mean(&p.trk[3]), mean(&p.trk[4]), m1, l1, h1, 100.0 * (1.0 - mean(&p.trk[4]) / mean(&p.trk[1])), m2, l2, h2);
    }
    let _ = writeln!(md, "\n### Intensity error (kt MAE of maximum sustained wind, lower is better)\n");
    let _ = writeln!(md, "| Lead | n | Persistence | Ridge | Poly ridge | MLP | **AI blend** | AI − persistence (kt) [95% CI] | Skill vs persistence |\n|---|---|---|---|---|---|---|---|---|");
    for (li, p) in pooled.iter().enumerate() {
        let d: Vec<f64> = p.int[3].iter().zip(&p.int[0]).map(|(a, b)| a - b).collect();
        let (m, l, h) = boot_ci(&p.storm, &d, 1000, 31 + li as u64);
        let _ = writeln!(md, "| {}h | {} | {:.2} | {:.2} | {:.2} | {:.2} | **{:.2}** | {:+.2} [{:+.2}, {:+.2}] | {:+.1}% |", LEAD_STEPS[li] * 6, p.storm.len(), mean(&p.int[0]), mean(&p.int[1]), mean(&p.int[4]), mean(&p.int[2]), mean(&p.int[3]), m, l, h, 100.0 * (1.0 - mean(&p.int[3]) / mean(&p.int[0])));
    }
    let n = ri.y.len();
    let pos = ri.y.iter().filter(|&&b| b).count();
    let b_ai = brier(&ri.p_ai, &ri.y);
    let b_cl = brier(&ri.p_clim, &ri.y);
    let thr = 0.2;
    let tp = (0..n).filter(|&i| ri.p_ai[i] >= thr && ri.y[i]).count();
    let fp = (0..n).filter(|&i| ri.p_ai[i] >= thr && !ri.y[i]).count();
    let _ = writeln!(md, "\n### Rapid intensification (≥{RI_KT:.0} kt gain in 24 h)\n");
    let _ = writeln!(md, "n={n}, events={pos} ({:.1}%). Brier: AI {:.4} vs climatology {:.4} (skill {:+.1}%); AUC {:.3}. At p≥{thr}: hits {tp}/{pos} (POD {:.2}), false alarms {fp} (precision {:.2}).", 100.0 * pos as f64 / n.max(1) as f64, b_ai, b_cl, 100.0 * (1.0 - b_ai / b_cl), auc(&ri.p_ai, &ri.y), tp as f64 / pos.max(1) as f64, tp as f64 / (tp + fp).max(1) as f64);
    md
}
