//! Multi-horizon AI forecaster: for every lead time h = 1..=H it trains
//!  * an MLP-ensemble regressor for the anomalies of all channels (multi-task),
//!  * a ridge-regression baseline on identical features,
//!  * an MLP-ensemble rain/no-rain probability classifier (if a precipitation channel exists),
//!  * an MLP-ensemble weather-condition classifier (if labels exist).

use crate::climatology::{from_z, Climatology};
use crate::data::{Dataset, Kind};
use crate::features::{anomaly_target, forecast_features, Context, MIN_T};
use crate::linalg::{ridge_fit, ridge_predict, std_dev, Mat, Standardizer};
use crate::mlp::{Ensemble, Loss, MlpConfig};

#[derive(Clone, Debug)]
pub struct TrainCfg {
    pub horizons: usize,
    pub ensemble: usize,
    pub mlp: MlpConfig,
    pub ridge_lambda: f64,
    pub seed: u64,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg { horizons: 7, ensemble: 5, mlp: MlpConfig::default(), ridge_lambda: 30.0, seed: 42 }
    }
}

impl TrainCfg {
    /// Faster settings for tests / smoke runs.
    pub fn quick() -> Self {
        TrainCfg { horizons: 3, ensemble: 2, mlp: MlpConfig { epochs: 40, patience: 8, ..MlpConfig::default() }, ridge_lambda: 30.0, seed: 42 }
    }
}

#[derive(Clone, Debug)]
pub struct HorizonModel {
    pub h: usize,
    pub xs: Standardizer,
    pub y_scale: Vec<f64>,
    pub reg: Ensemble,
    pub ridge: Mat,
    pub rain: Option<Ensemble>,
    /// Weight on the MLP rain probability; the remainder goes to monthly climatology (chosen on validation Brier).
    pub rain_w: f64,
    pub month_wet: [f64; 13],
    pub cond: Option<Ensemble>,
}

#[derive(Clone, Debug)]
pub struct Forecaster {
    pub clim: Climatology,
    pub models: Vec<HorizonModel>,
    pub rain_channel: Option<usize>,
    pub label_names: Vec<String>,
}

/// One forecast for one channel set at one lead time.
#[derive(Clone, Debug)]
pub struct Point {
    pub h: usize,
    /// Physical-unit forecast per channel from the MLP ensemble.
    pub mlp: Vec<f64>,
    /// Physical-unit forecast per channel from the ridge baseline.
    pub ridge: Vec<f64>,
    /// Climatological value per channel.
    pub clim: Vec<f64>,
    pub p_rain: Option<f64>,
    pub cond_probs: Option<Vec<f64>>,
}

fn onehot(k: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[k] = 1.0;
    v
}

impl Forecaster {
    /// Train on all samples whose *target* index is <= `train_last`. The most recent 20% of the
    /// training samples (chronologically) form the early-stopping validation set.
    pub fn train(ds: &Dataset, clim: &Climatology, ctx: &Context, train_last: usize, cfg: &TrainCfg) -> Forecaster {
        let rain_channel = ds.precip_channel();
        let mut models = Vec::new();
        for h in 1..=cfg.horizons {
            let ts: Vec<usize> = (MIN_T..=train_last.saturating_sub(h)).collect();
            assert!(ts.len() > 50, "not enough training samples");
            let n_val = ts.len() / 5;
            let split = ts.len() - n_val;
            let build = |idx: &[usize]| {
                let mut x = Mat::default();
                let mut y = Mat::default();
                let mut yr = Mat::default();
                let mut yc = Mat::default();
                for &t in idx {
                    x.push(&forecast_features(ds, ctx, t, h));
                    y.push(&anomaly_target(ctx, t, h));
                    if let Some(rc) = rain_channel {
                        yr.push(&onehot(if ds.ch[rc].v[t + h] > 0.0 { 1 } else { 0 }, 2));
                    }
                    if let Some(l) = &ds.labels {
                        yc.push(&onehot(l[t + h], ds.label_names.len()));
                    }
                }
                (x, y, yr, yc)
            };
            let (x_tr, y_tr, yr_tr, yc_tr) = build(&ts[..split]);
            let (x_va, y_va, yr_va, yc_va) = build(&ts[split..]);
            let xs = Standardizer::fit(&x_tr);
            let (xtr, xva) = (xs.transform(&x_tr), xs.transform(&x_va));
            // Scale each channel's anomaly target to unit variance so the multi-task MSE is balanced.
            let y_scale: Vec<f64> = (0..y_tr.c)
                .map(|j| {
                    let col: Vec<f64> = (0..y_tr.r).map(|i| y_tr.at(i, j)).collect();
                    let s = std_dev(&col);
                    if s < 1e-8 { 1.0 } else { s }
                })
                .collect();
            let scale = |m: &Mat| {
                let mut o = Mat::with_cols(m.c);
                for i in 0..m.r {
                    let r: Vec<f64> = m.row(i).iter().enumerate().map(|(j, v)| v / y_scale[j]).collect();
                    o.push(&r);
                }
                o
            };
            let (ytr_s, yva_s) = (scale(&y_tr), scale(&y_va));
            let seed = cfg.seed.wrapping_add(1000 * h as u64);
            let reg = Ensemble::fit(&xtr, &ytr_s, &xva, &yva_s, Loss::Mse, &cfg.mlp, cfg.ensemble, seed);
            let ridge = ridge_fit(&xtr, &ytr_s, cfg.ridge_lambda);
            let rain = rain_channel.map(|_| Ensemble::fit(&xtr, &yr_tr, &xva, &yr_va, Loss::SoftmaxCe, &cfg.mlp, cfg.ensemble, seed + 1));
            let mut month_wet = [0.5f64; 13];
            if let Some(rc) = rain_channel {
                let (mut w, mut tt) = ([0.0f64; 13], [0.0f64; 13]);
                for i in 0..=train_last {
                    let m = ds.dates[i].m as usize;
                    tt[m] += 1.0;
                    if ds.ch[rc].v[i] > 0.0 { w[m] += 1.0; }
                }
                for m in 1..13 { if tt[m] > 0.0 { month_wet[m] = w[m] / tt[m]; } }
            }
            let mut rain_w = 1.0;
            if let (Some(e), Some(rc)) = (&rain, rain_channel) {
                let vt = &ts[split..];
                let pm: Vec<f64> = (0..xva.r).map(|i| e.predict_row(xva.row(i))[1]).collect();
                let mut best = f64::INFINITY;
                for k in 0..=10 {
                    let w = k as f64 / 10.0;
                    let b: f64 = vt.iter().zip(&pm).map(|(&t, &p)| {
                        let pc = month_wet[ds.dates[t + h].m as usize];
                        let q = w * p + (1.0 - w) * pc;
                        let y = if ds.ch[rc].v[t + h] > 0.0 { 1.0 } else { 0.0 };
                        (q - y).powi(2)
                    }).sum::<f64>() / vt.len().max(1) as f64;
                    if b < best - 1e-12 { best = b; rain_w = w; }
                }
            }
            let cond = ds.labels.as_ref().map(|_| Ensemble::fit(&xtr, &yc_tr, &xva, &yc_va, Loss::SoftmaxCe, &cfg.mlp, cfg.ensemble, seed + 2));
            models.push(HorizonModel { h, xs, y_scale, reg, ridge, rain, rain_w, month_wet, cond });
        }
        Forecaster { clim: clim.clone(), models, rain_channel, label_names: ds.label_names.clone() }
    }

    pub fn predict(&self, ds: &Dataset, ctx: &Context, t: usize, h: usize) -> Point {
        let m = &self.models[h - 1];
        let x = m.xs.transform_row(&forecast_features(ds, ctx, t, h));
        let date = ds.dates[t].plus(h as i64);
        let a_mlp = m.reg.predict_row(&x);
        let a_rdg = ridge_predict(&m.ridge, &x);
        let nch = ds.ch.len();
        let mut mlp = Vec::with_capacity(nch);
        let mut ridge = Vec::with_capacity(nch);
        let mut clim = Vec::with_capacity(nch);
        for c in 0..nch {
            let kind: Kind = ds.ch[c].kind;
            let cz = self.clim.z(c, date);
            mlp.push(from_z(kind, cz + a_mlp[c] * m.y_scale[c]));
            ridge.push(from_z(kind, cz + a_rdg[c] * m.y_scale[c]));
            clim.push(from_z(kind, cz));
        }
        Point {
            h,
            mlp,
            ridge,
            clim,
            p_rain: m.rain.as_ref().map(|e| m.rain_w * e.predict_row(&x)[1] + (1.0 - m.rain_w) * m.month_wet[date.m as usize]),
            cond_probs: m.cond.as_ref().map(|e| e.predict_row(&x)),
        }
    }
}
