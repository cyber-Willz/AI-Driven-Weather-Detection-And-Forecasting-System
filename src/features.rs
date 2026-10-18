//! Leakage-safe feature construction. A forecast issued at day `t` for day `t+h` only uses
//! observations up to and including day `t`, plus the calendar position of day `t+h`.

use crate::climatology::{basis, to_z, Climatology, N_BASIS};
use crate::data::{Dataset, Kind};

/// First origin index with a full 30-day history window.
pub const MIN_T: usize = 29;
pub const LAGS: usize = 7;
pub const WINDOWS: [usize; 4] = [3, 7, 14, 30];

/// Pre-computed transformed values (z) and climatology anomalies (a) per channel.
pub struct Context {
    pub z: Vec<Vec<f64>>,
    pub a: Vec<Vec<f64>>,
    /// Anomalies of the input-only (aux) series.
    pub aux_a: Vec<Vec<f64>>,
}

impl Context {
    pub fn new(ds: &Dataset, clim: &Climatology) -> Self {
        let mut z = Vec::new();
        let mut a = Vec::new();
        for (c, ch) in ds.ch.iter().enumerate() {
            let zc: Vec<f64> = ch.v.iter().map(|&v| to_z(ch.kind, v)).collect();
            let ac: Vec<f64> = zc.iter().enumerate().map(|(i, &v)| v - clim.z(c, ds.dates[i])).collect();
            z.push(zc);
            a.push(ac);
        }
        let aux_a = ds
            .aux
            .iter()
            .enumerate()
            .map(|(i, x)| x.v.iter().enumerate().map(|(t, &v)| v - clim.aux_value(i, ds.dates[t])).collect())
            .collect();
        Context { z, a, aux_a }
    }
}

/// Features for a forecast issued at origin `t` for horizon `h` days ahead.
pub fn forecast_features(ds: &Dataset, ctx: &Context, t: usize, h: usize) -> Vec<f64> {
    assert!(t >= MIN_T && t < ds.len());
    let mut f = Vec::with_capacity(96);
    f.extend_from_slice(&basis(ds.dates[t].plus(h as i64)));
    for (c, ch) in ds.ch.iter().enumerate() {
        for l in 0..LAGS {
            f.push(ctx.a[c][t - l]);
        }
        for &w in &WINDOWS {
            let s: f64 = ctx.a[c][t + 1 - w..=t].iter().sum();
            f.push(s / w as f64);
        }
        if ch.kind == Kind::Precip {
            let wet7 = (0..7).filter(|&i| ch.v[t - i] > 0.0).count() as f64;
            let wet30 = (0..30).filter(|&i| ch.v[t - i] > 0.0).count() as f64;
            let dsw = (0..14).find(|&i| ch.v[t - i] > 0.0).unwrap_or(14) as f64;
            f.extend_from_slice(&[wet7, wet30, dsw]);
        }
    }
    // Input-only series: neighbour temperature anomalies (today, yesterday) and pressure
    // anomaly + 1-day tendency (front passage / upstream advection proxies).
    for (i, x) in ds.aux.iter().enumerate() {
        let a = &ctx.aux_a[i];
        match x.kind {
            crate::data::AuxKind::Temp => f.extend_from_slice(&[a[t], a[t - 1]]),
            crate::data::AuxKind::Pressure => f.extend_from_slice(&[a[t], a[t] - a[t - 1]]),
        }
    }
    if let Some(lab) = &ds.labels {
        for k in 0..ds.label_names.len() {
            f.push(if lab[t] == k { 1.0 } else { 0.0 });
        }
    }
    f
}

/// Anomaly targets (z-space minus climatology) for every channel at day `t+h`.
pub fn anomaly_target(ctx: &Context, t: usize, h: usize) -> Vec<f64> {
    ctx.a.iter().map(|a| a[t + h]).collect()
}

/// Features describing *today's observations only* (used by the condition detector).
pub fn detection_features(ds: &Dataset, ctx: &Context, t: usize) -> Vec<f64> {
    let mut f = Vec::with_capacity(2 * ds.ch.len() + N_BASIS);
    for c in 0..ds.ch.len() {
        f.push(ctx.z[c][t]);
        f.push(ctx.a[c][t]);
        if ds.ch[c].kind == Kind::Precip {
            // sharp "any precipitation recorded" cue (tanh units struggle to carve this from log1p)
            f.push(if ds.ch[c].v[t] > 0.0 { 1.0 } else { 0.0 });
        }
    }
    f.extend_from_slice(&basis(ds.dates[t]));
    f
}
