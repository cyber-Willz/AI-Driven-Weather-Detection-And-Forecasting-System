//! Smooth seasonal climatology: harmonic regression (3 harmonics) fitted per channel on
//! training data only. Precipitation is modelled in log1p space.

use crate::data::{Dataset, Date, Kind};
use crate::linalg::{ridge_fit, Mat};

pub const HARMONICS: usize = 3;
pub const N_BASIS: usize = 2 * HARMONICS;

/// Transform used for modelling (log1p for precipitation, identity otherwise).
pub fn to_z(kind: Kind, v: f64) -> f64 {
    if kind == Kind::Precip { v.max(0.0).ln_1p() } else { v }
}
pub fn from_z(kind: Kind, z: f64) -> f64 {
    if kind == Kind::Precip { z.exp_m1().max(0.0) } else { z }
}

pub fn basis(date: Date) -> [f64; N_BASIS] {
    let t = 2.0 * std::f64::consts::PI * (date.doy() as f64 - 1.0) / 365.25;
    let mut b = [0.0; N_BASIS];
    for k in 0..HARMONICS {
        let a = t * (k + 1) as f64;
        b[2 * k] = a.sin();
        b[2 * k + 1] = a.cos();
    }
    b
}

#[derive(Clone, Debug)]
pub struct Climatology {
    /// Per channel: intercept followed by N_BASIS harmonic coefficients (in z-space).
    pub coef: Vec<Vec<f64>>,
    pub kinds: Vec<Kind>,
    /// Std of (z - climatology) on the fitting data, per channel.
    pub anom_std: Vec<f64>,
    /// Harmonic coefficients for each input-only (aux) series, fitted on training data.
    pub aux_coef: Vec<Vec<f64>>,
}

impl Climatology {
    /// Fit on indices `0..end` (exclusive) of the dataset.
    pub fn fit(ds: &Dataset, end: usize) -> Self {
        let mut x = Mat::with_cols(N_BASIS);
        for i in 0..end {
            x.push(&basis(ds.dates[i]));
        }
        let mut coef = Vec::new();
        let mut anom_std = Vec::new();
        let mut kinds = Vec::new();
        for ch in &ds.ch {
            let mut y = Mat::with_cols(1);
            for i in 0..end {
                y.push(&[to_z(ch.kind, ch.v[i])]);
            }
            let w = ridge_fit(&x, &y, 1e-6);
            let c: Vec<f64> = (0..w.r).map(|i| w.at(i, 0)).collect();
            let mut ss = 0.0;
            for i in 0..end {
                let e = to_z(ch.kind, ch.v[i]) - eval(&c, ds.dates[i]);
                ss += e * e;
            }
            anom_std.push((ss / end.max(1) as f64).sqrt());
            coef.push(c);
            kinds.push(ch.kind);
        }
        let mut aux_coef = Vec::new();
        for a in &ds.aux {
            let mut y = Mat::with_cols(1);
            for i in 0..end {
                y.push(&[a.v[i]]);
            }
            let w = ridge_fit(&x, &y, 1e-6);
            aux_coef.push((0..w.r).map(|i| w.at(i, 0)).collect());
        }
        Climatology { coef, kinds, anom_std, aux_coef }
    }
    /// Climatological value in z-space.
    pub fn z(&self, ch: usize, date: Date) -> f64 {
        eval(&self.coef[ch], date)
    }
    /// Climatological value of aux series `i`.
    pub fn aux_value(&self, i: usize, date: Date) -> f64 {
        eval(&self.aux_coef[i], date)
    }
    /// Climatological value in physical units.
    pub fn value(&self, ch: usize, date: Date) -> f64 {
        from_z(self.kinds[ch], self.z(ch, date))
    }
}

fn eval(c: &[f64], date: Date) -> f64 {
    let b = basis(date);
    let mut s = c[0];
    for i in 0..N_BASIS {
        s += c[i + 1] * b[i];
    }
    s
}
