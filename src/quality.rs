//! Sensor-fault / extreme-anomaly detection from 1-day-ahead forecast residuals, evaluated by
//! injecting isolated synthetic faults (+/- m * sigma) into the held-out series.

use crate::metrics::auc;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct FaultResult {
    pub magnitude: f64,
    pub auc_model: f64,
    pub auc_clim: f64,
    pub prec_model: f64,
    pub rec_model: f64,
    pub prec_clim: f64,
    pub rec_clim: f64,
}

/// `obs`, `pred` (1-day-ahead model forecast from clean history) and `clim` are aligned test series.
/// Faults are 5% of days, never adjacent. Averaged over `reps` random draws. Flag threshold: z > 3.
pub fn fault_detection(obs: &[f64], pred: &[f64], clim: &[f64], sig_model: f64, sig_clim: f64, mags: &[f64], reps: usize, seed: u64) -> Vec<FaultResult> {
    let n = obs.len();
    let mut out = Vec::new();
    for &m in mags {
        let mut acc = [0.0f64; 6];
        let mut cnt: f64 = 0.0;
        for r in 0..reps {
            let mut rng = Rng::new(seed + r as u64 * 31 + (m * 100.0) as u64);
            let mut fault = vec![false; n];
            for i in 0..n {
                if rng.f64() < 0.05 && !(i > 0 && fault[i - 1]) {
                    fault[i] = true;
                }
            }
            let (mut sm, mut sc) = (Vec::new(), Vec::new());
            for i in 0..n {
                let mut o = obs[i];
                if fault[i] {
                    o += if rng.f64() < 0.5 { 1.0 } else { -1.0 } * m * sig_clim;
                }
                sm.push((o - pred[i]).abs() / sig_model);
                sc.push((o - clim[i]).abs() / sig_clim);
            }
            let pr = |s: &[f64]| {
                let tp = (0..n).filter(|&i| s[i] > 3.0 && fault[i]).count() as f64;
                let fp = (0..n).filter(|&i| s[i] > 3.0 && !fault[i]).count() as f64;
                let fneg = (0..n).filter(|&i| s[i] <= 3.0 && fault[i]).count() as f64;
                (if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 }, if tp + fneg > 0.0 { tp / (tp + fneg) } else { 0.0 })
            };
            let (pm, rm) = pr(&sm);
            let (pc, rc) = pr(&sc);
            let (a1, a2) = (auc(&sm, &fault), auc(&sc, &fault));
            if a1.is_nan() { continue; }
            for (k, v) in [a1, a2, pm, rm, pc, rc].iter().enumerate() {
                acc[k] += v;
            }
            cnt += 1.0;
        }
        let c = cnt.max(1.0);
        out.push(FaultResult { magnitude: m, auc_model: acc[0] / c, auc_clim: acc[1] / c, prec_model: acc[2] / c, rec_model: acc[3] / c, prec_clim: acc[4] / c, rec_clim: acc[5] / c });
    }
    out
}
