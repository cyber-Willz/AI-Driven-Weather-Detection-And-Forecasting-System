//! Verification metrics.

pub fn mae(obs: &[f64], pred: &[f64]) -> f64 {
    obs.iter().zip(pred).map(|(o, p)| (o - p).abs()).sum::<f64>() / obs.len().max(1) as f64
}
pub fn mse(obs: &[f64], pred: &[f64]) -> f64 {
    obs.iter().zip(pred).map(|(o, p)| (o - p).powi(2)).sum::<f64>() / obs.len().max(1) as f64
}
pub fn rmse(obs: &[f64], pred: &[f64]) -> f64 {
    mse(obs, pred).sqrt()
}
/// 1 - MSE_model / MSE_reference (positive = better than the reference).
pub fn skill(mse_model: f64, mse_ref: f64) -> f64 {
    if mse_ref <= 0.0 { 0.0 } else { 1.0 - mse_model / mse_ref }
}
pub fn brier(p: &[f64], y: &[bool]) -> f64 {
    p.iter().zip(y).map(|(p, &y)| (p - if y { 1.0 } else { 0.0 }).powi(2)).sum::<f64>() / p.len().max(1) as f64
}
pub fn log_loss(p: &[f64], y: &[bool]) -> f64 {
    p.iter().zip(y).map(|(&p, &y)| { let p = p.clamp(1e-6, 1.0 - 1e-6); -(if y { p.ln() } else { (1.0 - p).ln() }) }).sum::<f64>() / p.len().max(1) as f64
}
/// ROC AUC via the rank-sum statistic (ties get average rank).
pub fn auc(scores: &[f64], y: &[bool]) -> f64 {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = r;
        }
        i = j + 1;
    }
    let np = y.iter().filter(|&&v| v).count() as f64;
    let nn = n as f64 - np;
    if np == 0.0 || nn == 0.0 {
        return f64::NAN;
    }
    let sr: f64 = (0..n).filter(|&i| y[i]).map(|i| ranks[i]).sum();
    (sr - np * (np + 1.0) / 2.0) / (np * nn)
}

pub fn accuracy(y: &[usize], p: &[usize]) -> f64 {
    y.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / y.len().max(1) as f64
}

/// conf[true][pred]
pub fn confusion(k: usize, y: &[usize], p: &[usize]) -> Vec<Vec<usize>> {
    let mut c = vec![vec![0usize; k]; k];
    for (a, b) in y.iter().zip(p) {
        c[*a][*b] += 1;
    }
    c
}

/// Per-class (precision, recall, f1, support).
pub fn prf(conf: &[Vec<usize>]) -> Vec<(f64, f64, f64, usize)> {
    let k = conf.len();
    (0..k)
        .map(|c| {
            let tp = conf[c][c] as f64;
            let fp: f64 = (0..k).filter(|&r| r != c).map(|r| conf[r][c] as f64).sum();
            let fneg: f64 = (0..k).filter(|&p| p != c).map(|p| conf[c][p] as f64).sum();
            let pr = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let rc = if tp + fneg > 0.0 { tp / (tp + fneg) } else { 0.0 };
            let f1 = if pr + rc > 0.0 { 2.0 * pr * rc / (pr + rc) } else { 0.0 };
            (pr, rc, f1, conf[c].iter().sum())
        })
        .collect()
}

/// Macro-F1 over classes that occur in the ground truth.
pub fn macro_f1(conf: &[Vec<usize>]) -> f64 {
    let s: Vec<(f64, f64, f64, usize)> = prf(conf).into_iter().filter(|t| t.3 > 0).collect();
    s.iter().map(|t| t.2).sum::<f64>() / s.len().max(1) as f64
}
