//! Chronological hold-out evaluation + markdown report generation.

use crate::climatology::Climatology;
use crate::data::{Dataset, Date};
use crate::detect::{argmax, ConditionDetector};
use crate::features::{Context, MIN_T};
use crate::linalg::std_dev;
use crate::metrics::*;
use crate::models::{Forecaster, TrainCfg};
use crate::quality::fault_detection;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct Split {
    pub name: String,
    pub train_end: Date,
    pub test_start: Date,
    pub test_end: Date,
}

/// Key numbers (also used by integration tests). Method order: [mlp, ridge, climatology, persistence].
#[derive(Clone, Debug, Default)]
pub struct SplitResult {
    pub md: String,
    pub mae_h1: Vec<[f64; 4]>,
    pub mae_hmax: Vec<[f64; 4]>,
    pub brier_h1: Option<[f64; 3]>, // mlp, monthly climatology, persistence
    pub detect_acc: Option<(f64, f64, f64)>, // mlp, majority, rule
    pub fault_auc_4sigma: Option<(f64, f64)>, // model-residual, climatology
}

fn f3(x: f64) -> String {
    format!("{:.3}", x)
}

pub fn evaluate_split(ds: &Dataset, split: &Split, cfg: &TrainCfg) -> SplitResult {
    let n = ds.len();
    let train_last = ds.last_index_on_or_before(split.train_end).expect("train_end before data");
    let test: Vec<usize> = (0..n).filter(|&i| ds.dates[i] >= split.test_start && ds.dates[i] <= split.test_end).collect();
    let (lo, hi) = (test[0], *test.last().unwrap());
    let clim = Climatology::fit(ds, train_last + 1);
    let ctx = Context::new(ds, &clim);
    let fc = Forecaster::train(ds, &clim, &ctx, train_last, cfg);
    let mut res = SplitResult::default();
    let md = &mut res.md;
    let _ = writeln!(md, "### {} — {}\n", ds.name, split.name);
    let _ = writeln!(md, "Train: {} .. {} ({} days) · Test: {} .. {} ({} days) · gap-filled days in source: {}\n", ds.dates[0], ds.dates[train_last], train_last + 1, ds.dates[lo], ds.dates[hi], test.len(), ds.n_filled);

    // ---------- forecasting ----------
    let nch = ds.ch.len();
    let _ = writeln!(md, "#### Forecast error by lead time (MAE / RMSE, physical units; lower is better)\n");
    let _ = writeln!(md, "| Variable | Lead | AI-MLP MAE | Ridge MAE | Clim. MAE | Persist. MAE | AI-MLP RMSE | Clim. RMSE | Persist. RMSE | Skill vs clim. (MSE) |");
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|");
    let mut rain_rows: Vec<String> = Vec::new();
    let mut cond_rows: Vec<String> = Vec::new();
    let mut h1_series: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
    res.mae_h1 = vec![[0.0; 4]; nch];
    res.mae_hmax = vec![[0.0; 4]; nch];
    let rain_c = ds.precip_channel();
    let tc = ds.primary_temp_channel();
    // monthly rain climatology from training data
    let mut wet = [0.0f64; 13];
    let mut tot = [0.0f64; 13];
    if let Some(rc) = rain_c {
        for i in 0..=train_last {
            tot[ds.dates[i].m as usize] += 1.0;
            if ds.ch[rc].v[i] > 0.0 { wet[ds.dates[i].m as usize] += 1.0; }
        }
    }
    let majority = ds.labels.as_ref().map(|l| {
        let mut c = vec![0usize; ds.label_names.len()];
        for i in 0..=train_last { c[l[i]] += 1; }
        (0..c.len()).max_by_key(|&k| c[k]).unwrap()
    });
    for h in 1..=cfg.horizons {
        let ts: Vec<usize> = (lo.saturating_sub(h).max(MIN_T)..=hi - h).filter(|&t| t + h >= lo).collect();
        let mut obs = vec![Vec::new(); nch];
        let mut pr: Vec<[Vec<f64>; 4]> = (0..nch).map(|_| [Vec::new(), Vec::new(), Vec::new(), Vec::new()]).collect();
        let (mut yr, mut p_m, mut p_c, mut p_p) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut yc, mut c_m, mut c_p) = (Vec::new(), Vec::new(), Vec::new());
        let mut tmp: Vec<f64> = Vec::new();
        for &t in &ts {
            let pt = fc.predict(ds, &ctx, t, h);
            for c in 0..nch {
                obs[c].push(ds.ch[c].v[t + h]);
                pr[c][0].push(pt.mlp[c]);
                pr[c][1].push(pt.ridge[c]);
                pr[c][2].push(pt.clim[c]);
                pr[c][3].push(ds.ch[c].v[t]);
            }
            if let (Some(rc), Some(p)) = (rain_c, pt.p_rain) {
                yr.push(ds.ch[rc].v[t + h] > 0.0);
                p_m.push(p);
                let m = ds.dates[t + h].m as usize;
                p_c.push(wet[m] / tot[m].max(1.0));
                p_p.push(if ds.ch[rc].v[t] > 0.0 { 1.0 } else { 0.0 });
            }
            if let (Some(l), Some(pb)) = (&ds.labels, &pt.cond_probs) {
                yc.push(l[t + h]);
                c_m.push(argmax(pb));
                c_p.push(l[t]);
            }
            tmp.clear();
        }
        for c in 0..nch {
            let mae_v: Vec<f64> = (0..4).map(|k| mae(&obs[c], &pr[c][k])).collect();
            let rm: Vec<f64> = (0..4).map(|k| rmse(&obs[c], &pr[c][k])).collect();
            let sk = skill(mse(&obs[c], &pr[c][0]), mse(&obs[c], &pr[c][2]));
            let _ = writeln!(md, "| {} | {}d | {} | {} | {} | {} | {} | {} | {} | {:+.3} |", ds.ch[c].name, h, f3(mae_v[0]), f3(mae_v[1]), f3(mae_v[2]), f3(mae_v[3]), f3(rm[0]), f3(rm[2]), f3(rm[3]), sk);
            if h == 1 { res.mae_h1[c] = [mae_v[0], mae_v[1], mae_v[2], mae_v[3]]; }
            if h == cfg.horizons { res.mae_hmax[c] = [mae_v[0], mae_v[1], mae_v[2], mae_v[3]]; }
        }
        if !yr.is_empty() {
            let b = [brier(&p_m, &yr), brier(&p_c, &yr), brier(&p_p, &yr)];
            rain_rows.push(format!("| {}d | {} | {} | {} | {} | {} | {} |", h, f3(b[0]), f3(b[1]), f3(b[2]), f3(auc(&p_m, &yr)), f3(auc(&p_c, &yr)), f3(log_loss(&p_m, &yr))));
            if h == 1 { res.brier_h1 = Some(b); }
        }
        if !yc.is_empty() {
            let k = ds.label_names.len();
            let cf = confusion(k, &yc, &c_m);
            let cp = confusion(k, &yc, &c_p);
            let maj = vec![majority.unwrap(); yc.len()];
            cond_rows.push(format!("| {}d | {} | {} | {} | {} | {} |", h, f3(accuracy(&yc, &c_m)), f3(accuracy(&yc, &c_p)), f3(accuracy(&yc, &maj)), f3(macro_f1(&cf)), f3(macro_f1(&cp))));
        }
        if h == 1 {
            if let Some(tc) = tc {
                h1_series = Some((obs[tc].clone(), pr[tc][0].clone(), pr[tc][2].clone()));
            }
        }
    }
    let _ = writeln!(md, "\n*Precipitation point forecasts are median-type (log1p space) values; judge precipitation mainly by the probabilistic table below.*\n");
    if !rain_rows.is_empty() {
        let _ = writeln!(md, "#### Rain / no-rain probability (precip > 0)\n");
        let _ = writeln!(md, "| Lead | Brier AI (blend) | Brier monthly clim. | Brier persistence | AUC AI (blend) | AUC clim. | Log-loss AI (blend) |");
        let _ = writeln!(md, "|---|---|---|---|---|---|---|");
        for r in &rain_rows { let _ = writeln!(md, "{r}"); }
        let _ = writeln!(md);
    }
    if !cond_rows.is_empty() {
        let _ = writeln!(md, "#### Next-condition forecast ({})\n", ds.label_names.join("/"));
        let _ = writeln!(md, "| Lead | Acc AI-MLP | Acc persistence | Acc majority | Macro-F1 AI-MLP | Macro-F1 persistence |");
        let _ = writeln!(md, "|---|---|---|---|---|---|");
        for r in &cond_rows { let _ = writeln!(md, "{r}"); }
        let _ = writeln!(md);
    }

    // ---------- condition detection (same-day) ----------
    if let Some(det) = ConditionDetector::train(ds, &ctx, train_last, cfg) {
        let lab = ds.labels.as_ref().unwrap();
        let k = ds.label_names.len();
        let y: Vec<usize> = test.iter().map(|&t| lab[t]).collect();
        let p: Vec<usize> = test.iter().map(|&t| argmax(&det.probs(ds, &ctx, t))).collect();
        let maj = vec![majority.unwrap(); y.len()];
        let (ri, si) = (ds.label_names.iter().position(|s| s == "rain"), ds.label_names.iter().position(|s| s == "sun"));
        let rc = rain_c.unwrap();
        let rule: Vec<usize> = test.iter().map(|&t| if ds.ch[rc].v[t] > 0.0 { ri.unwrap_or(0) } else { si.unwrap_or(0) }).collect();
        let cf = confusion(k, &y, &p);
        let acc = (accuracy(&y, &p), accuracy(&y, &maj), accuracy(&y, &rule));
        res.detect_acc = Some(acc);
        let _ = writeln!(md, "#### Current-condition detection from same-day readings\n");
        let _ = writeln!(md, "Accuracy: AI-MLP **{}**, majority-class {}, rule (precip>0 → rain else sun) {} · Macro-F1: AI-MLP {}, rule {}\n", f3(acc.0), f3(acc.1), f3(acc.2), f3(macro_f1(&cf)), f3(macro_f1(&confusion(k, &y, &rule))));
        let _ = writeln!(md, "| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|");
        for (c, (pp, rr, ff, s)) in prf(&cf).into_iter().enumerate() {
            let _ = writeln!(md, "| {} | {} | {} | {} | {} |", ds.label_names[c], f3(pp), f3(rr), f3(ff), s);
        }
        let _ = writeln!(md, "\nConfusion matrix (rows = truth, cols = prediction; order {}):\n", ds.label_names.join(", "));
        let _ = writeln!(md, "```");
        for row in &cf { let _ = writeln!(md, "{}", row.iter().map(|v| format!("{:4}", v)).collect::<Vec<_>>().join(" ")); }
        let _ = writeln!(md, "```\n");
    }

    // ---------- fault / anomaly detection ----------
    if let (Some((obs, pm, pc)), Some(tc)) = (h1_series, tc) {
        let ts: Vec<usize> = (MIN_T..train_last).collect();
        let vs = &ts[ts.len() - ts.len() / 5..];
        let resid: Vec<f64> = vs.iter().map(|&t| ds.ch[tc].v[t + 1] - fc.predict(ds, &ctx, t, 1).mlp[tc]).collect();
        let sig_m = std_dev(&resid).max(1e-6);
        let sig_c = (0..=train_last).map(|i| (ds.ch[tc].v[i] - clim.value(tc, ds.dates[i])).powi(2)).sum::<f64>().div_euclid(1.0);
        let sig_c = (sig_c / (train_last + 1) as f64).sqrt();
        let fr = fault_detection(&obs, &pm, &pc, sig_m, sig_c, &[2.0, 3.0, 4.0, 6.0], 30, 99);
        let _ = writeln!(md, "#### Sensor-fault / anomaly detection on `{}` (injected ±m·σ faults on 5% of test days, threshold z>3, mean of 30 draws)\n", ds.ch[tc].name);
        let _ = writeln!(md, "σ(model residual)={:.2}, σ(climatology anomaly)={:.2}\n", sig_m, sig_c);
        let _ = writeln!(md, "| Fault size | AUC forecast-residual | AUC climatology z | Precision / Recall (forecast) | Precision / Recall (clim.) |\n|---|---|---|---|---|");
        for r in &fr {
            let _ = writeln!(md, "| {}σ | {} | {} | {} / {} | {} / {} |", r.magnitude, f3(r.auc_model), f3(r.auc_clim), f3(r.prec_model), f3(r.rec_model), f3(r.prec_clim), f3(r.rec_clim));
            if r.magnitude == 4.0 { res.fault_auc_4sigma = Some((r.auc_model, r.auc_clim)); }
        }
        let _ = writeln!(md);
    }
    res
}
