//! Weather-condition *detection*: classify today's condition (sun / fog / drizzle / rain / snow)
//! from today's instrument readings (+ season), with no future information.

use crate::features::{detection_features, Context};
use crate::data::Dataset;
use crate::linalg::{Mat, Standardizer};
use crate::mlp::{Ensemble, Loss};
use crate::models::TrainCfg;

#[derive(Clone, Debug)]
pub struct ConditionDetector {
    pub xs: Standardizer,
    pub ens: Ensemble,
    pub names: Vec<String>,
}

impl ConditionDetector {
    /// Train on days `0..=train_last`; the most recent 20% (chronological) is used for early stopping.
    pub fn train(ds: &Dataset, ctx: &Context, train_last: usize, cfg: &TrainCfg) -> Option<Self> {
        let lab = ds.labels.as_ref()?;
        let n = train_last + 1;
        let split = n - n / 5;
        let k = ds.label_names.len();
        let build = |r: std::ops::Range<usize>| {
            let (mut x, mut y) = (Mat::default(), Mat::default());
            for t in r {
                x.push(&detection_features(ds, ctx, t));
                let mut oh = vec![0.0; k];
                oh[lab[t]] = 1.0;
                y.push(&oh);
            }
            (x, y)
        };
        let (x_tr, y_tr) = build(0..split);
        let (x_va, y_va) = build(split..n);
        let xs = Standardizer::fit(&x_tr);
        let ens = Ensemble::fit(&xs.transform(&x_tr), &y_tr, &xs.transform(&x_va), &y_va, Loss::SoftmaxCe, &cfg.mlp, cfg.ensemble, cfg.seed + 777);
        Some(ConditionDetector { xs, ens, names: ds.label_names.clone() })
    }

    pub fn probs(&self, ds: &Dataset, ctx: &Context, t: usize) -> Vec<f64> {
        self.ens.predict_row(&self.xs.transform_row(&detection_features(ds, ctx, t)))
    }
}

pub fn argmax(v: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}
