//! Small fully-connected network (tanh hidden units) with manual backprop, AdamW and
//! early stopping, plus a seed ensemble. Supports MSE regression and softmax classification.

use crate::linalg::Mat;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    Mse,
    SoftmaxCe,
}

#[derive(Clone, Debug)]
pub struct MlpConfig {
    pub hidden: Vec<usize>,
    pub lr: f64,
    pub epochs: usize,
    pub batch: usize,
    pub weight_decay: f64,
    pub patience: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        MlpConfig { hidden: vec![32, 16], lr: 2e-3, epochs: 200, batch: 32, weight_decay: 1e-2, patience: 20 }
    }
}

#[derive(Clone, Debug)]
struct Layer {
    n_in: usize,
    n_out: usize,
    w: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Layer>,
    pub loss: Loss,
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

impl Mlp {
    pub fn new(n_in: usize, n_out: usize, hidden: &[usize], loss: Loss, rng: &mut Rng) -> Self {
        let mut sizes = vec![n_in];
        sizes.extend_from_slice(hidden);
        sizes.push(n_out);
        let mut layers = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (ni, no) = (sizes[l], sizes[l + 1]);
            let scale = (1.0 / ni as f64).sqrt();
            layers.push(Layer { n_in: ni, n_out: no, w: (0..ni * no).map(|_| rng.normal() * scale).collect(), b: vec![0.0; no] });
        }
        Mlp { layers, loss }
    }

    /// acts[0] = input, acts[l+1] = tanh output of layer l (the last entry holds raw outputs/logits).
    fn forward(&self, x: &[f64], acts: &mut Vec<Vec<f64>>) {
        acts.clear();
        acts.push(x.to_vec());
        let last = self.layers.len() - 1;
        for (l, layer) in self.layers.iter().enumerate() {
            let inp = &acts[l];
            let mut out = vec![0.0; layer.n_out];
            for o in 0..layer.n_out {
                let row = &layer.w[o * layer.n_in..(o + 1) * layer.n_in];
                let mut s = layer.b[o];
                for i in 0..layer.n_in {
                    s += row[i] * inp[i];
                }
                out[o] = if l == last { s } else { s.tanh() };
            }
            acts.push(out);
        }
    }

    /// Regression: predicted outputs. Classification: class probabilities.
    pub fn predict_row(&self, x: &[f64]) -> Vec<f64> {
        let mut acts = Vec::new();
        self.forward(x, &mut acts);
        let out = acts.pop().unwrap();
        if self.loss == Loss::SoftmaxCe { softmax(&out) } else { out }
    }

    fn sample_loss(&self, raw: &[f64], y: &[f64]) -> f64 {
        match self.loss {
            Loss::Mse => raw.iter().zip(y).map(|(o, t)| (o - t).powi(2)).sum::<f64>() / raw.len() as f64,
            Loss::SoftmaxCe => {
                let p = softmax(raw);
                -y.iter().zip(&p).map(|(t, pi)| t * pi.max(1e-12).ln()).sum::<f64>()
            }
        }
    }

    pub fn dataset_loss(&self, x: &Mat, y: &Mat) -> f64 {
        if x.r == 0 {
            return 0.0;
        }
        let mut acts = Vec::new();
        let mut tot = 0.0;
        for i in 0..x.r {
            self.forward(x.row(i), &mut acts);
            tot += self.sample_loss(acts.last().unwrap(), y.row(i));
        }
        tot / x.r as f64
    }

    /// Train with minibatch AdamW; keeps the weights with the best validation loss.
    pub fn fit(x: &Mat, y: &Mat, xv: &Mat, yv: &Mat, loss: Loss, cfg: &MlpConfig, seed: u64) -> Mlp {
        let mut rng = Rng::new(seed);
        let mut net = Mlp::new(x.c, y.c, &cfg.hidden, loss, &mut rng);
        let nl = net.layers.len();
        let mut mw: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut vw = mw.clone();
        let mut mb: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        let mut vb = mb.clone();
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        let mut step = 0i32;
        let mut order: Vec<usize> = (0..x.r).collect();
        let has_val = xv.r > 0;
        let mut best = net.clone();
        let mut best_loss = f64::INFINITY;
        let mut since = 0usize;
        let mut acts: Vec<Vec<f64>> = Vec::new();
        for _epoch in 0..cfg.epochs {
            rng.shuffle(&mut order);
            for chunk in order.chunks(cfg.batch) {
                let mut gw: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
                let mut gb: Vec<Vec<f64>> = net.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
                for &i in chunk {
                    net.forward(x.row(i), &mut acts);
                    let raw = acts.last().unwrap().clone();
                    let t = y.row(i);
                    let mut delta: Vec<f64> = match net.loss {
                        Loss::Mse => raw.iter().zip(t).map(|(o, tt)| 2.0 * (o - tt) / raw.len() as f64).collect(),
                        Loss::SoftmaxCe => softmax(&raw).iter().zip(t).map(|(p, tt)| p - tt).collect(),
                    };
                    for l in (0..nl).rev() {
                        let layer = &net.layers[l];
                        let inp = &acts[l];
                        for o in 0..layer.n_out {
                            gb[l][o] += delta[o];
                            for k in 0..layer.n_in {
                                gw[l][o * layer.n_in + k] += delta[o] * inp[k];
                            }
                        }
                        if l > 0 {
                            let mut prev = vec![0.0; layer.n_in];
                            for o in 0..layer.n_out {
                                for k in 0..layer.n_in {
                                    prev[k] += layer.w[o * layer.n_in + k] * delta[o];
                                }
                            }
                            for k in 0..layer.n_in {
                                prev[k] *= 1.0 - inp[k] * inp[k];
                            }
                            delta = prev;
                        }
                    }
                }
                step += 1;
                let bs = chunk.len() as f64;
                let c1 = 1.0 - b1.powi(step);
                let c2 = 1.0 - b2.powi(step);
                for l in 0..nl {
                    let layer = &mut net.layers[l];
                    for k in 0..layer.w.len() {
                        let g = gw[l][k] / bs;
                        mw[l][k] = b1 * mw[l][k] + (1.0 - b1) * g;
                        vw[l][k] = b2 * vw[l][k] + (1.0 - b2) * g * g;
                        layer.w[k] -= cfg.lr * ((mw[l][k] / c1) / ((vw[l][k] / c2).sqrt() + eps) + cfg.weight_decay * layer.w[k]);
                    }
                    for k in 0..layer.b.len() {
                        let g = gb[l][k] / bs;
                        mb[l][k] = b1 * mb[l][k] + (1.0 - b1) * g;
                        vb[l][k] = b2 * vb[l][k] + (1.0 - b2) * g * g;
                        layer.b[k] -= cfg.lr * (mb[l][k] / c1) / ((vb[l][k] / c2).sqrt() + eps);
                    }
                }
            }
            if has_val {
                let vl = net.dataset_loss(xv, yv);
                if vl < best_loss - 1e-6 {
                    best_loss = vl;
                    best = net.clone();
                    since = 0;
                } else {
                    since += 1;
                    if since >= cfg.patience {
                        break;
                    }
                }
            }
        }
        if has_val { best } else { net }
    }
}

/// Seed ensemble: members differ in initialisation and minibatch order; outputs are averaged.
#[derive(Clone, Debug)]
pub struct Ensemble {
    pub members: Vec<Mlp>,
}

impl Ensemble {
    pub fn fit(x: &Mat, y: &Mat, xv: &Mat, yv: &Mat, loss: Loss, cfg: &MlpConfig, n: usize, seed: u64) -> Self {
        let members = (0..n.max(1)).map(|k| Mlp::fit(x, y, xv, yv, loss, cfg, seed.wrapping_mul(7919).wrapping_add(k as u64 * 104_729 + 1))).collect();
        Ensemble { members }
    }
    pub fn predict_row(&self, x: &[f64]) -> Vec<f64> {
        let mut acc: Vec<f64> = Vec::new();
        for m in &self.members {
            let p = m.predict_row(x);
            if acc.is_empty() {
                acc = p;
            } else {
                for (a, b) in acc.iter_mut().zip(p) {
                    *a += b;
                }
            }
        }
        let n = self.members.len() as f64;
        acc.iter().map(|v| v / n).collect()
    }
}
