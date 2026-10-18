//! Minimal dense linear algebra: row-major matrix, ridge regression, standardiser.

#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f64>,
}

impl Mat {
    pub fn zeros(r: usize, c: usize) -> Self {
        Mat { r, c, d: vec![0.0; r * c] }
    }
    pub fn with_cols(c: usize) -> Self {
        Mat { r: 0, c, d: Vec::new() }
    }
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        if rows.is_empty() {
            return Mat::default();
        }
        let c = rows[0].len();
        let mut d = Vec::with_capacity(rows.len() * c);
        for r in rows {
            assert_eq!(r.len(), c, "ragged rows");
            d.extend_from_slice(r);
        }
        Mat { r: rows.len(), c, d }
    }
    pub fn push(&mut self, row: &[f64]) {
        if self.c == 0 && self.r == 0 {
            self.c = row.len();
        }
        assert_eq!(row.len(), self.c);
        self.d.extend_from_slice(row);
        self.r += 1;
    }
    pub fn row(&self, i: usize) -> &[f64] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
    pub fn select_rows(&self, idx: &[usize]) -> Mat {
        let mut m = Mat::with_cols(self.c);
        for &i in idx {
            m.push(self.row(i));
        }
        m
    }
}

/// Solve A X = B in place (A is n x n, B is n x k, both row-major) by Gaussian elimination
/// with partial pivoting. Returns None if (numerically) singular.
pub fn solve(a: &mut [f64], n: usize, b: &mut [f64], k: usize) -> Option<()> {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            for j in 0..k {
                b.swap(col * k + j, piv * k + j);
            }
        }
        let p = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / p;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= f * a[col * n + j];
            }
            for j in 0..k {
                b[r * k + j] -= f * b[col * k + j];
            }
        }
    }
    for col in (0..n).rev() {
        for j in 0..k {
            let mut s = b[col * k + j];
            for c2 in col + 1..n {
                s -= a[col * n + c2] * b[c2 * k + j];
            }
            b[col * k + j] = s / a[col * n + col];
        }
    }
    Some(())
}

/// Ridge regression with an unpenalised intercept. Returns weights of shape (c+1) x k
/// (row 0 is the intercept).
pub fn ridge_fit(x: &Mat, y: &Mat, lambda: f64) -> Mat {
    assert_eq!(x.r, y.r);
    let p = x.c + 1;
    let k = y.c;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p * k];
    let mut xi = vec![0.0; p];
    for i in 0..x.r {
        xi[0] = 1.0;
        xi[1..].copy_from_slice(x.row(i));
        for a in 0..p {
            for b in 0..p {
                ata[a * p + b] += xi[a] * xi[b];
            }
            for j in 0..k {
                atb[a * k + j] += xi[a] * y.at(i, j);
            }
        }
    }
    for a in 1..p {
        ata[a * p + a] += lambda;
    }
    ata[0] += 1e-9;
    solve(&mut ata, p, &mut atb, k).expect("ridge normal equations singular");
    Mat { r: p, c: k, d: atb }
}

pub fn ridge_predict(w: &Mat, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; w.c];
    for j in 0..w.c {
        let mut s = w.at(0, j);
        for (i, xv) in x.iter().enumerate() {
            s += xv * w.at(i + 1, j);
        }
        out[j] = s;
    }
    out
}

/// Per-column standardiser fitted on training data only.
#[derive(Clone, Debug)]
pub struct Standardizer {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Standardizer {
    pub fn fit(x: &Mat) -> Self {
        let mut mean = vec![0.0; x.c];
        let mut std = vec![1.0; x.c];
        if x.r == 0 {
            return Standardizer { mean, std };
        }
        for j in 0..x.c {
            let m = (0..x.r).map(|i| x.at(i, j)).sum::<f64>() / x.r as f64;
            let v = (0..x.r).map(|i| (x.at(i, j) - m).powi(2)).sum::<f64>() / x.r as f64;
            mean[j] = m;
            std[j] = if v.sqrt() < 1e-8 { 1.0 } else { v.sqrt() };
        }
        Standardizer { mean, std }
    }
    pub fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, v)| (v - self.mean[j]) / self.std[j]).collect()
    }
    pub fn transform(&self, x: &Mat) -> Mat {
        let mut m = Mat::with_cols(x.c);
        for i in 0..x.r {
            m.push(&self.transform_row(x.row(i)));
        }
        m
    }
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}
pub fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}
