//! Daily weather datasets: CSV loading, calendar regularisation, gap filling.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    pub y: i32,
    pub m: u32,
    pub d: u32,
}

impl Date {
    pub fn new(y: i32, m: u32, d: u32) -> Self {
        Date { y, m, d }
    }
    pub fn parse(s: &str) -> Option<Date> {
        let s = s.trim().trim_matches('"');
        let mut it = s.split('-');
        let y = it.next()?.trim().parse().ok()?;
        let m = it.next()?.trim().parse().ok()?;
        let d = it.next()?.trim().parse().ok()?;
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
            return None;
        }
        Some(Date { y, m, d })
    }
    /// Days since 1970-01-01 (proleptic Gregorian).
    pub fn ordinal(&self) -> i64 {
        let y = self.y as i64 - if self.m <= 2 { 1 } else { 0 };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = if self.m > 2 { self.m as i64 - 3 } else { self.m as i64 + 9 };
        let doy = (153 * mp + 2) / 5 + self.d as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }
    pub fn from_ordinal(n: i64) -> Date {
        let z = n + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        Date { y: (y + if m <= 2 { 1 } else { 0 }) as i32, m, d }
    }
    pub fn plus(&self, n: i64) -> Date {
        Date::from_ordinal(self.ordinal() + n)
    }
    /// Day of year, 1-based.
    pub fn doy(&self) -> u32 {
        (self.ordinal() - Date::new(self.y, 1, 1).ordinal() + 1) as u32
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.y, self.m, self.d)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Temp,
    Wind,
    Precip,
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub name: String,
    pub kind: Kind,
    pub v: Vec<f64>,
}

/// Role of an input-only series (never a forecast target).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuxKind {
    /// Temperature at a neighbouring station (upstream-advection signal).
    Temp,
    /// Sea-level pressure (own or neighbouring station): level and 1-day tendency.
    Pressure,
}

/// Input-only series, e.g. neighbouring stations' temperature / pressure.
#[derive(Clone, Debug)]
pub struct Aux {
    pub name: String,
    pub kind: AuxKind,
    pub v: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    pub dates: Vec<Date>,
    pub ch: Vec<Channel>,
    /// Optional categorical weather condition per day (index into `label_names`).
    pub labels: Option<Vec<usize>>,
    pub label_names: Vec<String>,
    /// Number of calendar days that were missing in the source file and were interpolated.
    pub n_filled: usize,
    /// Optional input-only series (neighbour stations, pressure). Empty for single-station data.
    pub aux: Vec<Aux>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.dates.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }
    pub fn channel_index(&self, name: &str) -> Option<usize> {
        self.ch.iter().position(|c| c.name == name)
    }
    pub fn precip_channel(&self) -> Option<usize> {
        self.ch.iter().position(|c| c.kind == Kind::Precip)
    }
    pub fn primary_temp_channel(&self) -> Option<usize> {
        self.ch.iter().position(|c| c.kind == Kind::Temp)
    }
    /// Last index whose date is <= `d`.
    pub fn last_index_on_or_before(&self, d: Date) -> Option<usize> {
        (0..self.len()).rev().find(|&i| self.dates[i] <= d)
    }
}

fn parse_value(s: &str) -> f64 {
    // Some public files contain stray markers such as "?0.2"; strip them rather than drop the day.
    let t = s.trim().trim_matches('"').trim_start_matches(|c| c == '?' || c == '*');
    t.parse::<f64>().unwrap_or(f64::NAN)
}

pub(crate) fn split_csv_line(line: &str) -> Vec<String> {
    line.trim_end_matches(&['\r', '\n'][..])
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect()
}

struct RawRow {
    ord: i64,
    vals: Vec<f64>,
    label: Option<String>,
}

/// Put rows on a gap-free daily calendar, linearly interpolating any missing day / NaN value.
fn regularize(
    name: &str,
    mut rows: Vec<RawRow>,
    chans: Vec<(&str, Kind)>,
    label_names: Vec<String>,
) -> Result<Dataset, String> {
    if rows.is_empty() {
        return Err("no rows parsed".into());
    }
    rows.sort_by_key(|r| r.ord);
    rows.dedup_by_key(|r| r.ord);
    let first = rows[0].ord;
    let last = rows.last().unwrap().ord;
    let n = (last - first + 1) as usize;
    let nch = chans.len();
    let mut vals = vec![vec![f64::NAN; n]; nch];
    let mut lab: Vec<Option<usize>> = vec![None; n];
    let mut present = vec![false; n];
    for r in &rows {
        let i = (r.ord - first) as usize;
        present[i] = true;
        for c in 0..nch {
            vals[c][i] = r.vals[c];
        }
        if let Some(l) = &r.label {
            lab[i] = label_names.iter().position(|x| x == l);
        }
    }
    let mut n_filled = 0usize;
    for c in 0..nch {
        let v = &mut vals[c];
        let mut i = 0;
        while i < n {
            if v[i].is_nan() {
                let start = i;
                while i < n && v[i].is_nan() {
                    i += 1;
                }
                let before = if start > 0 { Some(v[start - 1]) } else { None };
                let after = if i < n { Some(v[i]) } else { None };
                for k in start..i {
                    v[k] = match (before, after) {
                        (Some(b), Some(a)) => b + (a - b) * (k - start + 1) as f64 / (i - start + 1) as f64,
                        (Some(b), None) => b,
                        (None, Some(a)) => a,
                        (None, None) => 0.0,
                    };
                }
            } else {
                i += 1;
            }
        }
    }
    for i in 0..n {
        if !present[i] {
            n_filled += 1;
        }
    }
    let labels = if label_names.is_empty() {
        None
    } else {
        let mut out = vec![0usize; n];
        let mut prev = lab.iter().flatten().next().copied().unwrap_or(0);
        for i in 0..n {
            if let Some(l) = lab[i] {
                prev = l;
            }
            out[i] = prev;
        }
        Some(out)
    };
    Ok(Dataset {
        name: name.to_string(),
        dates: (0..n).map(|i| Date::from_ordinal(first + i as i64)).collect(),
        ch: chans
            .into_iter()
            .enumerate()
            .map(|(i, (nm, k))| Channel { name: nm.to_string(), kind: k, v: vals[i].clone() })
            .collect(),
        labels,
        label_names,
        n_filled,
        aux: Vec::new(),
    })
}

/// Labelled daily weather in the vega-datasets layout. Accepts both
/// `date,precipitation,temp_max,temp_min,wind,weather` and the multi-station
/// `location,date,...` form (then `location` selects the station).
pub fn parse_labeled(text: &str, location: Option<&str>, name: &str) -> Result<Dataset, String> {
    let mut rows = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut off = 0usize;
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        if ln == 0 {
            off = if f[0].eq_ignore_ascii_case("location") { 1 } else { 0 };
            continue;
        }
        if f.len() < 6 + off {
            return Err(format!("line {}: expected {} fields, got {}", ln + 1, 6 + off, f.len()));
        }
        if let (1, Some(loc)) = (off, location) {
            if f[0] != loc {
                continue;
            }
        }
        let date = Date::parse(&f[off]).ok_or_else(|| format!("line {}: bad date '{}'", ln + 1, f[off]))?;
        let lab = f[off + 5].clone();
        if !names.contains(&lab) {
            names.push(lab.clone());
        }
        rows.push(RawRow {
            ord: date.ordinal(),
            vals: vec![parse_value(&f[off + 2]), parse_value(&f[off + 3]), parse_value(&f[off + 4]), parse_value(&f[off + 1])],
            label: Some(lab),
        });
    }
    names.sort();
    regularize(
        name,
        rows,
        vec![("tmax_c", Kind::Temp), ("tmin_c", Kind::Temp), ("wind_ms", Kind::Wind), ("precip_mm", Kind::Precip)],
        names,
    )
}

/// Seattle-Tacoma daily weather 2012-2015.
pub fn parse_seattle(text: &str) -> Result<Dataset, String> {
    parse_labeled(text, None, "Seattle-Tacoma daily 2012-2015")
}

/// Single-variable daily series: `Date,<value>` (e.g. Melbourne daily minimum temperature 1981-1990).
pub fn parse_single(text: &str, name: &str, chan: &str, kind: Kind) -> Result<Dataset, String> {
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if ln == 0 || line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        if f.len() < 2 {
            return Err(format!("line {}: expected 2 fields", ln + 1));
        }
        let date = Date::parse(&f[0]).ok_or_else(|| format!("line {}: bad date '{}'", ln + 1, f[0]))?;
        rows.push(RawRow { ord: date.ordinal(), vals: vec![parse_value(&f[1])], label: None });
    }
    regularize(name, rows, vec![(chan, kind)], vec![])
}

pub fn load_seattle(path: &str) -> Result<Dataset, String> {
    parse_seattle(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
}

pub fn load_melbourne(path: &str) -> Result<Dataset, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse_single(&t, "Melbourne daily min temperature 1981-1990", "tmin_c", Kind::Temp)
}

/// New York daily weather 2012-2015 from the multi-station vega-datasets `weather.csv`.
pub fn load_new_york(path: &str) -> Result<Dataset, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse_labeled(&t, Some("New York"), "New York daily 2012-2015")
}

/// Wide multi-station ECA&D table (florian-huber/weather_prediction_dataset): `DATE` is YYYYMMDD and
/// columns are `<STATION>_<variable>`. Targets: temp_max, temp_min, wind_speed and precipitation
/// (the file stores precipitation in units of 10 mm; converted here to mm).
/// With `neighbors = true` every *other* station's mean temperature and pressure, plus the station's
/// own pressure, are attached as input-only series (`Dataset::aux`).
pub fn parse_eca_wide(text: &str, station: &str, name: &str, neighbors: bool) -> Result<Dataset, String> {
    let mut lines = text.lines();
    let header = split_csv_line(lines.next().ok_or("empty file")?);
    let pos = |want: &str| header.iter().position(|h| h == want);
    let find = |v: &str| -> Result<usize, String> {
        let want = format!("{station}_{v}");
        pos(&want).ok_or_else(|| format!("column {want} not found"))
    };
    let (ci_date, c_tx, c_tn, c_w, c_p) = (
        pos("DATE").ok_or("no DATE column")?,
        find("temp_max")?,
        find("temp_min")?,
        find("wind_speed")?,
        find("precipitation")?,
    );
    // input-only columns: (name, kind, column index)
    let mut aux_cols: Vec<(String, AuxKind, usize)> = Vec::new();
    if neighbors {
        if let Some(ci) = pos(&format!("{station}_pressure")) {
            aux_cols.push((format!("{station}_pressure(own)"), AuxKind::Pressure, ci));
        }
        for (ci, h) in header.iter().enumerate() {
            if let Some(o) = h.strip_suffix("_temp_mean") {
                if o != station {
                    aux_cols.push((format!("{o}_temp_mean"), AuxKind::Temp, ci));
                    if let Some(cp) = pos(&format!("{o}_pressure")) {
                        aux_cols.push((format!("{o}_pressure"), AuxKind::Pressure, cp));
                    }
                }
            }
        }
    }
    let mut rows = Vec::new();
    let mut aux_rows: Vec<(i64, Vec<f64>)> = Vec::new();
    for (ln, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        let ds = &f[ci_date];
        if ds.len() != 8 {
            return Err(format!("line {}: bad DATE '{}'", ln + 2, ds));
        }
        let date = Date::parse(&format!("{}-{}-{}", &ds[0..4], &ds[4..6], &ds[6..8])).ok_or_else(|| format!("line {}: bad DATE '{}'", ln + 2, ds))?;
        rows.push(RawRow {
            ord: date.ordinal(),
            vals: vec![parse_value(&f[c_tx]), parse_value(&f[c_tn]), parse_value(&f[c_w]), parse_value(&f[c_p]) * 10.0],
            label: None,
        });
        if !aux_cols.is_empty() {
            aux_rows.push((date.ordinal(), aux_cols.iter().map(|(_, _, ci)| parse_value(&f[*ci])).collect()));
        }
    }
    let mut ds = regularize(
        name,
        rows,
        vec![("tmax_c", Kind::Temp), ("tmin_c", Kind::Temp), ("wind_ms", Kind::Wind), ("precip_mm", Kind::Precip)],
        vec![],
    )?;
    if !aux_cols.is_empty() {
        let map: std::collections::HashMap<i64, &Vec<f64>> = aux_rows.iter().map(|(o, v)| (*o, v)).collect();
        for (k, (nm, kind, _)) in aux_cols.iter().enumerate() {
            let mut v: Vec<f64> = ds.dates.iter().map(|d| map.get(&d.ordinal()).map(|r| r[k]).unwrap_or(f64::NAN)).collect();
            // carry the last valid value forward over any gaps (none in the bundled file)
            let first_ok = v.iter().cloned().find(|x| !x.is_nan()).unwrap_or(0.0);
            let mut prev = first_ok;
            for x in v.iter_mut() {
                if x.is_nan() { *x = prev; } else { prev = *x; }
            }
            ds.aux.push(Aux { name: nm.clone(), kind: *kind, v });
        }
    }
    Ok(ds)
}

pub fn load_eca_station(path: &str, station: &str, neighbors: bool) -> Result<Dataset, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse_eca_wide(&t, station, &format!("ECA&D {station} daily 2000-2010{}", if neighbors { " (+neighbours)" } else { "" }), neighbors)
}
