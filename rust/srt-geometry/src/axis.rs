//! Steering in read-out space.
//!
//! A behavioural direction can be applied to the residual stream, which needs
//! model hooks and a full re-encode per setting, or to the projected query,
//! which is one vector addition. This is the second.
//!
//! An axis is a difference of class means over already-projected vectors, so
//! building one costs a single offline encode of two small text sets and the
//! artifact is `proj_dim` floats: about 2KB at fp16 for a 1024-dim head. At
//! query time the model is not involved at all.
//!
//! The measurement discipline carries over intact. An axis only means
//! something if a random axis of the same norm does not reproduce its effect,
//! so [`Axis::random_like`] exists to make that control cheap to run.

use crate::normalize;

/// Calibrated steering strength: strong effect, query still in charge.
///
/// Measured rather than chosen. See [`Axis::apply`] for the dose-response.
pub const DEFAULT_ALPHA: f32 = 0.5;

#[derive(Clone)]
pub struct Axis {
    v: Vec<f32>,
    name: String,
}

impl Axis {
    /// Difference of means between two sets of projected vectors.
    ///
    /// Both sides must already be head-projected. Mixing raw and projected
    /// vectors here silently produces a direction in the wrong space.
    pub fn from_contrast(name: &str, positive: &[Vec<f32>], negative: &[Vec<f32>]) -> Self {
        let a = crate::mean(positive);
        let b = crate::mean(negative);
        let mut v: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        normalize(&mut v);
        Axis { v, name: name.to_string() }
    }

    pub fn from_vec(name: &str, mut v: Vec<f32>) -> Self {
        normalize(&mut v);
        Axis { v, name: name.to_string() }
    }

    /// A matched-norm random direction: the control every steering claim needs.
    ///
    /// Seeded xorshift rather than a dependency, since this runs in WASM.
    pub fn random_like(&self, seed: u64) -> Self {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) - 0.5
        };
        let mut v: Vec<f32> = (0..self.v.len()).map(|_| next()).collect();
        normalize(&mut v);
        Axis { v, name: format!("random({})", self.name) }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.v
    }

    /// `normalize(q + alpha * axis)`.
    ///
    /// Alpha is in units of the normalized axis against a normalized query, so
    /// it is comparable across heads. **Calibrate it against retention, not
    /// against how well the steering appears to work.** Class purity rises
    /// monotonically with alpha all the way to 0.9, but past alpha ~1 that is
    /// the axis replacing the query rather than steering it, and every query
    /// starts returning the same results.
    ///
    /// Measured on real image retrieval, three text-defined contrasts, 32
    /// matched-norm random controls per point:
    ///
    /// | alpha | class lift | neighbourhood retained |
    /// |---|---|---|
    /// | 0.25 | ~2x | 0.93 |
    /// | 0.50 | 10-30x | 0.61-0.77 |
    /// | 0.75 | 20-40x | 0.31-0.51 |
    /// | 1.00 | 30-160x | 0.13-0.29 |
    /// | 2.00 | 37-220x | 0.01-0.03 (query erased) |
    ///
    /// [`DEFAULT_ALPHA`] is the calibrated setting. Use
    /// [`Index::retention`](crate::Index::retention) to re-derive it for a new
    /// head or gallery rather than inheriting it.
    pub fn apply(&self, query: &[f32], alpha: f32) -> Vec<f32> {
        let mut out: Vec<f32> = query
            .iter()
            .zip(&self.v)
            .map(|(q, a)| q + alpha * a)
            .collect();
        normalize(&mut out);
        out
    }
}
