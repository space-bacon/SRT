//! Read-out geometry for SRT-instrumented frozen models.
//!
//! This crate is the arithmetic between a backbone's hidden states and a
//! usable answer. It deliberately knows nothing about how the states were
//! produced, so the same code serves a browser running a 0.6B model under
//! WASM, a laptop running a 31B model under Metal, and a datacenter GPU.
//! That is the program's central claim expressed as a dependency graph:
//! the substrate varies, this does not.
//!
//! Four operations, in the order a deployment performs them:
//!
//! 1. [`Head`] projects a raw hidden state into the shared read-out space.
//!    Per-modality centering is mandatory, not optional: raw cosine on these
//!    states is dominated by an anisotropy attractor and is not interpretable.
//! 2. [`Head::recalibrate`] swaps in means measured on the local runtime. This
//!    is the 42KB fix. Skipping it cost 24 i2t R@1 points on one measured
//!    runtime pair, a failure invisible to same-transform agreement metrics
//!    and visible only on an end task.
//! 3. [`Axis`] shifts a query along a direction defined in read-out space.
//!    Steering without touching the model.
//! 4. [`Index`] searches. Vectors are stored projected and normalized, so
//!    search is a dot product.
//!
//! ```no_run
//! use srt_geometry::{Head, Index, Modality};
//! # let state: Vec<f32> = Vec::new(); // a backbone hidden state, tapped mid-stack
//! let head = Head::from_safetensors_path("sunstone_head.safetensors")?;
//! let q = head.project(&state, Modality::Text);
//! let idx = Index::load("gallery.bin")?;
//! for (key, score) in idx.search(&q, 5) { println!("{score:.4} {key}"); }
//! # Ok::<(), srt_geometry::Error>(())
//! ```

mod axis;
mod head;
mod index;

pub use axis::{Axis, DEFAULT_ALPHA};
pub use head::{Head, Modality};
pub use index::Index;

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Format(String),
    Shape { expected: usize, got: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Format(m) => write!(f, "format: {m}"),
            Error::Shape { expected, got } => {
                write!(f, "shape: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// L2-normalize in place. Returns the pre-normalization norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-8 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    n
}

/// Cosine similarity. Inputs need not be normalized.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-8)
}

/// Mean of a set of equal-length vectors: the per-modality anchor.
///
/// Deployments compute this over their own states to recalibrate a head.
/// Anchor *domain* match matters more than anchor count: 150 in-domain images
/// beat 4,000 out-of-domain ones on a measured runtime pair.
pub fn mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let d = vectors[0].len();
    let mut acc = vec![0.0f64; d];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v) {
            *a += *x as f64;
        }
    }
    let inv = 1.0 / vectors.len() as f64;
    acc.into_iter().map(|a| (a * inv) as f32).collect()
}
