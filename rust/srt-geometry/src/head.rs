//! The linear read-out head: two projections plus their anchors.

use crate::{normalize, Error};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Text,
}

#[derive(Clone)]
struct Proj {
    /// row-major [out, in]
    w: Vec<f32>,
    b: Vec<f32>,
    mu: Vec<f32>,
    d_in: usize,
    d_out: usize,
}

impl Proj {
    fn apply(&self, v: &[f32], out: &mut [f32]) {
        for o in 0..self.d_out {
            let row = &self.w[o * self.d_in..(o + 1) * self.d_in];
            let mut acc = 0.0f32;
            for i in 0..self.d_in {
                acc += row[i] * (v[i] - self.mu[i]);
            }
            out[o] = acc + self.b[o];
        }
    }
}

/// A trained read-out head.
///
/// Loaded from safetensors with keys `img.weight`, `img.bias`, `txt.weight`,
/// `txt.bias`, `mu_img`, `mu_txt`. Export a published `.pt` head to this
/// format with `scripts/export_head_safetensors.py`; the exporter also emits
/// reference projections so a port can be checked rather than trusted.
#[derive(Clone)]
pub struct Head {
    img: Proj,
    txt: Proj,
}

impl Head {
    pub fn from_safetensors_path<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        Self::from_safetensors(&bytes)
    }

    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, Error> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| Error::Format(format!("safetensors: {e}")))?;
        let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>), Error> {
            let t = st
                .tensor(name)
                .map_err(|_| Error::Format(format!("missing tensor `{name}`")))?;
            let raw = t.data();
            let v: Vec<f32> = match t.dtype() {
                safetensors::Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                d => return Err(Error::Format(format!("dtype {d:?} for `{name}`"))),
            };
            Ok((v, t.shape().to_vec()))
        };

        let mk = |wn: &str, bn: &str, mn: &str| -> Result<Proj, Error> {
            let (w, ws) = get(wn)?;
            let (b, _) = get(bn)?;
            let (mu, _) = get(mn)?;
            if ws.len() != 2 {
                return Err(Error::Format(format!("`{wn}` must be 2-D")));
            }
            let (d_out, d_in) = (ws[0], ws[1]);
            if mu.len() != d_in {
                return Err(Error::Shape { expected: d_in, got: mu.len() });
            }
            if b.len() != d_out {
                return Err(Error::Shape { expected: d_out, got: b.len() });
            }
            Ok(Proj { w, b, mu, d_in, d_out })
        };

        Ok(Head {
            img: mk("img.weight", "img.bias", "mu_img")?,
            txt: mk("txt.weight", "txt.bias", "mu_txt")?,
        })
    }

    pub fn proj_dim(&self) -> usize {
        self.txt.d_out
    }

    pub fn input_dim(&self, m: Modality) -> usize {
        match m {
            Modality::Image => self.img.d_in,
            Modality::Text => self.txt.d_in,
        }
    }

    /// Center by the modality anchor, project, then L2-normalize.
    ///
    /// The centering step is not a preprocessing nicety. These states carry a
    /// dominant shared direction, so uncentered cosine puts unrelated items
    /// far above zero and compresses the differences that matter.
    pub fn project(&self, state: &[f32], m: Modality) -> Vec<f32> {
        let p = match m {
            Modality::Image => &self.img,
            Modality::Text => &self.txt,
        };
        assert_eq!(state.len(), p.d_in, "state dim {} != head {}", state.len(), p.d_in);
        let mut out = vec![0.0f32; p.d_out];
        p.apply(state, &mut out);
        normalize(&mut out);
        out
    }

    /// Replace a modality anchor with one measured on the local runtime.
    ///
    /// This is the whole per-runtime fix: a mean vector, nothing more. It is
    /// load-bearing. On one measured cross-runtime pair, omitting it cost 24
    /// i2t R@1 points, and the loss was invisible to agreement metrics that
    /// apply the same transform to both sides. Only an end task showed it.
    pub fn recalibrate(&mut self, mu: &[f32], m: Modality) -> Result<(), Error> {
        let p = match m {
            Modality::Image => &mut self.img,
            Modality::Text => &mut self.txt,
        };
        if mu.len() != p.d_in {
            return Err(Error::Shape { expected: p.d_in, got: mu.len() });
        }
        p.mu.copy_from_slice(mu);
        Ok(())
    }

    pub fn anchor(&self, m: Modality) -> &[f32] {
        match m {
            Modality::Image => &self.img.mu,
            Modality::Text => &self.txt.mu,
        }
    }
}
