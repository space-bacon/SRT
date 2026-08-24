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
///
/// Either side may be omitted. A runtime that only encodes text does not need
/// the image projection, because the gallery it searches was projected
/// elsewhere and ships already in read-out space. Dropping the unused side and
/// storing fp16 takes a typical head from ~24MB to ~2MB, which is the
/// difference between a page a phone will load and one it will not.
#[derive(Clone)]
pub struct Head {
    img: Option<Proj>,
    txt: Option<Proj>,
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

        let mk = |wn: &str, bn: &str, mn: &str| -> Result<Option<Proj>, Error> {
            if st.tensor(wn).is_err() {
                return Ok(None);
            }
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
            Ok(Some(Proj { w, b, mu, d_in, d_out }))
        };

        let head = Head {
            img: mk("img.weight", "img.bias", "mu_img")?,
            txt: mk("txt.weight", "txt.bias", "mu_txt")?,
        };
        if head.img.is_none() && head.txt.is_none() {
            return Err(Error::Format("head has neither modality".into()));
        }
        Ok(head)
    }

    fn side(&self, m: Modality) -> Result<&Proj, Error> {
        match m {
            Modality::Image => &self.img,
            Modality::Text => &self.txt,
        }
        .as_ref()
        .ok_or_else(|| Error::Format(format!("head carries no {m:?} projection")))
    }

    pub fn has(&self, m: Modality) -> bool {
        match m {
            Modality::Image => self.img.is_some(),
            Modality::Text => self.txt.is_some(),
        }
    }

    pub fn proj_dim(&self) -> usize {
        self.txt
            .as_ref()
            .or(self.img.as_ref())
            .map(|p| p.d_out)
            .unwrap_or(0)
    }

    pub fn input_dim(&self, m: Modality) -> usize {
        self.side(m).map(|p| p.d_in).unwrap_or(0)
    }

    /// Center by the modality anchor, project, then L2-normalize.
    ///
    /// The centering step is not a preprocessing nicety. These states carry a
    /// dominant shared direction, so uncentered cosine puts unrelated items
    /// far above zero and compresses the differences that matter.
    ///
    /// Returns an error rather than panicking on a width mismatch: pairing a
    /// head with the wrong backbone is a realistic deployment mistake, and in
    /// WebAssembly a panic aborts the module and takes the worker with it.
    pub fn project(&self, state: &[f32], m: Modality) -> Result<Vec<f32>, Error> {
        let p = self.side(m)?;
        if state.len() != p.d_in {
            return Err(Error::Shape { expected: p.d_in, got: state.len() });
        }
        let mut out = vec![0.0f32; p.d_out];
        p.apply(state, &mut out);
        normalize(&mut out);
        Ok(out)
    }

    /// Replace a modality anchor with one measured on the local runtime.
    ///
    /// This is the whole per-runtime fix: a mean vector, nothing more. It is
    /// load-bearing, and not subtly. Measured on a quantized runtime against
    /// the fp16 one its head was fit on, text-to-image R@1 read 0.008 with the
    /// head as-is and 0.229 after 4KB of anchor, matching the reference. The
    /// failure is silent: rankings look confident either way, and no metric
    /// that applies the same transform to both sides of a comparison will show
    /// it. Only an end task does.
    pub fn recalibrate(&mut self, mu: &[f32], m: Modality) -> Result<(), Error> {
        let p = match m {
            Modality::Image => &mut self.img,
            Modality::Text => &mut self.txt,
        }
        .as_mut()
        .ok_or_else(|| Error::Format(format!("head carries no {m:?} projection")))?;
        if mu.len() != p.d_in {
            return Err(Error::Shape { expected: p.d_in, got: mu.len() });
        }
        p.mu.copy_from_slice(mu);
        Ok(())
    }

    pub fn anchor(&self, m: Modality) -> &[f32] {
        self.side(m).map(|p| p.mu.as_slice()).unwrap_or(&[])
    }
}
