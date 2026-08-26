//! Turning one record into soft tokens.
//!
//! The head projects a state into the shared space; this goes the other way
//! round the loop, taking a vector and producing the prefix a language model
//! reads before it starts writing. Two layers and a GELU, which is small
//! enough to be plain arithmetic like the rest of this crate: the runtime
//! supplies hardware, not maths.
//!
//! Weights come from `scripts/export_verbalizer_browser.py`. They are held at
//! half precision because the browser pays for every megabyte it keeps
//! resident, and widened per multiply-accumulate. The centring statistics ride
//! in the same file: the prefix was fitted in that frame, and an uncentred
//! vector degrades the output quietly rather than failing.

use safetensors::SafeTensors;

use crate::Error;

/// A prefix network: record in, soft tokens out.
pub struct Verbalizer {
    w0: Vec<half::f16>,
    b0: Vec<f32>,
    w2: Vec<half::f16>,
    b2: Vec<f32>,
    mu: Vec<f32>,
    sd: f32,
    hidden: usize,
    pub d_in: usize,
    pub n_tok: usize,
    pub d_model: usize,
}

fn f32s(t: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    let raw = t.data();
    match t.dtype() {
        safetensors::Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

fn f16s(t: &safetensors::tensor::TensorView<'_>) -> Vec<half::f16> {
    let raw = t.data();
    match t.dtype() {
        safetensors::Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
    }
}

impl Verbalizer {
    pub fn from_safetensors(bytes: &[u8], n_tok: usize, d_model: usize) -> Result<Self, Error> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| Error::Format(format!("safetensors: {e}")))?;
        let get = |name: &str| {
            st.tensor(name)
                .map_err(|_| Error::Format(format!("missing tensor `{name}`")))
        };

        let t0 = get("net0.weight")?;
        let t2 = get("net2.weight")?;
        let (hidden, d_in) = (t0.shape()[0], t0.shape()[1]);
        let out = t2.shape()[0];
        if t2.shape()[1] != hidden {
            return Err(Error::Shape { expected: hidden, got: t2.shape()[1] });
        }
        if out != n_tok * d_model {
            return Err(Error::Shape { expected: n_tok * d_model, got: out });
        }
        let mu = f32s(&get("mu")?);
        if mu.len() != d_in {
            return Err(Error::Shape { expected: d_in, got: mu.len() });
        }
        let sd = f32s(&get("sd")?)
            .first()
            .copied()
            .ok_or_else(|| Error::Format("empty `sd`".into()))?;

        Ok(Self {
            w0: f16s(&t0),
            b0: f32s(&get("net0.bias")?),
            w2: f16s(&t2),
            b2: f32s(&get("net2.bias")?),
            mu,
            sd,
            hidden,
            d_in,
            n_tok,
            d_model,
        })
    }

    /// One record -> `n_tok * d_model` values, laid out token-major.
    pub fn soft_tokens(&self, v: &[f32]) -> Result<Vec<f32>, Error> {
        if v.len() != self.d_in {
            return Err(Error::Shape { expected: self.d_in, got: v.len() });
        }
        let x: Vec<f32> = v
            .iter()
            .zip(&self.mu)
            .map(|(a, m)| (a - m) / self.sd)
            .collect();

        let mut h = vec![0f32; self.hidden];
        for (j, hj) in h.iter_mut().enumerate() {
            let row = &self.w0[j * self.d_in..(j + 1) * self.d_in];
            let dot: f32 = row.iter().zip(&x).map(|(w, a)| w.to_f32() * a).sum();
            let z = dot + self.b0[j];
            // GELU (erf), matching torch.nn.GELU's default.
            *hj = 0.5 * z * (1.0 + erf(z * std::f32::consts::FRAC_1_SQRT_2));
        }

        let out_dim = self.n_tok * self.d_model;
        let mut o = vec![0f32; out_dim];
        for (j, oj) in o.iter_mut().enumerate() {
            let row = &self.w2[j * self.hidden..(j + 1) * self.hidden];
            let dot: f32 = row.iter().zip(&h).map(|(w, a)| w.to_f32() * a).sum();
            *oj = dot + self.b2[j];
        }
        Ok(o)
    }
}

/// Abramowitz and Stegun 7.1.26, which is well inside f16 weight noise.
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_)* t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_59)
            * t
            * (-x * x).exp();
    s * y
}
