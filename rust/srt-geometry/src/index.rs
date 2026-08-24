//! Flat index over head-projected vectors.
//!
//! Rows are stored projected and L2-normalized, so search is a dot product and
//! needs no model. 123K images fit in roughly 250MB at fp16 and search well
//! under a second on a CPU, which is what lets the whole retrieval half of a
//! deployment run client-side with no backbone present at all.
//!
//! On-disk format, little-endian, matching what a browser can `fetch` and use
//! without a parser:
//!
//! ```text
//! magic  "SRTIDX01"        8 bytes
//! dim    u32               projection width
//! count  u32               rows
//! vecs   f16 * dim * count row-major, already normalized
//! keys   u32 len + utf8    repeated `count` times
//! ```

use crate::Error;
use half::f16;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SRTIDX01";

pub struct Index {
    dim: usize,
    mat: Vec<f32>,
    keys: Vec<String>,
}

impl Index {
    pub fn new(dim: usize) -> Self {
        Index { dim, mat: Vec::new(), keys: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Key at a row position, for callers holding integer ground truth.
    pub fn key(&self, row: usize) -> Option<&str> {
        self.keys.get(row).map(|s| s.as_str())
    }

    /// Add a projected, normalized vector. Normalization is enforced here
    /// rather than assumed, because a single unnormalized row silently
    /// outranks everything else.
    pub fn add(&mut self, key: impl Into<String>, mut vec: Vec<f32>) -> Result<(), Error> {
        if vec.len() != self.dim {
            return Err(Error::Shape { expected: self.dim, got: vec.len() });
        }
        crate::normalize(&mut vec);
        self.mat.extend_from_slice(&vec);
        self.keys.push(key.into());
        Ok(())
    }

    /// Top-k by dot product, descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
        if query.len() != self.dim || self.keys.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = (0..self.keys.len())
            .map(|i| {
                let row = &self.mat[i * self.dim..(i + 1) * self.dim];
                (i, row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>())
            })
            .collect();
        let k = k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(i, s)| (self.keys[i].as_str(), s)).collect()
    }

    /// Share of each query's unsteered top-k that survives steering, averaged.
    ///
    /// This is the calibration procedure for [`Axis`](crate::Axis) alpha, and
    /// it is the measurement that stops a steering slider from lying. Class
    /// purity keeps improving as alpha grows, so optimising for it alone walks
    /// straight past the point where the axis has replaced the query and every
    /// search returns the same thing. Retention near 1 means nothing happened;
    /// retention near 0 means the query no longer matters. Pick the largest
    /// alpha that still holds roughly half the neighbourhood.
    ///
    /// Re-derive this per head and per gallery. Inheriting an alpha measured
    /// somewhere else is how the earlier 8x overshoot happened.
    pub fn retention(&self, queries: &[Vec<f32>], axis: &crate::Axis, alpha: f32, k: usize) -> f32 {
        if queries.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for q in queries {
            let before: Vec<&str> = self.search(q, k).into_iter().map(|(s, _)| s).collect();
            let after = self.search(&axis.apply(q, alpha), k);
            let kept = after.iter().filter(|(s, _)| before.contains(s)).count();
            total += kept as f32 / k.max(1) as f32;
        }
        total / queries.len() as f32
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {        let mut f = std::fs::File::create(path)?;
        f.write_all(MAGIC)?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.keys.len() as u32).to_le_bytes())?;
        let mut buf = Vec::with_capacity(self.mat.len() * 2);
        for x in &self.mat {
            buf.extend_from_slice(&f16::from_f32(*x).to_le_bytes());
        }
        f.write_all(&buf)?;
        for k in &self.keys {
            f.write_all(&(k.len() as u32).to_le_bytes())?;
            f.write_all(k.as_bytes())?;
        }
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut f = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(Error::Format("bad magic".into()));
        }
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let vec_bytes = dim * count * 2;
        if bytes.len() < 16 + vec_bytes {
            return Err(Error::Format("truncated vectors".into()));
        }
        let mat: Vec<f32> = bytes[16..16 + vec_bytes]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let mut off = 16 + vec_bytes;
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            if off + 4 > bytes.len() {
                return Err(Error::Format("truncated keys".into()));
            }
            let n = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + n > bytes.len() {
                return Err(Error::Format("truncated key".into()));
            }
            keys.push(String::from_utf8_lossy(&bytes[off..off + n]).into_owned());
            off += n;
        }
        Ok(Index { dim, mat, keys })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_search() {
        let mut idx = Index::new(4);
        idx.add("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add("b", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add("c", vec![0.9, 0.1, 0.0, 0.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(hits[0].0, "a");
        assert_eq!(hits[1].0, "c");

        let tmp = std::env::temp_dir().join("srt_idx_test.bin");
        idx.save(&tmp).unwrap();
        let back = Index::load(&tmp).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.search(&[1.0, 0.0, 0.0, 0.0], 1)[0].0, "a");
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn axis_shifts_ranking_and_random_control_does_not_match() {
        use crate::Axis;
        let ax = Axis::from_vec("up", vec![0.0, 1.0, 0.0, 0.0]);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let steered = ax.apply(&q, 1.0);
        assert!(steered[1] > 0.6, "axis should move the query along itself");
        let rnd = ax.random_like(42);
        assert!(crate::cosine(rnd.as_slice(), ax.as_slice()).abs() < 0.9);
    }
}
