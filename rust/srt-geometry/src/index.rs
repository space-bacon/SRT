//! Flat index over head-projected vectors.
//!
//! Rows are stored projected and L2-normalized, so search is a dot product and
//! needs no model. That is what lets the retrieval half of a deployment run
//! client-side with no backbone present at all.
//!
//! # Precision
//!
//! At gallery scale the binding constraint is memory, not download. A
//! 123K-image gallery held as `f32` is 505MB resident, which no phone keeps.
//! Measured against fp32 on real retrieval, 1,000 images and 5,001 captions:
//!
//! | store | t2i R@1 | 123K gallery |
//! |---|---|---|
//! | f32 | 0.2300 | 505 MB |
//! | f16 | 0.2300 | 252 MB |
//! | int8, per-row scale | 0.2306 | 127 MB |
//!
//! int8 costs nothing here, so it is what anything large should use. The rows
//! are unit-norm, which is why one symmetric scale per row suffices.
//!
//! # On-disk format, little-endian
//!
//! ```text
//! magic  "SRTIDX01" (f16) or "SRTIDX02" (int8)
//! dim    u32
//! count  u32
//! [02]   scales  f32 * count
//! vecs   f16 or i8, dim * count, row-major, already normalized
//! keys   u32 len + utf8, repeated count times
//! ```

use crate::Error;
use half::f16;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC_F16: &[u8; 8] = b"SRTIDX01";
const MAGIC_I8: &[u8; 8] = b"SRTIDX02";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F16,
    /// Symmetric per-row int8. Free on measured retrieval, quarter the memory.
    Int8,
}

enum Store {
    F16(Vec<f16>),
    Int8 { data: Vec<i8>, scale: Vec<f32> },
}

pub struct Index {
    dim: usize,
    store: Store,
    keys: Vec<String>,
}

impl Index {
    pub fn new(dim: usize) -> Self {
        Self::with_precision(dim, Precision::F16)
    }

    pub fn with_precision(dim: usize, p: Precision) -> Self {
        Index {
            dim,
            store: match p {
                Precision::F16 => Store::F16(Vec::new()),
                Precision::Int8 => Store::Int8 { data: Vec::new(), scale: Vec::new() },
            },
            keys: Vec::new(),
        }
    }

    pub fn precision(&self) -> Precision {
        match self.store {
            Store::F16(_) => Precision::F16,
            Store::Int8 { .. } => Precision::Int8,
        }
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

    /// Bytes the vectors occupy in memory, excluding keys.
    pub fn resident_bytes(&self) -> usize {
        match &self.store {
            Store::F16(v) => v.len() * 2,
            Store::Int8 { data, scale } => data.len() + scale.len() * 4,
        }
    }

    /// Key at a row position, for callers holding integer ground truth.
    pub fn key(&self, row: usize) -> Option<&str> {
        self.keys.get(row).map(|s| s.as_str())
    }

    /// Add a projected vector. Normalization is enforced rather than assumed,
    /// because one unnormalized row silently outranks everything else.
    pub fn add(&mut self, key: impl Into<String>, mut vec: Vec<f32>) -> Result<(), Error> {
        if vec.len() != self.dim {
            return Err(Error::Shape { expected: self.dim, got: vec.len() });
        }
        crate::normalize(&mut vec);
        match &mut self.store {
            Store::F16(v) => v.extend(vec.iter().map(|x| f16::from_f32(*x))),
            Store::Int8 { data, scale } => {
                let m = vec.iter().fold(0.0f32, |a, x| a.max(x.abs()));
                let s = (m / 127.0).max(1e-12);
                data.extend(vec.iter().map(|x| (x / s).round().clamp(-127.0, 127.0) as i8));
                scale.push(s);
            }
        }
        self.keys.push(key.into());
        Ok(())
    }

    /// The stored vector for one row, dequantized and unit-normalized as
    /// `add` left it.
    ///
    /// A deployment that already holds the index holds a record for every item
    /// in it, so anything that reads records, a verbalizer for instance, needs
    /// no further download.
    pub fn row(&self, row: usize) -> Option<Vec<f32>> {
        if row >= self.keys.len() {
            return None;
        }
        let mut v = match &self.store {
            Store::F16(d) => d[row * self.dim..(row + 1) * self.dim]
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<f32>>(),
            Store::Int8 { data, scale } => data[row * self.dim..(row + 1) * self.dim]
                .iter()
                .map(|x| *x as f32 * scale[row])
                .collect::<Vec<f32>>(),
        };
        crate::normalize(&mut v);
        Some(v)
    }

    /// Element-wise mean of every stored vector, normalized.
    ///
    /// The control for a verbalizer: what a reader says when handed the
    /// average of all records rather than any one of them.
    pub fn mean_row(&self) -> Vec<f32> {
        let mut acc = vec![0f32; self.dim];
        for i in 0..self.keys.len() {
            if let Some(v) = self.row(i) {
                for (a, x) in acc.iter_mut().zip(&v) {
                    *a += *x;
                }
            }
        }
        let n = self.keys.len().max(1) as f32;
        for a in acc.iter_mut() {
            *a /= n;
        }
        crate::normalize(&mut acc);
        acc
    }

    fn row_score(&self, i: usize, query: &[f32]) -> f32 {
        match &self.store {
            Store::F16(v) => {
                let row = &v[i * self.dim..(i + 1) * self.dim];
                row.iter().zip(query).map(|(a, b)| a.to_f32() * b).sum()
            }
            Store::Int8 { data, scale } => {
                let row = &data[i * self.dim..(i + 1) * self.dim];
                let dot: f32 = row.iter().zip(query).map(|(a, b)| *a as f32 * b).sum();
                dot * scale[i]
            }
        }
    }

    /// Top-k as (row, score), descending.
    ///
    /// Rows rather than keys because a deployment usually has side-files laid
    /// out in gallery order, and addressing them by position beats shipping a
    /// second key-to-offset map the size of the key table.
    pub fn search_rows(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dim || self.keys.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = (0..self.keys.len())
            .map(|i| (i, self.row_score(i, query)))
            .collect();
        let k = k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Top-k by dot product, descending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(&str, f32)> {
        self.search_rows(query, k)
            .into_iter()
            .map(|(i, s)| (self.keys[i].as_str(), s))
            .collect()
    }

    /// Share of each query's unsteered top-k that survives steering, averaged.
    ///
    /// This is the calibration procedure for [`Axis`](crate::Axis) alpha, and
    /// the measurement that stops a steering slider from lying. Class purity
    /// keeps improving as alpha grows, so optimising for it alone walks past
    /// the point where the axis has replaced the query and every search
    /// returns the same thing. Retention near 1 means nothing happened; near 0
    /// means the query no longer matters. Pick the largest alpha that still
    /// holds roughly half the neighbourhood.
    ///
    /// Re-derive per head and per gallery. Inheriting an alpha measured
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

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let mut f = std::fs::File::create(path)?;
        match &self.store {
            Store::F16(v) => {
                f.write_all(MAGIC_F16)?;
                f.write_all(&(self.dim as u32).to_le_bytes())?;
                f.write_all(&(self.keys.len() as u32).to_le_bytes())?;
                let mut buf = Vec::with_capacity(v.len() * 2);
                for x in v {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                f.write_all(&buf)?;
            }
            Store::Int8 { data, scale } => {
                f.write_all(MAGIC_I8)?;
                f.write_all(&(self.dim as u32).to_le_bytes())?;
                f.write_all(&(self.keys.len() as u32).to_le_bytes())?;
                let mut buf = Vec::with_capacity(scale.len() * 4);
                for s in scale {
                    buf.extend_from_slice(&s.to_le_bytes());
                }
                f.write_all(&buf)?;
                let bytes: Vec<u8> = data.iter().map(|x| *x as u8).collect();
                f.write_all(&bytes)?;
            }
        }
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
        if bytes.len() < 16 {
            return Err(Error::Format("truncated header".into()));
        }
        let magic = &bytes[..8];
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut off = 16usize;

        let store = if magic == MAGIC_F16 {
            let n = dim * count * 2;
            if bytes.len() < off + n {
                return Err(Error::Format("truncated vectors".into()));
            }
            let v: Vec<f16> = bytes[off..off + n]
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();
            off += n;
            Store::F16(v)
        } else if magic == MAGIC_I8 {
            let ns = count * 4;
            let nd = dim * count;
            if bytes.len() < off + ns + nd {
                return Err(Error::Format("truncated vectors".into()));
            }
            let scale: Vec<f32> = bytes[off..off + ns]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += ns;
            let data: Vec<i8> = bytes[off..off + nd].iter().map(|b| *b as i8).collect();
            off += nd;
            Store::Int8 { data, scale }
        } else {
            return Err(Error::Format("bad magic".into()));
        };

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
        Ok(Index { dim, store, keys })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_search() {
        for p in [Precision::F16, Precision::Int8] {
            let mut idx = Index::with_precision(4, p);
            idx.add("a", vec![1.0, 0.0, 0.0, 0.0]).unwrap();
            idx.add("b", vec![0.0, 1.0, 0.0, 0.0]).unwrap();
            idx.add("c", vec![0.9, 0.1, 0.0, 0.0]).unwrap();
            let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 2);
            assert_eq!(hits[0].0, "a", "{p:?}");
            assert_eq!(hits[1].0, "c", "{p:?}");

            let tmp = std::env::temp_dir().join(format!("srt_idx_{p:?}.bin"));
            idx.save(&tmp).unwrap();
            let back = Index::load(&tmp).unwrap();
            assert_eq!(back.len(), 3);
            assert_eq!(back.precision(), p);
            assert_eq!(back.search(&[1.0, 0.0, 0.0, 0.0], 1)[0].0, "a");
            std::fs::remove_file(tmp).ok();
        }
    }

    #[test]
    fn int8_halves_f16_memory_and_keeps_the_same_ranking() {
        let dim = 256;
        let mk = |p| {
            let mut idx = Index::with_precision(dim, p);
            for i in 0..200 {
                let v: Vec<f32> = (0..dim)
                    .map(|j| (((i * 37 + j * 11) % 97) as f32 - 48.0) * 0.01)
                    .collect();
                idx.add(format!("k{i}"), v).unwrap();
            }
            idx
        };
        let (a, b) = (mk(Precision::F16), mk(Precision::Int8));
        let vec_bytes = b.resident_bytes() - b.len() * 4;
        assert_eq!(vec_bytes * 2, a.resident_bytes());

        let q: Vec<f32> = (0..dim).map(|j| ((j % 13) as f32 - 6.0) * 0.02).collect();
        let (ha, hb) = (a.search(&q, 5), b.search(&q, 5));
        assert_eq!(
            ha.iter().map(|x| x.0).collect::<Vec<_>>(),
            hb.iter().map(|x| x.0).collect::<Vec<_>>(),
            "int8 reordered the top 5 against f16"
        );
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
