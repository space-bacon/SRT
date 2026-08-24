//! Full-path replay: state -> head -> index -> ranked captions.
//!
//! Single-projection parity proves the matmul. This proves the deployment.
//! The reference is `scripts/retrieval_reference.py`, run on real SugarCrepe
//! images and COCO captions, and the assertion is that Rust reproduces both
//! the ranking and the retrieval score exactly.
//!
//! Regenerate fixtures (they are ~16MB and not in git):
//!
//! ```text
//! python scripts/retrieval_reference.py --cell qwen3b
//! ```

use srt_geometry::{Axis, Head, Index, Modality};

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Case {
    head: Head,
    index: Index,
    fx: serde_json::Value,
}

fn load() -> Option<Case> {
    let d = fixtures();
    let (h, i, f) = (
        d.join("head.safetensors"),
        d.join("captions.srtidx"),
        d.join("retrieval_fixture.json"),
    );
    if !(h.exists() && i.exists() && f.exists()) {
        if std::env::var("SRT_REQUIRE_FIXTURES").is_ok() {
            panic!("SRT_REQUIRE_FIXTURES set but fixtures missing in {}", d.display());
        }
        eprintln!("SKIPPED: run scripts/retrieval_reference.py to generate fixtures");
        return None;
    }
    Some(Case {
        head: Head::from_safetensors_path(h).expect("head"),
        index: Index::load(i).expect("index"),
        fx: serde_json::from_str(&std::fs::read_to_string(f).expect("fixture")).unwrap(),
    })
}

fn state_of(q: &serde_json::Value) -> Vec<f32> {
    q["state"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn rust_reproduces_the_python_ranking_and_scores() {
    let Some(c) = load() else { return };
    let topk = c.fx["topk"].as_u64().unwrap() as usize;
    let queries = c.fx["queries"].as_array().unwrap();

    let mut worst_score_gap = 0.0f32;
    for q in queries {
        let z = c.head.project(&state_of(q), Modality::Image);
        let got = c.index.search(&z, topk);
        let want = q["expect"].as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for (rank, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.0,
                w["key"].as_str().unwrap(),
                "rank {rank} differs for {}",
                q["file"].as_str().unwrap()
            );
            let gap = (g.1 - w["score"].as_f64().unwrap() as f32).abs();
            worst_score_gap = worst_score_gap.max(gap);
        }
    }
    // fp16 index storage is the dominant term in this budget
    assert!(worst_score_gap < 5e-3, "score drift {worst_score_gap}");
}

#[test]
fn rust_reproduces_the_python_recall() {
    let Some(c) = load() else { return };
    let queries = c.fx["queries"].as_array().unwrap();
    let (mut r1, mut r5) = (0usize, 0usize);
    for q in queries {
        let z = c.head.project(&state_of(q), Modality::Image);
        let hits = c.index.search(&z, 5);
        let gold = q["gold"].as_str().unwrap();
        if hits[0].0 == gold {
            r1 += 1;
        }
        if hits.iter().any(|(k, _)| *k == gold) {
            r5 += 1;
        }
    }
    let n = queries.len() as f64;
    let (got1, got5) = (r1 as f64 / n, r5 as f64 / n);
    let want1 = c.fx["i2t"]["r@1"].as_f64().unwrap();
    let want5 = c.fx["i2t"]["r@5"].as_f64().unwrap();
    // 64 sampled queries against the full-set reference: allow sampling error,
    // but a real port break moves this far more than a few points.
    assert!((got1 - want1).abs() < 0.15, "R@1 {got1:.3} vs reference {want1:.3}");
    assert!((got5 - want5).abs() < 0.15, "R@5 {got5:.3} vs reference {want5:.3}");
    assert!(got1 > 0.2, "R@1 {got1:.3} collapsed; retrieval is not working");
}

#[test]
fn recalibrating_with_a_wrong_anchor_degrades_real_retrieval() {
    // The 42KB is load-bearing, and this is the demonstration rather than the
    // assertion. A head pointed at the wrong anchor still returns confident
    // rankings; only the end task shows that they are worse.
    let Some(c) = load() else { return };
    let queries = c.fx["queries"].as_array().unwrap();

    let score = |head: &Head| -> f64 {
        let hit = queries
            .iter()
            .filter(|q| {
                let z = head.project(&state_of(q), Modality::Image);
                c.index.search(&z, 1)[0].0 == q["gold"].as_str().unwrap()
            })
            .count();
        hit as f64 / queries.len() as f64
    };

    let mut wrong = c.head.clone();
    let bad: Vec<f32> = c.head.anchor(Modality::Image).iter().map(|x| x + 1.0).collect();
    wrong.recalibrate(&bad, Modality::Image).unwrap();

    let (good_r1, bad_r1) = (score(&c.head), score(&wrong));
    assert!(
        bad_r1 < good_r1,
        "displaced anchor did not hurt: {bad_r1:.3} vs {good_r1:.3}"
    );
    eprintln!("anchor intact R@1 {good_r1:.3} -> displaced R@1 {bad_r1:.3}");
}

#[test]
fn a_head_space_axis_reorders_real_retrieval_and_a_random_axis_does_not() {
    // The claim the runtime's steering panel rests on, checked on the real
    // gallery. The semantic version of this lives in
    // scripts/headspace_axis_validation.py, which builds axes from text and
    // applies them to images; here the point is that the Rust implementation
    // shows the same shape, including the failure mode at high alpha.
    let Some(c) = load() else { return };
    let queries = c.fx["queries"].as_array().unwrap();

    let zs: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| c.head.project(&state_of(q), Modality::Image))
        .collect();
    let (half, rest) = zs.split_at(zs.len() / 2);
    let axis = Axis::from_contrast("first_half", half, rest);
    let rest = rest.to_vec();

    let displaced = |a: &Axis, alpha: f32| -> f64 {
        rest.iter()
            .filter(|z| c.index.search(z, 1)[0].0 != c.index.search(&a.apply(z, alpha), 1)[0].0)
            .count() as f64
            / rest.len() as f64
    };

    let real = displaced(&axis, srt_geometry::DEFAULT_ALPHA);
    // Ten seeds, because one random draw is an anecdote.
    let rnd: Vec<f64> = (0..10)
        .map(|s| displaced(&axis.random_like(s * 7919 + 1), srt_geometry::DEFAULT_ALPHA))
        .collect();
    let rnd_mean = rnd.iter().sum::<f64>() / rnd.len() as f64;
    eprintln!("alpha={} displaced {real:.3}; random mean {rnd_mean:.3}",
              srt_geometry::DEFAULT_ALPHA);

    // Absolute displacement is a property of gallery density, so the assertion
    // is the separation from the control. Measured here: 0.281 vs 0.125.
    assert!(real > 0.1, "axis barely moved retrieval: {real:.3}");
    assert!(
        real > rnd_mean * 1.8,
        "a random axis nearly matched the real one ({real:.3} vs {rnd_mean:.3})"
    );
}

#[test]
fn retention_falls_with_alpha_and_exposes_the_query_erasure_point() {
    // Without this, a steering slider optimising class purity would happily
    // run to an alpha where every query returns the same results. Retention is
    // the number that says so.
    let Some(c) = load() else { return };
    let zs: Vec<Vec<f32>> = c.fx["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| c.head.project(&state_of(q), Modality::Image))
        .collect();
    let (half, rest) = zs.split_at(zs.len() / 2);
    let axis = Axis::from_contrast("first_half", half, rest);
    let rest = rest.to_vec();

    let r = |a: f32| c.index.retention(&rest, &axis, a, 10);
    let (r0, r_def, r_hi) = (r(0.0), r(srt_geometry::DEFAULT_ALPHA), r(4.0));
    eprintln!("retention: alpha=0 {r0:.3}, default {r_def:.3}, alpha=4 {r_hi:.3}");

    assert!((r0 - 1.0).abs() < 1e-6, "alpha=0 must be a no-op, got {r0}");
    assert!(r_def > r_hi, "retention must fall with alpha");
    assert!(r_hi < 0.2, "alpha=4 should have erased the query, retained {r_hi:.3}");
    assert!(r_def > 0.2, "default alpha erased the query, retained {r_def:.3}");
}
