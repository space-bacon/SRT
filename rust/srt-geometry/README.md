# srt-geometry

The arithmetic between a frozen model's hidden states and a usable answer, in Rust, with no model dependency.

This crate exists because of a specific claim the SRT program makes and has measured: the read-out is a property of the representation, not of the runtime that produced it. If that is true, the read-out code should be movable onto any hardware without changing, and the only per-runtime artifact should be small. This crate is that claim written as a dependency graph. It compiles to `wasm32-unknown-unknown` and to native, it pulls in no inference engine, and the entire per-runtime correction is a mean vector.

## What it does

Four operations, in the order a deployment performs them.

| type | operation |
|---|---|
| `Head` | project a hidden state into the shared read-out space |
| `Head::recalibrate` | swap in anchors measured on the local runtime |
| `Axis` | shift a projected query along a behavioural direction |
| `Index` | search, as a dot product over stored projected vectors |

Nothing here loads a model. Retrieval, steering and search operate on projected vectors, which is why a deployment can do all three with no backbone present at all.

## Centering is not optional

The states these heads consume carry a dominant shared direction. Uncentered cosine on them puts unrelated items far above zero and compresses the differences that matter, so a raw similarity number is not interpretable on its own. `Head::project` therefore subtracts the modality anchor before projecting, always, with no flag to turn it off.

## Recalibration

A head trained against one runtime does not transfer unchanged to another, because the two place their states in slightly different positions. The fix is a mean vector per modality, roughly 42KB, and nothing is retrained.

Two things about collecting it, both learned by getting them wrong:

- Omitting it cost 24 i2t R@1 points on one measured runtime pair, and the loss was invisible to every metric that applies the same transform to both sides of a comparison. Only an end task surfaced it. Do not certify a recalibration with an agreement score.
- Anchor domain beats anchor count. 150 in-domain anchors beat 4,000 out-of-domain ones.

## Steering

`Axis` acts on the projected query rather than the residual stream, so it costs one vector addition and no re-encode, and the artifact is about 2KB. `Axis::random_like` produces a matched-norm random direction, because a steering claim means nothing until you have shown a random axis does not reproduce it.

The mechanism is validated on real data by `scripts/headspace_axis_validation.py`: axes are built from **captions** and applied to **image** queries, so a positive result is a claim about a shared space and not about memorised neighbours, and the captions used to build an axis are held out from the images evaluated. Three text-defined contrasts, 32 matched-norm random controls per point, top-20 class purity:

| alpha | animal←vehicle | food←sport | indoor←outdoor | random control | retention |
|---|---|---|---|---|---|
| 0 | 0.024 | 0.004 | 0.009 | same | 1.00 |
| 0.5 | 0.252 | 0.117 | 0.128 | 0.009–0.024 | 0.61–0.77 |
| 1.0 | 0.744 | 0.640 | 0.587 | 0.010–0.025 | 0.13–0.29 |
| 2.0 | 0.908 | 0.851 | 0.895 | 0.013–0.029 | 0.01–0.03 |

The random control never moves off baseline at any alpha (z-scores of the real axis run 12 to 520), so the direction is carrying meaning rather than degrading the query into a class prior.

**The high-alpha numbers are not the result.** At alpha 2 retention is 0.01, which means the axis has replaced the query and every search returns the same images. That is why `Index::retention` exists and why `DEFAULT_ALPHA` is 0.5 rather than the alpha with the best-looking purity. Re-derive it per head and per gallery; do not inherit it.

## Verification

The port is pinned to the Python path that produced every published number, at two levels.

`scripts/export_head_safetensors.py` writes a loadable head plus real states with their expected projections, checking the projection itself. `scripts/retrieval_reference.py` runs the whole deployment path on real SugarCrepe images and COCO captions and writes both the result and a replay fixture, checking the thing that actually ships. Rust reproduces the Python ranking exactly across 64 queries × top-10, with score drift under 5e-3.

Reference numbers, 1,542 images against 1,541 deduplicated captions:

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| i2t | 0.394 | 0.691 | 0.796 |
| shuffled control | 0.001 | 0.007 | 0.009 |

One detail worth keeping: the reference scores against the **fp16** index that ships, not the fp32 vectors it was built from. Those two disagree on near-ties, and reporting the fp32 number would be reporting a gallery no deployment ever holds.

```
python scripts/export_head_safetensors.py \
    --head checkpoints/gemma4_readout/qwen3b_v6_head.pt \
    --states artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz \
    --modality txt --out rust/srt-geometry/tests/fixtures
python scripts/retrieval_reference.py --cell qwen3b
python scripts/headspace_axis_validation.py
cargo test -p srt-geometry
cargo build --target wasm32-unknown-unknown --release
```

Fixtures are ~20MB and gitignored, so the tests skip when they are absent. `SRT_REQUIRE_FIXTURES=1` turns a missing fixture into a failure, because a skip that passes silently is the same hazard the tests exist to catch.

## Galleries

`scripts/export_index_srtidx.py` writes the `SRTIDX01` file the Rust index reads, from either pre-projected vectors or raw states plus a head. It prints the index's own mean pairwise cosine on the way out: head space should be close to isotropic, and a high value there means the projection is not doing its job and every retrieval score will read as suspiciously high. Measured 0.021 on the COCO val gallery and 0.041 on SugarCrepe.

## Where the rest is

This crate is public and deliberately contains no runtime. Model loading, hidden-state taps, tier selection and UI live in BlackWindow, which depends on this and supplies hardware, not maths.
