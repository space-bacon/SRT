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

Calibrate `alpha` against a behavioural measurement on neutral inputs. Setting it from likelihood on target-class text is a known trap: on one measured axis the two disagreed by a factor of eight.

## Verification

The port is pinned to the Python path that produced every published number. `scripts/export_head_safetensors.py` writes both a loadable head and a fixture of real states with their expected projections; `cargo test` fails if the Rust read-out drifts from it. A second implementation of the geometry is only worth having if it is checked against the first.

```
python scripts/export_head_safetensors.py \
    --head checkpoints/gemma4_readout/qwen3b_v6_head.pt \
    --states artifacts/.../states.npz --modality txt \
    --out rust/srt-geometry/tests/fixtures
cargo test -p srt-geometry
cargo build --target wasm32-unknown-unknown --release
```

## Where the rest is

This crate is public and deliberately contains no runtime. Model loading, hidden-state taps, tier selection and UI live in BlackWindow, which depends on this and supplies hardware, not maths.
