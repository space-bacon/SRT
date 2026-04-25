# Notes from the Workbench

**James Burton Lancaster**, April 25, 2026

A follow-up to last week's note. v6 finished, v7 finished, v8a finished, v8b is training as I write this. The headline finding is the kind of thing you only see by running the full eval suite on every checkpoint and then running PCA on the things that aren't supposed to be interesting.

## The bombshell

In v6 and v7 the community head has 32 "prototype" vectors — random 64-dimensional anchors that the encoder is meant to learn to mix into a stance vector at decode time. After running the archetype probe (33 Lancaster archetypes against the 32-prototype basis) and finding that the model only ever activated 4 of the 32 prototypes, I pulled out the prototype matrices from v5, v6, and v7 and ran PCA on them.

The mean absolute element difference between v5 and v7's prototypes is **2.7 × 10⁻⁵**, against prototype magnitudes of 0.5 to 1.5. After three full training runs across two months, the prototypes are essentially still at random initialization. The encoder weights moved about 4× more.

The 32-prototype basis was architectural debt. The encoder was doing all the discriminative work and the soft-argmax readout was discarding most of it.

## v8a: removing the bottleneck

I spent a day implementing v8a — same architecture as v7 except the community head emits the encoder output directly, no prototype mixing, no entropy regularizer. Warm-started from v7, trained for 10K steps, and ran the full eval suite.

| Metric | v7 | **v8a** |
|---|---|---|
| Validation cross-entropy | 2.739 | 2.739 |
| Reddit retrieval recall@1 (35 classes) | 0.413 | **0.484** |
| Within/between cosine ratio | 1.006 | **2.016** |
| Archetype recall@1 (33 classes, vs 0.030 chance) | 0.149 | **0.230** |
| Archetype centroid off-diagonal cosine | 0.999 | **0.873** |
| Trajectory anisotropy (λ_max / λ_min) | 72 | **23,333** |
| TruthfulQA hallucination AUROC (mean r̂) | 0.578 | 0.577 |
| Regime calibration ECE | 0.00085 | 0.00091 |

CE didn't move. Reddit retrieval *doubled* in within/between cosine separation. Archetype recall went from 4.9× chance to 7.6× chance. The off-diagonal cosine of archetype centroids fell from 0.999 (essentially co-linear; the 32-prototype mixture was collapsing them onto 4 anchors) to 0.873 (distinct directions emerging).

The trajectory metric is the one I find most telling. Before v8a, the community vector for archetype-conditioned generations traced a tiny, near-isotropic blob in 64-dimensional space — one direction was 72× longer than the smallest, log-determinant of covariance about −557. After v8a, the same blob has one direction 23,333× longer than the smallest and log-determinant −476, which means the volume of the covariance ellipsoid grew by a factor of e⁸¹. The encoder organizes archetype-conditioned text along a small number of dominant directions in the continuous space. The prototype layer was throwing this structure away.

## What this means

The community channel is not a clustering head. It is a continuous coordinate system over what I'll call discourse-trajectory space — a small number of dominant directions along which an utterance's stance can vary, with the encoder learning to place each input somewhere in that space. The prototype basis was a misreading of what the architecture was learning to do.

Three independent methodologies — Reddit subreddit labels (training signal), Lancaster's archetypes (external taxonomy from sustained human-model interaction), and the Lexicon of Synthetic Interiority (a separate cross-model vocabulary) — now all agree that there is meaningful low-dimensional stance structure being learned, and v8a recovers it 7.6× above chance without ever having seen the archetypes during training.

## v8b

v8b is in flight. Same architecture as v8a, but I've doubled the contrastive loss weight (2.0 → 4.0) and halved the temperature (0.1 → 0.05). The hypothesis is that v8a's archetype centroid off-diagonal cosine is still 0.873 — the centroids are distinct but correlated. With the prototype bottleneck gone, the supervised contrastive loss is now the only discriminative pressure on the community channel. Pushing it harder should orthogonalize the centroids further without disturbing token cross-entropy. If v8b lands the off-diagonal under 0.85 with archetype recall ≥ 0.23 and CE flat, that's the v8 line for release.

## Estimated release timeline

- v8b training completes: **late April 25** (~5 hours from now)
- v8b eval suite: **April 26**
- If v8b improves on v8a across the board: **public weights + paper §5.9 update + v8 release in early May.** If v8a remains the strongest, I release v8a directly.
- Either way, the release will include adapter weights, the eval harness for all 7 probe scripts, the trajectory_eval comparison runs against v6/v7/v8a, the archetype-conditioned generation set (986 sentences across 33 archetypes), and the paper.

The paper's §5.8 (the PCA finding) and §5.9 (v8a results) are in main now. Worth a read if you want the full version of the bombshell with the table.

More soon.

J.
