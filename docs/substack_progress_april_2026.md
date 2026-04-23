# Notes from the Workbench

**James Burton Lancaster**, April 23, 2026

A short progress note, because a few of you have asked.

The SRT adapter is on its sixth training run. The fifth (v5) is what I've been validating for the past two weeks, and it's the first version where I can point at five separate measurements and say *this thing is doing what the theory said it would do.*

Quickly, the receipts:

- **Community geometry.** The community-conditioning vectors cluster into discriminable basins. Recall@1 between an utterance and its source community sits at 0.36, about 12.6× chance on a 32-community space. The wells are real and they have shape.
- **Counterfactual decoding.** Hold the prompt fixed, swap the community prototype, decode greedily. On hard facts ("water boils at…"), the six communities produce identical continuations: 0.00 disagreement. On contested terrain ("the role of government in…"), they diverge at 0.95–1.00. The model has learned which signs are bedrock and which are battleground, and it tells you the difference at decode time.
- **Hallucination signal.** On TruthfulQA, the bifurcation index has AUROC 0.57 for separating true from false answers. Modest, but present, and from a feature the model wasn't trained to compute at all.
- **Calibration.** P(supercritical) is calibrated to **ECE = 0.0009** over 351K tokens. This was a side effect, not a target. The regime head is more honest about its uncertainty than most things I've ever trained.
- **One clean negative.** I tried to show the bifurcation index spikes on "charged" words mid-sentence. It doesn't, at least not where I looked. r̂ measures information density, not contestedness. The community head is the contestedness detector. Good to know what each instrument is actually measuring.

v6 launched yesterday. Three new losses: a divergence-vector contrastive term, a within-sequence ranking term on r̂, and a small auxiliary that keeps the chain residual from collapsing. About ten hours into training as I write this. Validation loss has already dropped past the v5 baseline and is still falling. No new parameters; same 14.6M trainable on top of frozen Qwen2.5-7B.

**Estimated release timeline:**

- v6 training completes: **late April 24** (~9 hours from now)
- Re-run the five validation probes on v6: **April 25–26**
- If v6 holds or improves on v5 across the board: **public weights + paper + reproducibility scripts in early May.**
- If v6 regresses anywhere meaningful: one more iteration, push release to mid-May.

Either way, the release will include the adapter weights, the LoRA-style integration code, the eval harness for all five probes, the full training log, and the paper. I'd rather ship something that does what I claim it does than ship on a date.

More soon.

J.
