# Leverage — Top 10 Next Moves (2026-07-08)

Ranked by (impact × readiness), across research, distribution, and credibility.
Companion to [FORWARD_PLAN.md](FORWARD_PLAN.md); this is the strategic view,
that is the operational queue.

> **2026-07-24 addendum — the cross-modal linear-head arc changes the board.**
> One box-day produced: Procrustes refuted → anchor rule (domain beats
> size) → **the modality gap is linear** → a trained 22MB linear head
> hits 0.59 R@1 / 0.87 R@5 on frozen gemma-4 with no vision training
> (paper §11.6.4, docs/CROSSMODAL_LINEAR_HEAD.md). Strategic effects:
>
> - **Move 1 (arXiv) strengthens**: the Sunstone standalone paper now has
>   a citable benchmark, a controlled mechanism ladder
>   (orthogonal < identity < linear; MLP never wins), and a boundary
>   revision. This is a complete arc, not a demo.
> - **New product move: ship `srt-sunstone-linear-head`.** The "no new
>   model" pitch is now quantified: cross-modal retrieval as a ~22MB
>   auditable sidecar on an LLM the customer already runs. Backbone
>   untouched = zero regression risk; linear = inspectable; anchors =
>   productizable onboarding calibration ("give us 150–4,000
>   representative images").
> - **New credibility/press move: the scale-floor experiment.** If
>   linear alignment survives on a 2–4B multimodal host, "Sunstone on a
>   Raspberry Pi" is both a product tier (single prefill pass, no
>   generation — viable on Pi-5-class hardware for batch tagging) and a
>   better press artifact than the stereogram alone.
> - Honest scoping for all pitches: the claim is never "beats CLIP"; it
>   is comparable-band retrieval as a free rider on existing LLM infra,
>   from 39k pairs and minutes of training. Karpathy-split eval required
>   before any external number comparison.

## 1. Post the papers to arXiv (or SSRN) — stop being repository-hosted

Three manuscripts exist (SRT-Adapter under [arxiv/](arxiv/paper.md),
[paper_nla.md](paper_nla.md), and the Sunstone/stereogram result) with zero
citable DOIs beyond the two SSRN theory entries. Every other move below
(press, leaderboards, adoption) compounds off a citable preprint. Sunstone's
cross-modal result (§11.6) is arguably a standalone short paper with a viral
figure already made.

## 2. (removed — handled outside the repo)

## 3. Attack the greedy gap — RUN ON GEMMA-4, RESOLVED AS A NEGATIVE + A HYPOTHESIS

Executed 2026-07-08 on gemma-4-31B-it at L47 (`paper_nla.md` §11.7).
Draft-conditioned decoding is refuted with a mechanism: the activation-space
neighbour adds 0.03 nats of predictive value for the gold text, so CE never
learns to use it. The K-curve (+0.017/doubling, never reaching NN) patterns
with gpt-oss and yields the program's sharpest open conjecture: base models
verbalize, instruction-tuned hosts do not. Next lever: same pipeline on the
gemma-4 base checkpoint. Deployable decode on tuned hosts = retrieval, now
validated on the visual channel too (§11.6.3, live in the Sunstone demo).

## 4. Run the gpt-oss-120b port

Runbook exists ([docs/PORTING_GPT_OSS_120B.md](docs/PORTING_GPT_OSS_120B.md)),
sliding-window mask code is tested bit-exact, 20b validated the whole
pipeline. A read-out on a 120B open frontier model alongside the 235B
checkpoint makes "substrate-general" unassailable, for roughly one box-day of
Phase-A compute.

## 5. Finish the MTEB(eng, v2) leaderboard submission for v22c_a050

The run was launched 2026-07-03 on a box that's since gone; verify whether
results survived, relaunch if not, then file the two PRs (ModelMeta +
results). A public leaderboard row is permanent, passive distribution for the
adapter line.

## 6. Harvest the ginigen Metacognition leaderboard results

Four backbones queued since 2026-07-02, scored daily 09:00 KST — results may
already be sitting there. Pull `leaderboard_mcq.json`, check gpt-oss for
harmony parse failures, run the verbalizability-vs-metacog-gain correlation
for §13. That cross-benchmark correlation is a novel finding, nearly free.

## 7. Curate SRT-Sunstone into a real public repo, not a mirror

Currently a full clone of a sprawling research monorepo. Trim to: Sunstone
demo, stereogram scripts, the figure, a focused README leading with the 0.93
retrieval result and the live Space. This is the repo journalists and HN
should click through to; it is what makes moves 1–2 land.

## 8. Extend Sunstone beyond CIFAR-10 — FIRST STEP SHIPPED

Open-vocabulary caption retrieval (10k COCO pool) is live in the Sunstone
demo as a third read-out panel (5/5 CIFAR rank-1, per-category up to 0.778;
§11.6.3). Remaining: harder open-vocab image sets (ImageNet/COCO objects),
and audio via the Gemma4Unified omni variant for literal
modality-agnosticism.

## 9. Fix the Dependabot debt — DONE 2026-07-08

Gradio bumped 6.7.0 → 6.19.0 (CVE-2026-48545 cookie injection, fix validated
on the live Sunstone Space). Transformers CVE-2026-4372 cannot be fixed by
upgrade (patched only in 5.3.0, and 5.x verifiably breaks the adapter's
KV-cached generation); dismissed as tolerable risk on both repos with the
mitigation documented at every pin (fixed first-party model IDs only, no HF
Trainer). Commit `f896edab`.

## 10. Run the two queued mechanistic experiments on the next box

Multi-position spoof test (is the punctuation completeness flag confined to
the aggregation site?) and the Qwen2.5-7B replication of the
L24-surface/L18-semantic split. Together they upgrade the red-teaming section
from "one backbone's quirk" to a cross-backbone claim about layer roles.

## Sequencing

Moves 2, 5, 6, and 9 are each under an hour and unblock passively; do them
first. Then 1 and 7 as a pair (preprint + clean repo). Then pick one
compute-heavy lane (3, 4, or 8) for the next GPU box rather than splitting a
single machine three ways.
