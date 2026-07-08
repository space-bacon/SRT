# Leverage — Top 10 Next Moves (2026-07-08)

Ranked by (impact × readiness), across research, distribution, and credibility.
Companion to [FORWARD_PLAN.md](FORWARD_PLAN.md); this is the strategic view,
that is the operational queue.

## 1. Post the papers to arXiv (or SSRN) — stop being repository-hosted

Three manuscripts exist (SRT-Adapter under [arxiv/](arxiv/paper.md),
[paper_nla.md](paper_nla.md), and the Sunstone/stereogram result) with zero
citable DOIs beyond the two SSRN theory entries. Every other move below
(press, leaderboards, adoption) compounds off a citable preprint. Sunstone's
cross-modal result (§11.6) is arguably a standalone short paper with a viral
figure already made.

## 2. Send the WIRED pitch

Drafted, addressed to Sandra Upson, follows WIRED's pitch rules. The
stereogram story ("the model that couldn't see the Magic Eye until we gave it
eyes") is a tale, not a topic. Cost: 10 minutes. Add byline/credentials and
send; queue Tim Marchman as the 2-week fallback.

## 3. Attack the greedy gap — the program's one real open problem — PREPPED, RETARGETED TO GEMMA-4

Zero-training NN retrieval (ρ≈0.71) still beats the trained greedy decoder
(ρ≈0.28) on Qwen. The chosen lever is draft-conditioned decoding
(retrieval-then-edit): the AV conditions on the NN-retrieved text plus v, so
its worst case is copying the draft (= the NN baseline). Built and
smoke-tested 2026-07-08 (`scripts/train_nla_draft.py`, AV `context_ids`).
Retargeted to the vision backbone gemma-4-31B-it at L47 (the cross-modal
alignment peak) so the same AV powers the vision follow-on: verbalizing
image-position activations (`scripts/gemma4_vision_verbalize.py`). Full
box runbook: `scripts/train_gemma4_nla_all.sh` (phases 0–8). Needs one
96GB+ GPU box.

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

## 8. Extend Sunstone beyond CIFAR-10

The demo's honest caveat is 10 categories / 35 communities. A run on a harder
open-vocabulary set (ImageNet subset, COCO objects) either scales the claim
dramatically or finds its boundary — both publishable. Same geometry-only
method, no training, one box-day. Audio via the Gemma4Unified omni variant is
the follow-on that makes "modality-agnostic semiotics" literal.

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
