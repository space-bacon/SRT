# Leverage — Top Moves (revised 2026-08-29)

Ranked by (impact × readiness), across research, distribution, and credibility.
Companion to [FORWARD_PLAN.md](FORWARD_PLAN.md); this is the strategic view,
that is the operational queue.

> **2026-08-29 rewrite — the board changed and one prior move was backwards.**
> Three things reorder everything below.
>
> 1. **Medical imaging is the sharpest claim we have.** A linear probe on
>    frozen `gemma-4-31B-it` states scores **0.7590** mean AUROC on the
>    official ChestX-ray14 split against **0.7451** for the dataset authors'
>    fine-tuned ResNet-50, ahead on 12 of 14 findings. Split-matched,
>    patient-disjoint, three controls, published as dataset, model and Space.
> 2. **The caption head is the bottleneck, and it is now measured.**
>    Cross-vendor image agreement is **0.8024** against a 0.0007 floor;
>    within-vendor text-to-image is **0.1050**. Swapping the caption tower
>    moves both terms together (0.938 and 0.973), so the head is the shared
>    constraint. That 8x gap sits in one component and is the largest piece
>    of headroom in the program.
> 3. **GPUs are needed for exactly one thing: encoding.** Every fit, sweep,
>    probe and retrieval here runs on CPU. The chest probe refits locally in
>    48 seconds. Budget compute for "get a new backbone's reading of an image"
>    and for nothing else.
>
> **The previous move 5, finishing the MTEB submission, is reversed.** Our
> English STS cosine-Spearman is 0.5192 against a board where competitive
> models sit at 0.80 or better. Submitting would enter a competition the
> adapter was never built for. The STS line is closed; see FORWARD_PLAN. What
> we filed instead is a measurement question, MTEB issue #5330.

> **Standing note on corrections.** Two published claims were withdrawn on
> 2026-08-29: the vendor-first routing recipe (held on photographs, failed to
> replicate on radiology and satellite) and the enclosed-area reading of the
> cycle result (area was confounded with the number of distinct vendor
> boundaries crossed). Both were caught by controls a public reviewer asked
> for. The willingness to run those and publish the result is a real asset in
> that relationship and should not be traded for a cleaner-looking claim.

## 1. Post the papers — still the top blocker

Three manuscripts exist ([arxiv/](arxiv/paper.md), [paper_nla.md](paper_nla.md),
[paper_srt_program.md](paper_srt_program.md)) with no citable DOI beyond the two
SSRN theory entries. Every move below compounds off a preprint. Two blockers
cleared today: the "mean MTEB-STS 0.3744" phrasing that read like a leaderboard
figure is now labelled as our own harness, and the split-matched baseline for
the radiograph result is in hand.

The medical result is arguably the standalone short paper, and it is the one
with a reviewer-legible headline.

## 2. Attack the caption head

The largest measured gap in the program: 0.8024 image agreement against 0.1050
text-to-image, with the head confirmed as the shared bottleneck by swap. Every
retention-style number is throttled by it and the cross-vendor product is capped
by it. Options: a better head architecture, more pair data, a stronger
objective, or a distilled teacher. States for four vendors across three domains
are already on disk, so iteration is fast and needs no GPU.

## 3. Generalize the radiograph result across backbones — IN FLIGHT

`scripts/cxr_vendor_compare.py` and `scripts/cxr_probe_transport.py` are queued
on the current box. Two questions: is chest pathology linearly present in every
backbone, and is it the *same direction*. The transport arm trains the probe on
one vendor and reads it on another through a ridge map, which makes "train once,
read everywhere" a measurement on a clinical target rather than a slogan.

Every reader of the published result will ask whether 0.7590 is a gemma4 fact.

## 4. NLST and the early-detection question — BLOCKED ON ACCESS

CDAS quotes 1 to 4 weeks. The current CT result (0.9380 AUROC, 620 slices, 37 of
38 studies ranking the lesion higher) is **detection, not early detection**, and
the published scoping says so. The Lung Cancer Selection with its 438 interval
and post-screening cancers is what would change that, using nodule-positive
benign participants as controls rather than clean scans.

This is the only line that could produce a clinically interesting claim rather
than a representational one.

## 5. Measurement discipline as a contribution

MTEB issue #5330 is filed: models tying on a single-pass retrieval score is not
evidence they carry the same structure, with the iteration ladder as evidence.
Issue #4842, the Massive Omni Embedding Benchmark research project, is open and
active, and our four-vendor omni states are close to what it needs. That is a
better home for the portability work than any submission we could make alone.

## 6. Curate a clean public repo for the medical work

Same argument as the old move 7, new subject. The result has a dataset, a model,
a running Space and a pinned collection, but the code lives in a sprawling
research monorepo. A focused repo leading with the split-matched table is what a
journalist or a radiologist would click through.

## 7. Run the gpt-oss-120b port

Runbook exists ([docs/PORTING_GPT_OSS_120B.md](docs/PORTING_GPT_OSS_120B.md)),
sliding-window mask code tested bit-exact, 20b validated the pipeline. Roughly
one box-day. Less urgent than it was: the cross-vendor and medical results carry
the substrate-generality weight more directly now.

## 8. The two queued mechanistic experiments

Multi-position spoof test (is the punctuation completeness flag confined to the
aggregation site?) and the Qwen2.5-7B replication of the L24-surface /
L18-semantic split. Together they upgrade the red-teaming section from one
backbone's quirk to a cross-backbone claim about layer roles.

## 9. Extend the cross-vendor work to a fifth vendor

Everything measured so far uses four. The edge-count ladder, the composition
result and the head swap all rest on that set. A fifth vendor is the cheapest
test of whether any of it is a property of these four in particular.

## Sequencing

Move 3 is running and needs nothing. Move 1 is the highest-value thing a human
can do this week and needs no compute at all. Move 2 is where the research
headroom is and is CPU-bound against states already on disk.

Do not rent a box for anything that is a fit, a sweep, a probe or a retrieval.
Rent only to encode, and upload states already held rather than re-encoding
them: skipping gemma4 on the current run saved a quarter of the job.

## Operational lessons worth keeping

- **Boxes die without warning.** Three vast.ai instances became unreachable on
  2026-08-29 alone. Anything that exists only on a rented machine is ephemeral;
  pull states down or push them to the Hub the moment they land.
- **Read your own working config before debugging a new one.** Four failed Space
  builds came from starting on gradio 4.44 when every other SRT Space runs 6.x.
  The fix was one `cat` of a sibling `requirements.txt`.
- **Pull the real logs on the first failure.** Three of those four rounds were
  spent fixing a problem that no longer existed, read off a cached error page.
- **Gate cards, not just drafts.** Two wrong numbers reached public cards before
  `scripts/verify_cards.py` existed. It checks every figure in a card against
  its artifacts and found a third error immediately.
- **Superseded numbers outlive their run.** Both bad card figures came from a
  35k pilot that a 112k run had replaced. Delete or clearly mark pilot artifacts
  once the full run lands.

## Superseded

The 2026-07-08 version of this file, with the substrate-invariance addenda and
the pre-medical ranking, is in git history at `fe0ae4e9~1`. The invariance arc
it describes still stands; it is no longer the newest thing on the board.
