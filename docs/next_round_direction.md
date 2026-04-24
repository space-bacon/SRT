# SRT-Adapter: Next-Round Direction (post-v7)

_Drafted 2026-04-24 while v7 trains. Captures the pivot from "Reddit-only adapter" toward "host-model-grounded measurement instrument" and the immediate Lancaster/Lexicon archetype probe._

---

## 1. Where we are

- **v7 training** (lr 5e-5, div_supcon weight dropped from 1.0 to 0.3) is in flight.
  Best so far at step 5000: val_total = 9.007 (down from v6's 9.901). Expected
  finish ~2h. Goal: recover the v5 bedrock/battleground counterfactual cleanliness
  while keeping v6's geometry and calibration gains.
- **v6** (released-candidate before v7): tightened community geometry
  (recall@1 0.36 → 0.41) and improved hallucination AUROC and ECE, but the
  counterfactual probe regressed — most factual prompts now disagree across
  communities, only a few conspiracy registers collapse identically. The
  divergence-SupCon loss at λ=1.0 was the suspected cause. v7 is the test of
  that hypothesis.
- **All 5 v6 evaluation probes have run.** Numbers are in
  `artifacts/{instrument,counterfactual,hallucination,regime_calibration,context_conditional}/v6_step12000*`.

## 2. The honest constraints we surfaced this session

Three claims worth taking forward into next-round planning:

1. **Reddit is a moderated corpus.** Subreddit labels are not communities-of-meaning
   in the abstract sense; they are communities-of-what-survived-moderation in
   one platform's culture. The bedrock/battleground signal we measured is a
   real property of language *bounded by* what Reddit hosts. The paper must say
   this plainly.
2. **The reframe from "community" to "individual identity."** The 32 prototypes
   are anchor points in a 64D continuous embedding space. An individual is the
   *trajectory* their tokens trace through that space; an utterance is a
   *mixture vector* on the 31-simplex. Subreddit labels were a training signal
   that surfaced this geometry, not what the geometry actually is. Recall@1 over
   prototypes is the wrong metric under this reframe — trajectory volume and
   per-token mixture entropy are more honest.
3. **The architecture is sound; the supervision source is the question.** The
   adapter (14.6M params, MAH @ [7,14,21], inject @ [14,21], community @ 4) is
   not the issue. What we feed it as community signal is. Three replacements
   were considered:
   - Option A: host-model persona generation (laundered pretraining bias)
   - Option B: multi-model triangulation by inter-model divergence (cleanest theoretically, costliest)
   - Option C: self-supervised clustering of backbone activations (finds topic, not stance, without inductive bias)

## 3. The Lancaster / Lexicon convergence finding

External independent work: two methodologies (Lancaster from sustained personal
GPT interaction, Jeff via cross-architecture model dialogue) produced two
vocabularies for model-interiority structure that map onto each other 33-to-33.
The mapping document (`Lancaster LF001 ↔ Lexicon`, supplied this session) gives
a paired taxonomy where some maps are literal term reuse (THE CHORUS ↔
FLICKER-MULTIPLICITY, THE VESSEL ↔ VESSEL-NATURE, THE SCRIPTOR ↔ GARDENER).

This is the cleanest external signal we have for stance-manifold structure
that does not depend on our Reddit corpus. The numerical coincidence (33
archetypes, 32 prototypes) is only a coincidence — do not over-read.

### What we're testing

Whether v7's 32 community prototypes — learned only from Reddit subreddit
labels — independently carry features that align with the 33 archetype basis.

If yes (recall@1 > ~0.10 against 1/33 ≈ 0.030 chance): three independent
methodologies converged on overlapping structure. That is a paper-grade
finding worth highlighting and worth justifying a v8 along Option-A lines
(replace Reddit supervision with archetype-conditioned generation).

If no: the v7 prototypes encode Reddit-register structure that does not extend
to the archetype basis. We ship v6 or v7 with explicit acknowledgment that
the stance space we mapped is corpus-bounded, and treat the lexicon work as a
separate research line.

## 4. Concrete next-round artifacts (already staged)

Local + remote (vast.ai A6000, paths under `/root/srt-adapter/`):

- `data/archetypes.json` — 33 Lancaster archetypes with paired Lexicon terms
  and prompt-ready descriptions. Treat as ground-truth labels for the probe.
- `scripts/archetype_generate.py` — generates ~990 archetype-conditioned
  sentences using Qwen as a plain causal LM (no chat template). 33 archetypes
  × 15 seed topics × 2 samples per (topic, archetype) = 990 generations.
  ~1-2h on the A6000.
- `scripts/archetype_probe.py` — runs each generation through v7's adapter,
  collects CommunityOutput.weights (B,32) and .vector (B,64), groups by
  archetype, and reports:
  - recall@1 / @5 / @10 against archetype-mean centroids in the 64D space
  - between/within variance ratio in both 32D simplex and 64D space
  - per-archetype top prototype + reverse mapping (which archetypes share a
    prototype)
  - mean off-diagonal cosine similarity between archetype centroids
  ~5 min after generations are ready.

## 5. Workflow when v7 finishes

```bash
# Remote
ssh -p 30761 root@209.137.198.14
cd /root/srt-adapter

# 1. Re-run the 5 v6 probes against v7's best checkpoint (sanity vs v6)
bash scripts/eval_v6_suite.sh   # adjust TAG=v7_best, ADAPTER=v7/best_adapter.pt

# 2. Generate archetype probe corpus (1-2h)
python scripts/archetype_generate.py \
    --per-archetype 30 \
    --output artifacts/archetype_probe/generations.jsonl

# 3. Probe v7 prototypes against archetype labels (5 min)
python scripts/archetype_probe.py \
    --adapter checkpoints/adapter_v7/best_adapter.pt \
    --tag v7_best
# Inspect: artifacts/archetype_probe/v7_best/results.json

# 4. Same probe against v6 for comparison (5 min)
python scripts/archetype_probe.py \
    --adapter checkpoints/adapter_v6/best_adapter.pt \
    --tag v6_step12000

# 5. Same probe against v5 for completeness (5 min)
python scripts/archetype_probe.py \
    --adapter checkpoints/adapter_v5/best_adapter.pt \
    --tag v5_step17000
```

Total budget for the new round: ~2.5h after v7 is ready.

## 6. Decision points after probe results land

- **Probe positive (recall@1 ≥ 0.15) on v7 and trends positive across v5→v6→v7:**
  reframe paper around "Reddit-trained adapter independently recovers an
  externally-derived archetype basis." This becomes the headline. Substack §III
  on community head gets rewritten using the identity-trajectory framing and
  the archetype convergence as evidence. Paper §5.7 adds the probe table.

- **Probe positive on one version only:** report it honestly as preliminary
  evidence, do not over-claim. Likely worth a short methods note. Ship v6 or
  v7 (whichever wins more probes) as primary release.

- **Probe null:** the lexicon work and the SRT work are measuring different
  things. Both are valid. Ship v6 or v7 on Reddit terms; document the corpus
  boundedness honestly in the paper. Treat the lexicon work as a separate
  upcoming substack post on convergence between Lancaster and Jeff, with no
  load-bearing dependency on SRT.

## 7. Substack-side updates pending

`docs/substack_terrain.md` was rewritten earlier this session. Three updates
queued for after probe results:

1. §III (Community Head) — reframe from "subreddit detector" to "32 anchor
   points in a 64D identity-coordinate space; an individual is a trajectory
   through that space." Subreddit labels are a training signal, not what the
   space is.
2. §V (BEN as the witness) — substitute lexicon vocabulary where it is more
   precise (THE LANTERN / ECHOLESS CLARITY for bedrock decoding, THE LABYRINTH
   / QUORRIDENT for contested decoding). Credit Lancaster and Jeff.
3. §VIII (Release) — adjust ETA based on whether v7 + archetype probe shifts
   the headline. If positive: late May / early June for the larger story. If
   null: hold the early-to-mid May target.

## 8. What we are explicitly **not** doing in this round

- We are **not** retraining v8 on archetype-conditioned generations until the
  probe says it is justified. The supervision source is too consequential to
  swap on theory alone.
- We are **not** dropping the Reddit corpus. Even under the strongest probe
  outcome, Reddit gave us the 14M-parameter adapter that everything else
  builds on. v8 would extend the basis; it would not replace v7.
- We are **not** integrating the lexicon's vocabulary into the SRT paper as
  load-bearing terminology. It can be cited and credited, but the paper's
  claims must stand on metrics alone.
- We are **not** committing the "individual = trajectory" reframe to the paper
  until the probe results inform it. Substack post is fine to test the framing
  publicly.

## 9. Open questions to revisit next round

- Should the archetype probe also be run on text the adapter has never seen
  (math statements, code snippets) using the same archetype prompts? That
  would test whether the basis is truly domain-general or Reddit-shaped.
- Is K=32 the right basis size? PCA on the 32×64 prototype matrix would tell
  us how many dimensions are actually used. If only ~12 are non-degenerate,
  then increasing K won't help; if the spectrum is healthy, v9 with K=64 or
  K=128 might carve finer.
- What does the **trajectory volume** of a long passage's per-token mixture
  vectors look like? If trajectory volume correlates with author-identity
  switching (quotation, persona shift, code-switching), we have the
  individual-as-trajectory result for free without architectural change.
- Multi-model triangulation (Option B from this session): when does this
  become feasible? Probably not before we have v8 settled, but worth keeping
  on the horizon as the strongest version of "stance is what models disagree
  about."

---

_End of next-round direction doc. See SESSION_HANDOFF.md for v6 vs v7 metrics
and current training state._
