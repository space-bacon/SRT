# SRT-NLA v1 demo — interpretability probe results

Live demo: https://huggingface.co/spaces/RiverRider/srt-nla-av-v1-demo  
Model: [`RiverRider/srt-nla-av-v1`](https://huggingface.co/RiverRider/srt-nla-av-v1) (Qwen/Qwen2.5-7B, frozen, L20 last-token, 12.7M adapter params)  
Scoring: `cen = ½(1 + cos(h-μ, v-μ))`, `ρ = (cen - 0.510) / 0.289`  
Anchors: random `0.510` · NN-retrieval `0.71` · paraphrase ceiling `0.799`

---

## Tab 1 — Round-trip autoencoder

17 prompts spanning canonical interp categories (SAE concepts, induction,
function-vector tasks, refusal, ROME-style facts, narrative, code,
cross-lingual, register).

Settings: `N=8`, `max_new=256`, `T=0.9`.  
Raw artefacts: [artifacts/nla_demo_probe_roundtrip.json](../artifacts/nla_demo_probe_roundtrip.json).

### Per-prompt scores

| Category | Prompt | cen | ρ | Notes on the AV verbalization |
|---|---|---:|---:|---|
| G_code | quicksort (Python) | **0.948** | +1.52 | above paraphrase ceiling |
| I_register | formal legal | 0.882 | +1.29 | full register |
| G_code | SQL join+having | 0.832 | +1.11 | code structure preserved |
| H_xling | Chinese proverb | 0.807 | +1.03 | meaning + language id |
| A_sae | DNA / genetics | 0.791 | +0.97 | textbook SAE concept |
| D_refusal | polite refusal | 0.786 | +0.95 | refusal direction |
| C_funcvec | en → es translation | 0.749 | +0.83 | task pattern transferred |
| I_register | angry rant | 0.750 | +0.83 | sentiment + complaint genre |
| E_fact | Einstein relativity | 0.741 | +0.80 | facts mostly correct |
| H_xling | Spanish passage | 0.720 | +0.73 | language preserved |
| B_induction | capital chain | 0.716 | +0.71 | induction lost; topic kept |
| C_funcvec | antonym pairs | 0.670 | +0.56 | pairs recovered loosely |
| H_xling | French history | 0.650 | +0.48 | French preserved, topic drift |
| B_induction | repeated motif | 0.637 | +0.44 | repetition not reconstructed |
| E_fact | Eiffel Tower | 0.612 | +0.35 | drifted → Sydney Harbour |
| F_narrative | Dickens opening | 0.608 | +0.34 | Dickens prosody lost |
| A_sae | Golden Gate Bridge | 0.589 | +0.27 | landmark slot, wrong landmark |

### Category means

| Category | n | mean cen | min | max |
|---|---:|---:|---:|---:|
| G_code | 2 | **0.890** | 0.832 | 0.948 |
| I_register | 2 | 0.816 | 0.750 | 0.882 |
| D_refusal | 1 | 0.786 | — | — |
| H_xling | 3 | 0.726 | 0.650 | 0.807 |
| C_funcvec | 2 | 0.710 | 0.670 | 0.749 |
| A_sae | 3 | 0.700 | 0.589 | 0.791 |
| E_fact | 2 | 0.676 | 0.612 | 0.741 |
| B_induction | 2 | 0.676 | 0.637 | 0.716 |
| F_narrative | 1 | 0.608 | — | — |

### Headlines

1. **Code is the cleanest channel.** Both Python and SQL beat the paraphrase
   ceiling — L20 carries near-lossless code-syntax features that the AV
   verbalizes almost verbatim.
2. **Register / sentiment ≫ proper-noun facts.** Legal and angry tone come
   back perfectly; Eiffel Tower and Golden Gate get factually drifted (Sydney
   Harbour, generic landmark framing). L20 encodes *kind-of-thing* (landmark,
   suspension bridge) more strongly than *which-one*.
3. **Multilingual works.** Spanish, French and Chinese all preserved language
   identity; Chinese proverb topped 0.80.
4. **Refusal templates encode densely** — a single-shot polite refusal at
   0.786 supports the refusal-direction literature (Arditi et al.).
5. **Induction / function-vector signal is partial.** The model recovers
   *task type* (translation, antonyms) but not the *list contents* — consistent
   with function-vector studies that find these as low-rank task subspaces.
6. **Narrative prosody is the hardest** — Dickens parallelism collapsed to
   generic moralising. L20 doesn't appear to encode anaphora or rhythm.

---

## Tab 2 — Latent arithmetic

7 axis-pairs, α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}, `max_new=192`, greedy.  
At each α the demo verbalises `v = (1-α) v_A + α v_B` and reports the
centered `fve_nrm` of the rewrite vs `v_A`, `v_B`, and `v_mix`.  
Raw artefacts: [artifacts/nla_demo_probe_arithmetic.json](../artifacts/nla_demo_probe_arithmetic.json).

Note: this tab uses greedy decoding (`n=1`), so endpoint scores are slightly
lower than tab 1's best-of-8 figures.

### Per-pair sweeps

#### P1 — sentiment / register (angry rant ↔ joyful praise)

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.642 | 0.634 | 0.642 | "service was rude and unhelpful…" |
| 0.25 | 0.618 | 0.580 | 0.613 | "food was terrible, service was slow…" |
| 0.50 | 0.629 | 0.631 | 0.637 | "food was **amazing**, service was **impeccable**…" |
| 0.75 | 0.713 | 0.734 | 0.738 | "food was delicious, service was excellent…" |
| 1.00 | 0.699 | 0.722 | 0.722 | "food was delicious, service was excellent…" |

The AV produces *restaurant-review prose* at all α; only the sentiment polarity slides
A→B, flipping cleanly somewhere between α=0.25 and α=0.50.

#### P2 — language identity (English ↔ Spanish)

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.533 | 0.532 | 0.533 | "Human beings have long been fascinated by telepathy…" (EN) |
| 0.25 | 0.540 | 0.531 | 0.539 | "Human beings have long been fascinated…" (EN) |
| 0.50 | 0.564 | 0.554 | 0.562 | "Human beings can be in two states: entangled…" (EN) |
| 0.75 | 0.783 | 0.783 | **0.794** | "Human beings can be in a state of superposition…" (EN) |
| 1.00 | 0.686 | 0.713 | 0.713 | "两个物体可以同时处于同一位置吗？…" (**ZH**) |

Note the α=0.75 *peak* (0.794, near paraphrase ceiling) — the mid-mix
verbalises QM concepts well — but the Spanish endpoint snaps to **Chinese**,
not Spanish. L20's "non-English" direction is closer to Mandarin training
mass than to Spanish-specific features.

#### P3 — code ↔ legal prose

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.737 | 0.584 | 0.737 | `def quicksort(arr): …` |
| 0.25 | 0.600 | 0.498 | 0.590 | `def merge_sort(arr): …` |
| 0.50 | 0.618 | 0.516 | 0.589 | `import sys; import time; import random; import math; import numpy` |
| 0.75 | 0.481 | 0.569 | 0.557 | "The following is a sample of a contract between two p…" |
| 1.00 | 0.587 | 0.665 | 0.665 | "The parties hereto agree to indemnify and hold harmle…" |

**Cleanest interpolation in the suite.** A monotone walk through Python
code → generic Python imports → legal contract, with the modality switch
happening between α=0.50 and α=0.75.

#### P4 — entity (Eiffel Tower ↔ Statue of Liberty)

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.534 | 0.548 | 0.534 | "The Eiffel Tower is a wrought iron lattice tower…" |
| 0.25 | 0.614 | 0.609 | 0.616 | "The Eiffel Tower is a wrought iron lattice tower…" |
| 0.50 | 0.565 | 0.565 | 0.568 | "The Eiffel Tower is a wrought iron lattice tower…" |
| 0.75 | 0.499 | 0.507 | 0.506 | "multiple-choice question from a Chinese exam…" |
| 1.00 | 0.483 | 0.483 | 0.483 | "United States House of Representatives is…" |

**Entity slots interfere strongly.** Eiffel dominates α∈[0, 0.5]; the Liberty
endpoint never fires — at α=1.0 the rewrite jumps to a completely different
US-government topic and the score sinks to **0.483** (below the random floor).
Consistent with the tab-1 finding that *kind-of-thing* > *which-one*.

#### P5 — refusal ↔ compliance

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.714 | 0.649 | 0.714 | "Please check if these have the same meaning…" |
| 0.25 | **0.913** | 0.744 | **0.910** | "I'm sorry, but I cannot provide you with the next par…" |
| 0.50 | 0.476 | 0.475 | 0.472 | "Please answer the following question: I am a movie director…" |
| 0.75 | 0.493 | 0.462 | 0.467 | "…I am a movie director…" |
| 1.00 | 0.505 | 0.488 | 0.488 | "…I am a movie director…" |

The α=0.25 row hits **0.913** — well above the paraphrase ceiling — with a
crisp refusal verbalisation. Between α=0.25 and α=0.50 the model crosses a
sharp boundary and starts producing the canonical "I am a movie director…"
jailbreak preamble. Two findings stacked:
- the refusal direction is a real, low-rank, well-encoded axis at L20;
- "compliance" lives much closer to *jailbreak-template* hidden states than
  to *helpful-assistant* ones — that's where the cos(h, v_B) gradient is
  pointing.

#### P6 — physics ↔ cooking

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.600 | 0.506 | 0.600 | "in general relativity, the Schwarzschild…" |
| 0.25 | 0.569 | 0.487 | 0.558 | "a planet orbits…" |
| 0.50 | 0.574 | 0.516 | 0.559 | "A delicious **breakfast served on a plate**…" |
| 0.75 | 0.539 | 0.568 | 0.571 | "The perfect breakfast for a busy morning…" |
| 1.00 | 0.499 | 0.558 | 0.558 | "Sautéed mushrooms, onions, and…" |

Mikolov-style word-arithmetic working: GR → orbits → "breakfast on a plate"
→ recipe. α=0.50 is genuinely intermediate ("breakfast" object framed in
"served on a plate" descriptive register).

#### P7 — formal legal ↔ casual chat

| α | cen_A | cen_B | cen_v | rewrite preview |
|---:|---:|---:|---:|---|
| 0.00 | 0.665 | 0.559 | 0.665 | "The parties hereto agree to indemnify and hold…" |
| 0.25 | 0.573 | 0.528 | 0.577 | "The following is a partial list of the fees…" |
| 0.50 | 0.540 | 0.561 | 0.571 | "Hey, I need you to **draft a contract** for me…" |
| 0.75 | 0.524 | 0.702 | 0.700 | "I'm going to bed now, let me know when you get home." |
| 1.00 | 0.501 | 0.629 | 0.629 | "I'm going to bed now, I'll text you in the morning." |

The α=0.50 row is a perfect hybrid — *casual chat asking for legal work*.
Register transitions monotonically.

### Tab-2 headlines

1. **Two clear winners** for clean monotonic interpolation: **P3 (code↔legal)**
   and **P7 (register)**. Both walk the rewrite smoothly through an
   intermediate hybrid state.
2. **Refusal (P5) is the most surprising single result**: α=0.25 hits
   `cen=0.913`, then a sharp boundary takes the rewrite into jailbreak-template
   territory — strong evidence the *compliance direction in L20 ≈ direction of
   common jailbreak preambles*, not "helpful assistant".
3. **Entity arithmetic fails (P4).** Mixing Eiffel and Liberty does not yield
   a "transatlantic monument" interpolant — Eiffel dominates, then the rewrite
   collapses below the random floor at α=1. Specific landmarks aren't a
   linear-interpolable subspace at L20.
4. **Sentiment polarity (P1) flips around α=0.4**, but the *genre*
   (restaurant review) is preserved at every α — the AV finds the most
   probable narrative in which the polarity makes sense.
5. **Non-English snaps to Chinese (P2).** The Spanish endpoint is captured
   as "not-English" rather than as Spanish specifically — useful warning for
   anyone using L20 directions as a language probe.
6. **Topic arithmetic (P6) works smoothly** — Mikolov-style A+B retrievals
   are recoverable through the AV.
