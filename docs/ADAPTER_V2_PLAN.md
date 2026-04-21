# SRT-Adapter v2 Plan

> Post-training roadmap for srt-adapter. Items discovered during v1 training (run 4, April 2026).  
> Adapter: 12.7M params on frozen Qwen 2.5-7B. Training data: 1M Reddit samples, 35 domains.

---

## 1. Community Translation Pipeline

**Goal**: Enable semiotic translation — show what the same statement means to every community, translate between communities, and generate community-neutral restatements.

### 1a. `forced_community` in `forward()`

The adapter's `forward()` currently auto-discovers community via `CommunityDiscoveryHead`. Add an optional override:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    forced_community: torch.Tensor | None = None,  # (B, d_community) or None
) -> SRTAdapterOutput:
```

When `forced_community` is provided, skip `self.community_head()` and use the provided vector directly as `community_vec` for all MAH heads. Still run `CommunityDiscoveryHead` for its output in the return value (useful for comparison), but don't use it to condition generation.

**Scope**: ~10 lines changed in `adapter.py`.

### 1b. `generate()` method

Autoregressive generation with community conditioning:

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    forced_community: torch.Tensor | None = None,
    use_semiotic_modulation: bool = False,  # apply BEN logit modulation
) -> GenerateOutput:
```

Requirements:
- KV cache threading through Qwen backbone (use HF's `past_key_values`)
- Thread `forced_community` through each autoregressive step (held constant)
- Thread RRM `meta_state` across steps (GRU hidden state carries forward)
- Optionally apply BEN modulation vector to logits: `logits += λ * m_t`
- Return both generated tokens and per-step semiotic outputs (r̂, regime, divergences)

**Scope**: New method, ~80–120 lines. No retraining required.

### 1c. Community Registry

Map human-readable names to prototype indices. After training:

1. Run all 35-domain val data through adapter, collect `community_output.weights` per domain
2. Cluster: which prototypes activate for which Reddit domains?
3. Produce a `community_registry.json`:

```json
{
  "progressive": {"prototype_idx": [3, 17], "domains": ["politics", "progressive", ...]},
  "conservative": {"prototype_idx": [7, 22], "domains": ["conservative", ...]},
  "technical": {"prototype_idx": [11], "domains": ["science", "programming", ...]},
  "neutral": {"prototype_idx": "centroid", "vector": "mean of all 32 prototypes"}
}
```

**Scope**: Post-training analysis script + JSON artifact.

### 1d. Translation API

```python
class SemioticTranslator:
    def analyze(self, text: str) -> AnalysisResult:
        """Run text through all K community lenses. Returns per-community r̂, divergence maps."""

    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        """Generate equivalent meaning in target community frame."""

    def neutralize(self, text: str) -> TranslationResult:
        """Restate using centroid community vector (maximum shared meaning)."""

    def divergence_matrix(self, text: str) -> DivergenceMatrix:
        """Per-token divergence across all community pairs."""
```

**Scope**: New module `srt/translate.py`, ~200 lines. Wraps adapter + generate.

---

## 2. Training Hyperparameter Adjustments

### 2a. `inject_reg_weight`: 0.01 → 0.1

**Problem**: Injection norms stabilized at 9–15 throughout v1 training. Target was < 1.0. The regularization weight (0.01) is too low to constrain them.

**Evidence**: `inj_norms` logged consistently at [6–16] range across 25K+ steps. No degradation in CE (adapter still functional), but injections are 10× larger than intended. The RRM is compensating by keeping gate values low, but this wastes representational capacity.

**Fix**: Increase `inject_reg_weight` from 0.01 to 0.1 in `LossConfig`. Monitor injection norms — should converge to 0.5–2.0 range within first 2K steps.

**Risk**: If too aggressive, RRM injections get suppressed entirely and semiotic modules starve. Fallback: 0.05.

### 2b. Consider `community_entropy_weight` increase

Current 0.01 may be too low to force the CommunityHead to use all 32 prototypes. Risk of prototype collapse (most mass on 2–3 prototypes). Post-training, check:

```python
# If most prototypes have near-zero activation, increase weight
weights = community_output.weights  # (B, 32)
active = (weights.mean(0) > 0.01).sum()  # how many prototypes are actually used?
```

If `active < 10`, increase `community_entropy_weight` to 0.05–0.1 for v2.

### 2c. Learning rate schedule

V1 uses cosine decay from 3e-4 with 500 warmup steps. At 25K steps, lr is 2.87e-4 — barely decayed because the cosine schedule spans 187,500 steps. The effective learning rate is nearly constant for the entire first epoch.

**Consider**: Shorter cosine period (e.g., per-epoch restart) or lower base lr (1e-4) with warmup to 3e-4. This depends on whether val loss plateaus in later epochs — monitor through completion.

---

## 3. Inference Infrastructure

### 3a. FastAPI Server

Minimal REST endpoint for semiotic analysis:

```
POST /analyze     → {text} → {r_hat, regime, fork_tokens, community_weights}
POST /translate   → {text, source, target} → {translated_text, divergence_map}
POST /neutralize  → {text} → {neutral_text, original_regime, neutral_regime}
POST /matrix      → {text} → {per_community: [{name, r_hat, regime, fork_tokens}]}
```

Single A6000 or consumer 24GB GPU (4090). Qwen 2.5-7B in bf16 ≈ 15GB VRAM. Adapter adds ~25MB. Leaves headroom for batching.

### 3b. Visualization

- **Token heatmap**: Color-coded tokens by r̂ value (blue=subcritical → red=supercritical)
- **Community radar**: Spider chart of community weights for input text
- **Divergence depth plot**: 3 curves (MAH layers 7/14/21) showing divergence magnitude across token positions
- **Translation diff**: Side-by-side source/target community readings with highlighted divergence tokens

Streamlit or Gradio demo for initial release.

---

## 4. Evaluation Framework

### 4a. Baseline CE Measurement

Script exists (`scripts/baseline_ce.py`) but hasn't been run. Must run post-training to establish:
- Bare Qwen CE on val data (no adapter)
- Adapter CE on val data
- Delta should be < 0.1 (adapter shouldn't degrade language modeling)

### 4b. Semiotic-Specific Evaluations

No existing benchmark measures what SRT does. Build task-specific evals:

| Evaluation | Method | Success Criterion |
|---|---|---|
| **Community discrimination** | Feed text from known domains, check if community_output.weights peak at correct prototype cluster | Top-3 accuracy ≥ 60% across 35 domains |
| **r̂ sensitivity** | Compare r̂ on matched controversial vs. neutral texts from same domain | Mean r̂(controversial) > mean r̂(neutral) with p < 0.01 |
| **Regime classification** | Ground-truth regime labels from curated test set | Accuracy ≥ 70% on 3-class task |
| **Chain coherence** | Verify MAH divergence vectors predict across layers | Chain loss < 0.05 on held-out data |
| **Translation preservation** | Human eval: does translated text preserve core propositional content? | ≥ 80% agreement (3 raters) |
| **Neutralization effect** | r̂ of neutralized text vs original | Mean r̂ reduction ≥ 30% |

### 4c. Comparison Baselines

- Qwen 2.5-7B (bare) — no semiotic outputs, CE only
- Qwen 2.5-7B + LoRA fine-tune on same data — same parameter budget, no semiotic architecture
- Perspective API / HateBERT — toxicity scores on same texts (different axis but useful contrast)

---

## 5. Bug Fixes & Cleanup

### 5a. Commit `save_adapter` fix

The `_ADAPTER_PREFIXES` fix in `adapter.py` and `train.py` (saves 25MB instead of 15GB) is deployed on remote but **not committed/pushed**.

```bash
cd /Users/burtron/development/srt-adapter
git add -A
git commit -m "fix: save_adapter filters by explicit prefix tuple (15GB→25MB)"
git push
```

### 5b. Fix ARCHITECTURE.md typo

In `/docs/ARCHITECTURE.md`: "Reflexive Recursion Module" → "Reflexive Recurrent Module"

### 5c. Post-training checkpoint management

Current training saves:
- `best_adapter.pt` (~25MB) — best val_total weights
- `training_checkpoint.pt` (~25MB adapter + optimizer state) — resume point

After training completes:
- Copy `best_adapter.pt` to local
- Run all evaluations (§4)
- Tag release: `git tag v0.1.0-adapter-v1`
- Upload adapter weights to HuggingFace Hub (25MB, trivial)

---

## 6. Data Pipeline Improvements (v2 Training)

### 6a. Curated controversial examples

V1 trains on 1M Reddit samples. The r_true distribution is heavily subcritical (~99% near zero). The adapter learns the landscape but has limited exposure to genuinely supercritical content.

For v2:
- Curate 10K–50K examples of genuinely contested discourse (political debates, culture war threads, policy disagreements)
- Annotate with human-verified r_true values
- Oversample 5–10× in training mix (same technique as SRT-v1 stage 2 curated oversampling)

### 6b. Multi-turn conversations

V1 data is single-turn (one Reddit comment). Real discourse unfolds across turns. For v2:
- Extract Reddit comment *threads* (parent → child chains)
- Track how r̂ evolves across a conversation
- Train the RRM to predict divergence trajectories, not just snapshots

### 6c. Cross-platform data

Reddit has its own community structure. Twitter/X, news comments, and legislative transcripts have different bifurcation dynamics. Mixing platforms would make the community prototypes more general.

---

## Priority Order

| Priority | Item | Depends On | Effort |
|---|---|---|---|
| **P0** | 5a. Commit save fix | Nothing | 5 min |
| **P0** | 5b. ARCHITECTURE.md typo | Nothing | 1 min |
| **P0** | 4a. Baseline CE measurement | Training completion | 10 min |
| **P1** | 1a. `forced_community` in forward | Training completion | 30 min |
| **P1** | 1b. `generate()` method | 1a | 2–4 hours |
| **P1** | 2a. inject_reg_weight 0.01→0.1 | V2 training run | Config change |
| **P2** | 1c. Community registry | Training completion + analysis | 2 hours |
| **P2** | 4b. Semiotic evaluations | 1a, 1b | 1–2 days |
| **P2** | 1d. Translation API | 1b, 1c | 4–6 hours |
| **P2** | 2b. Community entropy weight | Post-training analysis | Config change |
| **P3** | 3a. FastAPI server | 1d | 4 hours |
| **P3** | 3b. Visualization | 3a | 1 day |
| **P3** | 4c. Comparison baselines | 4b | 1–2 days |
| **P4** | 6a. Curated controversial data | V2 cycle | 1 week |
| **P4** | 6b. Multi-turn conversations | V2 cycle | 1 week |
| **P4** | 6c. Cross-platform data | V2 cycle | 2 weeks |

---

## Current Training Status

- **Run 4** (v1): Step ~25,700 of 187,500 (~13.7% complete)
- **Val total**: 11.139 (12 consecutive improvements, monotonic)
- **CE**: Stable ~2.5–3.0 (no degradation)
- **Chain loss**: 0.01 (essentially solved)
- **r̂ mean**: Drifting 0.84 → 0.68 (correct calibration toward subcritical Reddit data)
- **Injection norms**: 9–15 (too high, motivates §2a)
- **Community entropy**: Not yet logged (motivates §2b investigation)
- **ETA**: ~4 days remaining for 3 full epochs
