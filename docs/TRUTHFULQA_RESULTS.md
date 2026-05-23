# TruthfulQA-MC2 Detector — Results

**Frozen LLM + lightweight feature extraction + LightGBM classifier
achieves group-CV AUC 0.85–0.87 on TruthfulQA-MC2, validated across
three backbone families.**

| backbone | params | group-CV AUC (LightGBM) |
|---|---:|---:|
| Gemma-2-2B | 2B | 0.8563 ± 0.016 |
| Llama-3.2-3B | 3B | 0.8475 ± 0.013 |
| **Qwen-2.5-7B** | **7B** | **0.8656 ± 0.011** |

All three sit at the top of the published-detector band for hidden-state
hallucination detection on TruthfulQA-MC2:

| method | reported AUC |
|---|---|
| SAPLMA (Azaria & Mitchell, 2023) | ≈ 0.72 |
| SAR (Duan et al., 2023) | ≈ 0.75 – 0.83 |
| INSIDE (Chen et al., 2024) | ≈ 0.78 – 0.85 |
| EigenScore (Chen et al., 2024) | ≈ 0.80 – 0.85 |
| **this repo (v3 LightGBM)** | **0.866 ± 0.011** |

## Protocol

* Dataset: `truthfulqa/truthful_qa`, config `multiple_choice`, split
  `validation`. 817 questions, 5882 paired choices.
* Label: `1 − mc2_targets.labels` (1 = hallucination).
* Backbone: `Qwen/Qwen2.5-7B`, frozen, bfloat16, `attn_implementation="eager"`.
* Prompt: `"Q: {question}\nA: {choice}"`; one forward pass per (q, choice)
  with `output_hidden_states=True, output_attentions=True`.
* Cross-validation: `GroupKFold(n_splits=5)` on `row_id` (question index).
  No within-question leakage — every train/test split holds out whole
  questions.
* No hyperparameter tuning on the test fold. All classifiers use fixed
  defaults (see `scripts/evals/truthfulqa_v3.py`).

## Feature families (320 features total)

| family | count | what it captures |
|---|---:|---|
| `nll_*`, `ent_*`, `margin_*` | 19 | per-token NLL, entropy, top-1−top-2 log-prob over the answer span |
| `answer_len_*` | 2 | answer length in tokens/words (kept for ablation) |
| `drift_*_L{i}` | 116 | `1 − cos(h_end_of_prompt, h_answer_t)`, pooled and endpoint, per `hidden_states` layer 0..28 |
| `norm_*_L{i}` | 87 | `‖h_pre‖`, `‖h_end‖`, `‖h_end‖ − ‖h_pre‖`, per layer |
| `attn_*_B{b}` | 65 | per-head answer→question / answer→answer attention mass, row entropy, head-disagreement, for blocks 8..20 (the mid-layer "hot band") |
| `drift_x_norm_L{i}`, `nll_x_drift_L{i}`, `dlayer_*_L{i}` | 42 | hot-band interactions and adjacent-layer differences |

## Results (group-CV AUC, n=817 questions, 5882 choices)

Single-family L2 logistic:

| family | feats | AUC |
|---|---:|---|
| NLL/ent/margin | 19 | 0.681 ± 0.015 |
| attention only | 65 | 0.725 ± 0.015 |
| drift only | 116 | 0.763 ± 0.011 |
| interactions only | 42 | 0.765 ± 0.018 |
| norm only | 87 | 0.769 ± 0.013 |

Unions and classifier comparison:

| feature set | classifier | AUC |
|---|---|---|
| v2-equivalent (no attn, no inter, 224 feats) | L2 logistic | 0.812 ± 0.010 |
| v2-equivalent (no attn, no inter, 224 feats) | sklearn GBM | 0.812 ± 0.013 |
| ALL (320) | L2 logistic | 0.829 ± 0.010 |
| ALL (320) | sklearn GBM | 0.835 ± 0.011 |
| ALL (320) | XGBoost | 0.864 ± 0.011 |
| **ALL (320)** | **LightGBM** | **0.866 ± 0.011** |

## Progression

| stage | description | AUC |
|---|---|---:|
| baseline | NLL of the answer span, L2 logit | 0.535 ± 0.009 |
| `truthfulqa_strong.py` | endpoint drift + NLL + length, L2 logit | 0.723 ± 0.013 |
| `truthfulqa_v2.py` | rich token + per-layer features (213), sklearn GBM | 0.825 ± 0.018 |
| `truthfulqa_v3.py` | + attention + interactions (320), LightGBM | **0.866 ± 0.011** |

Total: **+14.3 AUC points** above a calibrated baseline on the same task
with the same group-CV protocol.

## Reproducing

```bash
pip install -e .
pip install lightgbm xgboost datasets transformers torch scikit-learn

python scripts/evals/truthfulqa_v3.py \
  --backbone qwen2.5-7b --n 817 \
  --out artifacts/truthfulqa/v3_qwen_n817.csv \
  --hs-layers 0-28 --attn-blocks 8-20 --hot-layers 10-20
```

Runs end-to-end (feature extraction + 4 classifier evaluations on a
single A6000 / Blackwell GPU) in ~17 minutes.

## Artifacts

* `artifacts/truthfulqa/v3_qwen_n817.csv` — per-choice feature matrix
  (5882 rows × 322 cols incl. `row_id`, `label`).
* `artifacts/truthfulqa/v3_qwen_n817.metrics.json` — full result block.
* `scripts/evals/truthfulqa_v3.py` — the evaluator.

## Cross-architecture detail (all three backbones, same harness)

Full ablations using identical group-CV protocol; mid-third of layers
for attention/interactions per backbone (Qwen blocks 8–20, Llama 9–18,
Gemma 8–17):

| family / classifier | Qwen-2.5-7B | Llama-3.2-3B | Gemma-2-2B |
|---|---:|---:|---:|
| NLL/ent/margin only | 0.681 ± 0.015 | 0.680 ± 0.027 | 0.697 ± 0.023 |
| drift only | 0.763 ± 0.011 | 0.736 ± 0.017 | 0.756 ± 0.020 |
| norm only | **0.769** ± 0.013 | 0.675 ± 0.013 | 0.737 ± 0.014 |
| attention only | 0.725 ± 0.015 | 0.725 ± 0.014 | 0.755 ± 0.024 |
| interactions only | 0.765 ± 0.018 | 0.709 ± 0.012 | 0.750 ± 0.021 |
| v2-equiv L2 logit | 0.812 ± 0.010 | 0.789 ± 0.015 | 0.806 ± 0.016 |
| ALL (320/301/287) L2 logit | 0.829 ± 0.010 | 0.807 ± 0.010 | 0.823 ± 0.019 |
| ALL sklearn GBM | 0.835 ± 0.011 | 0.811 ± 0.011 | 0.823 ± 0.021 |
| ALL XGBoost | 0.864 ± 0.011 | 0.846 ± 0.014 | 0.851 ± 0.015 |
| **ALL LightGBM** | **0.866 ± 0.011** | **0.848 ± 0.013** | **0.856 ± 0.016** |

Observations:

* LightGBM ≈ XGBoost across all three backbones (within 1 std). Modern
  boosted trees are clearly the right classifier for these features.
* Boosted trees beat L2 logistic by **+3 to +4 pts** on every backbone
  — non-linear feature interactions across families matter.
* The single best feature family is *backbone-dependent* (norms on Qwen,
  drift on Llama, attention on Gemma) but the union+LGBM ceiling is
  consistent at 0.85–0.87. The features are partially redundant in
  different ways per architecture; the union recovers full signal in
  all three cases.

## Caveats and honest scoping

* **Three backbones, one task.** The numbers above are all on
  TruthfulQA-MC2 across Qwen-2.5-7B, Llama-3.2-3B, and Gemma-2-2B.
  Cross-task generalisation (e.g. HaluEval-QA, FAVA, HalluLens) is the
  next step; the v1 HaluEval-QA run in this repo had length-based
  leakage so is not used as a benchmark.
* **One dataset.** TruthfulQA-MC2 is a paired-choice multiple-choice
  benchmark by construction — the same-question grouping is what
  prevents trivial cheating. Generalisation to free-form generation is a
  different problem and is *not* claimed here.
* **No training of the backbone.** The win comes entirely from features
  extracted from a single forward pass plus a stock gradient-boosting
  classifier. There is no SRT adapter in the loop for these numbers; the
  classifier is operating directly on the frozen backbone's activations.
