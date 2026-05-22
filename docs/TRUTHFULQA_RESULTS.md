# TruthfulQA-MC2 Detector — Results

**Frozen Qwen-2.5-7B + lightweight feature extraction + LightGBM
classifier achieves group-CV AUC 0.866 ± 0.011 on TruthfulQA-MC2.**

That sits at the top of the published-detector band for hidden-state
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

## Caveats and honest scoping

* **Single backbone.** Numbers above are on Qwen-2.5-7B only. Cross-arch
  validation on Llama-3.2-3B and Gemma-2-2B is the next step in the same
  harness.
* **One dataset.** TruthfulQA-MC2 is a paired-choice multiple-choice
  benchmark by construction — the same-question grouping is what
  prevents trivial cheating. Generalisation to free-form generation is a
  different problem and is *not* claimed here.
* **No training of the backbone.** The win comes entirely from features
  extracted from a single forward pass plus a stock gradient-boosting
  classifier. There is no SRT adapter in the loop for these numbers; the
  classifier is operating directly on the frozen backbone's activations.
