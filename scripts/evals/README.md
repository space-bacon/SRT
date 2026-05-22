# HF leaderboard submission scaffolding

Currently wired:

| script | leaderboard / dataset | what it produces |
|---|---|---|
| `halueval_nla.py` | **HaluEval** (`pminervini/HaluEval`) — hallucination detection across QA / dialogue / summarization splits | per-example feature CSV + 5-fold ROC-AUC json |

## Local smoke test

```bash
pip install datasets scikit-learn
python scripts/evals/halueval_nla.py \
    --backbone gemma-2-2b \
    --split qa \
    --n 50 \
    --out artifacts/halueval/qa_smoke.csv
```

Smoke target on Gemma-2-2B: AUC ≳ 0.60 with just `(pre_fve, jaccard, drift)`
on the QA split is the "this works" bar. The published HHEM detector is at
~0.78 on its own training distribution, but trained on >100K labelled
documents; we're using **zero** labelled hallucination data — the AV
weights are frozen.

## Full run on the A6000

```bash
# vast.ai: ssh -p 30761 root@209.137.198.14
cd /root/srt-adapter
git pull
pip install datasets scikit-learn
python scripts/evals/halueval_nla.py --backbone qwen2.5-7b --split qa            --n 1000 --out artifacts/halueval/qa_qwen.csv
python scripts/evals/halueval_nla.py --backbone qwen2.5-7b --split dialogue      --n 1000 --out artifacts/halueval/dialogue_qwen.csv
python scripts/evals/halueval_nla.py --backbone qwen2.5-7b --split summarization --n  500 --out artifacts/halueval/summ_qwen.csv
```

Expect ~6 hrs total on a single A6000 (≈20s/row at 1000 rows × 2 sides × 3 splits).

## Reporting

After the run, each CSV gets a sibling `*.metrics.json` with `auc_cv_mean`.
For a leaderboard submission we need three artifacts:

1. **`predictions.jsonl`** — one row per HaluEval example, `{id, score, label}`.
2. **`model_card.md`** — describes the detector as "frozen Qwen2.5-7B + NLA
   adapter, logistic regression on 3 hidden-state-derived features".
3. **`reproduce.sh`** — the commands above.

The Vectara HHEM leaderboard isn't open-submission — it's a curated comparison.
For an open-submission analogue, the path is:

- **HaluEval public leaderboard** (Tsinghua) — accepts external entries via PR
  to `RUCAIBox/HaluEval`.
- **Open LLM Leaderboard v2** TruthfulQA component — submit a *generation* model
  (Qwen + NLA-steer-honest) via the standard `submit your model` form.

The HaluEval entry comes first because the feature CSV → predictions.jsonl
transform is one line of pandas.
