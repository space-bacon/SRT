#!/usr/bin/env bash
# =============================================================================
# NLA on gemma-4-31B-it — greedy-gap test on the vision backbone, then the
# vision transition. Single Blackwell box (>= 96GB GPU), /venv/gemma-style env
# (transformers >= 5.12 for gemma4).
#
# Phases:
#   0. env      — venv check: transformers>=5.12, torch cu13x, srt installed
#   1. smoke    — weights ACTUALLY load (Gemma4ForConditionalGeneration, NOT
#                 CausalLM which silently random-inits), bos!=eos, 61 hidden
#                 states, chat generation coherent
#   2. targets  — sample 10k text targets, seq 64, extraction layer L47
#                 (cross-modal alignment peak — chosen so the SAME AV serves
#                 the image-position verbalization afterward)
#   3. pairs    — gold {target_idx, gold_ids} pairs
#   4. anchors  — replay ceiling / NN baseline / random floor (centered frame)
#   5. av_ce    — plain CE baseline AV (np16). This is the "plain greedy" arm.
#   6. av_draft — draft-conditioned AV (retrieval-then-edit), warm from 5.
#                 THE GREEDY-GAP TEST: compare val greedy_cen vs draft_cen
#                 (copy/NN baseline) vs phase-5 plain greedy.
#   7. vision   — verbalize image-position activations with the phase-6 AV
#                 (scripts/gemma4_vision_verbalize.py) on the stereo images.
#   8. backup   — push checkpoints + artifacts to HF (box is EPHEMERAL).
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   PHASES="0 1 2 3 4 5 6 7 8" bash scripts/train_gemma4_nla_all.sh
# =============================================================================
set -euo pipefail

ROOT="${ROOT:-/workspace/srt-adapter}"
BACKBONE="${BACKBONE:-google/gemma-4-31B-it}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
VENV="${VENV:-/venv/gemma}"
PY="${VENV}/bin/python"
HF_USER="${HF_USER:-RiverRider}"

# gemma-4-31B-it text tower: 60 layers. Cross-modal alignment peaks L47/L54
# (paper §11.6); L47 ≈ 78% depth also sits in the NLA sweet band.
NLA_LAYER="${NLA_LAYER:-47}"
NLA_NUM_SEQ="${NLA_NUM_SEQ:-10000}"
NLA_SEQ_LEN="${NLA_SEQ_LEN:-64}"
# gemma-4-it is chat-tuned: bare-BOS self-sampling DEGENERATES into repetition
# loops (verified 2026-07-08: 'own own own...', anchors replay 0.524 vs floor
# 0.468 — no dynamic range). Targets must come from ENCODED CORPUS TEXT.
CORPUS="${CORPUS:-${ROOT}/data/corpus_targets.jsonl}"
# 31B bf16 = 62.5GB resident; batch must stay small on a 96GB card.
AV_BATCH="${AV_BATCH:-4}"
AV_EPOCHS="${AV_EPOCHS:-2}"
AV_LR_CE="${AV_LR_CE:-3e-4}"
AV_LR_DRAFT="${AV_LR_DRAFT:-1e-4}"
NP="${NP:-16}"

ART="${ROOT}/artifacts/nla/gemma4"
LOGS="${ROOT}/logs/gemma4_nla"
TARGETS="${ART}/targets_L${NLA_LAYER}_seq${NLA_SEQ_LEN}_10k.pt"
PAIRS="${ART}/gold_pairs_seq${NLA_SEQ_LEN}.jsonl"
mkdir -p "${ART}" "${LOGS}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

run_phase() { for ph in ${PHASES:-0 1 2 3 4 5 6 7 8}; do [[ "$ph" == "$1" ]] && return 0; done; return 1; }

# ---- 0. env ------------------------------------------------------------------
if run_phase 0; then
  echo "=== phase 0: env check ==="
  "${PY}" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
assert v >= (5, 12), f"need transformers>=5.12 for gemma4, got {transformers.__version__}"
assert hasattr(transformers, "Gemma4ForConditionalGeneration"), "no Gemma4 class"
import srt.nla  # noqa: F401
print("env OK")
EOF
fi

# ---- 1. smoke ------------------------------------------------------------------
if run_phase 1; then
  echo "=== phase 1: gemma-4 load smoke ==="
  "${PY}" - <<EOF
import torch
from srt.nla import load_frozen_backbone, num_layers_of, hidden_size_of
bb, tok = load_frozen_backbone("${BACKBONE}", "bfloat16", device="cuda")
assert type(bb).__name__ == "Gemma4ForConditionalGeneration", type(bb).__name__
assert num_layers_of(bb.config) == 60 and hidden_size_of(bb.config) == 5376
assert tok.bos_token_id != tok.eos_token_id, "bos==eos: sample_targets EOS guard would misfire"
# random-weights trap check: chat generation must be coherent
msgs = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France? Answer in one word."}]}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                              return_dict=True, return_tensors="pt")
ids = enc["input_ids"].cuda()
out = bb.generate(input_ids=ids, max_new_tokens=24, do_sample=False)
text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
print("gen:", text)
assert "Paris" in text, "generation incoherent — weights did NOT load (CausalLM trap?)"
o = bb(input_ids=ids, output_hidden_states=True, use_cache=False)
assert len(o.hidden_states) == 61, len(o.hidden_states)
mu_norm = o.hidden_states[${NLA_LAYER}][0, -1].float().norm().item()
print(f"L${NLA_LAYER} last-token ||h|| = {mu_norm:.1f}")
print("SMOKE OK")
EOF
fi

# ---- 2. targets ----------------------------------------------------------------
if run_phase 2; then
  echo "=== phase 2: encode ${NLA_NUM_SEQ} corpus targets @ L${NLA_LAYER} ==="
  [[ -f "${CORPUS}" ]] || { echo "MISSING corpus ${CORPUS} — rsync it first"; exit 1; }
  nohup "${PY}" scripts/sample_targets.py \
    --backbone "${BACKBONE}" --layer "${NLA_LAYER}" \
    --corpus "${CORPUS}" \
    --num-sequences "${NLA_NUM_SEQ}" --seq-len "${NLA_SEQ_LEN}" \
    --batch-size 8 --format pt --out "${TARGETS}" \
    > "${LOGS}/targets.log" 2>&1
  # sanity: targets must vary (902b746 guard)
  "${PY}" - <<EOF
import torch
obj = torch.load("${TARGETS}", map_location="cpu", weights_only=False)
t = torch.stack([a[-1] for a in obj["activations"]])
std = t.std(dim=0).mean().item()
print(f"target std = {std:.4f}")
assert std > 0.1, "targets collapsed — BOS/EOS trap"
EOF
fi

# ---- 3. pairs ------------------------------------------------------------------
if run_phase 3; then
  echo "=== phase 3: gold pairs ==="
  "${PY}" scripts/build_gold_pairs.py \
    --targets "${TARGETS}" --backbone "${BACKBONE}" \
    --max-len "${NLA_SEQ_LEN}" --out "${PAIRS}"
fi

# ---- 4. anchors ----------------------------------------------------------------
if run_phase 4; then
  echo "=== phase 4: anchors (centered frame) ==="
  nohup "${PY}" scripts/nla_anchors.py \
    --backbone "${BACKBONE}" --layer "${NLA_LAYER}" \
    --targets "${TARGETS}" --pairs "${PAIRS}" \
    --num-queries 100 --pool-size 2000 --prepend-bos \
    --out "${ART}/anchors_L${NLA_LAYER}.json" \
    > "${LOGS}/anchors.log" 2>&1
  cat "${ART}/anchors_L${NLA_LAYER}.json"
fi

# ---- 5. CE baseline AV (plain-greedy arm) --------------------------------------
if run_phase 5; then
  echo "=== phase 5: CE baseline AV ==="
  nohup "${PY}" scripts/train_nla_act.py \
    --targets "${TARGETS}" --pairs "${PAIRS}" \
    --backbone "${BACKBONE}" --layer "${NLA_LAYER}" \
    --num-prefix-tokens "${NP}" --max-seq-len "${NLA_SEQ_LEN}" \
    --ce-weight 1.0 --act-weight 0.0 \
    --select-metric centered --val-bestof 8 --prepend-bos \
    --epochs "${AV_EPOCHS}" --batch-size "${AV_BATCH}" --lr "${AV_LR_CE}" \
    --val-every 400 --val-vectors 200 \
    --out "${ART}/ce_seq${NLA_SEQ_LEN}_np${NP}" \
    > "${LOGS}/av_ce.log" 2>&1
fi

# ---- 6. draft-conditioned AV (THE greedy-gap test) ------------------------------
if run_phase 6; then
  echo "=== phase 6: draft-conditioned AV ==="
  nohup "${PY}" scripts/train_nla_draft.py \
    --targets "${TARGETS}" --pairs "${PAIRS}" \
    --backbone "${BACKBONE}" --layer "${NLA_LAYER}" \
    --init-from "${ART}/ce_seq${NLA_SEQ_LEN}_np${NP}/best_av.pt" \
    --num-prefix-tokens "${NP}" --max-seq-len "${NLA_SEQ_LEN}" \
    --prepend-bos \
    --epochs "${AV_EPOCHS}" --batch-size "${AV_BATCH}" --lr "${AV_LR_DRAFT}" \
    --val-every 400 --val-vectors 200 --val-bestof 8 \
    --out "${ART}/draft_seq${NLA_SEQ_LEN}_np${NP}" \
    > "${LOGS}/av_draft.log" 2>&1
  echo "READ OFF: val draft_cen (copy/NN baseline) vs greedy_cen (the test)"
  echo "vs phase-5 disc_cen (plain greedy). Win = greedy_cen > draft_cen."
fi

# ---- 7. vision transition --------------------------------------------------------
if run_phase 7; then
  echo "=== phase 7: verbalize image-position activations ==="
  nohup "${PY}" scripts/gemma4_vision_verbalize.py \
    --av-ckpt "${ART}/draft_seq${NLA_SEQ_LEN}_np${NP}/best_av.pt" \
    --targets "${TARGETS}" --pairs "${PAIRS}" --layer "${NLA_LAYER}" \
    --images artifacts/nla/gemma4/stereo/control.png \
             artifacts/nla/gemma4/stereo/stereogram.png \
             artifacts/nla/gemma4/stereo/disparity.png \
    --K 8 --out "${ART}/vision_verbalize_L${NLA_LAYER}.json" \
    > "${LOGS}/vision.log" 2>&1
  cat "${ART}/vision_verbalize_L${NLA_LAYER}.json"
fi

# ---- 8. backup ----------------------------------------------------------------
if run_phase 8; then
  echo "=== phase 8: HF backup (box is ephemeral) ==="
  "${VENV}/bin/hf" upload "${HF_USER}/srt-nla-gemma4-artifacts" "${ART}" . \
    --repo-type dataset --exclude "targets_*.pt" || true
  "${VENV}/bin/hf" upload "${HF_USER}/srt-nla-av-gemma4" \
    "${ART}/draft_seq${NLA_SEQ_LEN}_np${NP}/best_av.pt" draft/best_av.pt || true
  "${VENV}/bin/hf" upload "${HF_USER}/srt-nla-av-gemma4" \
    "${ART}/ce_seq${NLA_SEQ_LEN}_np${NP}/best_av.pt" ce/best_av.pt || true
  echo "backup done"
fi

echo "ALL REQUESTED PHASES DONE"
