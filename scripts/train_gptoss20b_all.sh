#!/usr/bin/env bash
# =============================================================================
# Full SRT + NLA training for gpt-oss-20b — ALL phases, ALL capability.
#
# Runs, end to end, on a single Blackwell box (2x RTX PRO 6000 here):
#   0. env      — dedicated venv: transformers>=4.55,<5 + kernels/triton + SRT
#   1. smoke    — tiny gpt_oss parity test + real-20b 5-check smoke + bos!=eos
#   2. adapter  — Phase-A read-out SRT adapter (regime/community/bif/chain heads)
#   3. probe    — held-out Phase-A probe (ECE / AUROC / NMI / r_hat)
#   4. nla_tgt  — sample MULTI-LAYER trace targets (input->output, all layers)
#   5. nla_pair — flatten to (v, prefix) pairs across layers/positions
#   6. nla_av   — train the Activation Verbalizer with the layer embedding
#   7. nla_book — build the persistent magic-number state codebook
#   8. backup   — push every checkpoint/artifact to HF (box is EPHEMERAL)
#
# The box filesystem does NOT persist (workspace_is_volume=false), so phase 8
# is not optional. Set HF_USER + HF token before a real run.
#
# Usage:
#   export HF_USER=RiverRider HF_TOKEN=hf_xxx
#   PHASES="0 1 2 3 4 5 6 7 8" bash scripts/train_gptoss20b_all.sh
#   # or a subset, e.g. just the NLA capability:
#   PHASES="4 5 6 7" bash scripts/train_gptoss20b_all.sh
#
# Every long phase is launched with nohup into logs/gptoss20b/<phase>.log; tail
# those in a separate shell to watch progress.
# =============================================================================
set -euo pipefail

# ---- knobs (override via env) ----------------------------------------------
ROOT="${ROOT:-/workspace/srt-adapter}"
BACKBONE="${BACKBONE:-openai/gpt-oss-20b}"
DTYPE="${DTYPE:-bfloat16}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
# Install into the image's main env (has torch 2.12+cu130 for Blackwell).
VENV="${VENV:-/venv/main}"
PY="${VENV}/bin/python"

# Data (rsync the phase1 corpus here first; see README/handoff).
DATA_DIR="${DATA_DIR:-${ROOT}/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/all_train.jsonl}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/all_val.jsonl}"

# gpt-oss-20b has 24 hidden layers. NLA extraction ~73% depth -> L18.
NLA_LAYER="${NLA_LAYER:-18}"
# Multi-layer trace coverage. "all" = every layer (heavy); default = a spread.
NLA_TRACE_LAYERS="${NLA_TRACE_LAYERS:-6,12,18,24}"

# Adapter Phase-A
ADAPTER_OUT="${ADAPTER_OUT:-${ROOT}/checkpoints/gptoss20b_phaseAB}"
ADAPTER_BATCH="${ADAPTER_BATCH:-8}"
ADAPTER_VAL_EVERY="${ADAPTER_VAL_EVERY:-2000}"
# Combined Phase A (read-out heads) + Phase B (closed-loop inject-CE) in ONE
# run = full mode (no --read-only). MXFP4 backprop through the frozen 20b works
# (verified: grad finite). Set READ_ONLY=1 to train heads only (Phase A alone).
READ_ONLY="${READ_ONLY:-}"

# NLA
NLA_NUM_SEQ="${NLA_NUM_SEQ:-10000}"
NLA_SEQ_LEN="${NLA_SEQ_LEN:-64}"
NLA_POS_STRIDE="${NLA_POS_STRIDE:-4}"
NLA_MAX_PAIRS="${NLA_MAX_PAIRS:-200000}"
NLA_PREFIX_TOKENS="${NLA_PREFIX_TOKENS:-16}"
NLA_EPOCHS="${NLA_EPOCHS:-2}"
CODEBOOK_MODE="${CODEBOOK_MODE:-vq}"
CODEBOOK_K="${CODEBOOK_K:-4096}"
CODEBOOK_NBITS="${CODEBOOK_NBITS:-32}"

HF_USER="${HF_USER:-}"

PHASES="${PHASES:-0 1 2 3 4 5 6 7 8}"

ART="${ROOT}/artifacts/nla/gptoss20b"
LOGS="${ROOT}/logs/gptoss20b"
mkdir -p "$ART" "$LOGS" "$ADAPTER_OUT"
cd "$ROOT"

have() { echo " $PHASES " | grep -q " $1 "; }
log()  { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 0. env ----------------------------------------------------------------
if have 0; then
  log "PHASE 0: env @ $VENV (reusing image torch cu13x)"
  # Install into the image's main env (already has Blackwell torch). Only add
  # transformers>=4.55 (gpt_oss) + MXFP4 kernels; never reinstall torch.
  if [ ! -x "$PY" ]; then
    python3 -m venv --system-site-packages "$VENV"
  fi
  "$VENV/bin/pip" install -q -U pip
  # gpt_oss needs transformers>=4.55; stay <5 (5.x garbles SRT KV-cached decode).
  # Pin triton to match the image torch (2.12 wants ==3.7.0) so MXFP4 kernels load.
  "$VENV/bin/pip" install -q -U "transformers>=4.55,<5" "kernels>=0.4" "accelerate>=0.34" \
      safetensors huggingface_hub numpy scikit-learn pyarrow pytest datasets
  "$VENV/bin/pip" install -q "triton==3.7.0"
  # SRT itself WITHOUT deps (its requirements.txt pins transformers==4.53.3).
  "$VENV/bin/pip" install -q -e . --no-deps
  "$PY" -c "import torch,transformers; from transformers.models import gpt_oss; \
    print('torch',torch.__version__,'transformers',transformers.__version__,'gpt_oss OK')"
fi

# ---- 1. smoke --------------------------------------------------------------
if have 1; then
  log "PHASE 1: smoke (tiny parity + real-20b 5-check + bos!=eos)"
  "$PY" -m pytest tests/test_gptoss_smoke.py -q
  "$PY" scripts/qwen3_smoke.py --backbone "$BACKBONE" --dtype "$DTYPE" 2>&1 | tee "$LOGS/smoke.log"
  "$PY" - <<PY
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("$BACKBONE")
print("bos", t.bos_token_id, "eos", t.eos_token_id,
      "-> DISTINCT" if t.bos_token_id != t.eos_token_id else "-> COLLIDE (902b746 guard must fire)")
PY
fi

# ---- 2. adapter Phase A+B --------------------------------------------------
if have 2; then
  log "PHASE 2: SRT adapter (A read-out + B inject-CE) -> $ADAPTER_OUT"
  [ -f "$TRAIN_DATA" ] || { echo "MISSING $TRAIN_DATA (rsync phase1 corpus first)"; exit 1; }
  RO_FLAG=""; [ -n "$READ_ONLY" ] && RO_FLAG="--read-only"
  nohup "$PY" scripts/train.py \
    --backbone "$BACKBONE" $RO_FLAG --dtype "$DTYPE" \
    --train-data "$TRAIN_DATA" --val-data "$VAL_DATA" \
    --max-val-samples 5000 --batch-size "$ADAPTER_BATCH" \
    --val-every "$ADAPTER_VAL_EVERY" --output-dir "$ADAPTER_OUT" \
    > "$LOGS/adapter.log" 2>&1 &
  echo "  launched adapter train (pid $!) -> tail -f $LOGS/adapter.log"
  wait $!
fi

# ---- 3. Phase-A probe ------------------------------------------------------
if have 3; then
  log "PHASE 3: held-out probe"
  "$PY" scripts/phaseA_probe.py --backbone "$BACKBONE" --dtype "$DTYPE" \
    --adapter "$ADAPTER_OUT/best_adapter.pt" --val-data "$VAL_DATA" \
    --max-samples 3000 --n-clusters 64 \
    --out "$ROOT/artifacts/regime_calibration/gptoss20b_phaseA.json" 2>&1 | tee "$LOGS/probe.log"
fi

# ---- 4. NLA multi-layer trace targets --------------------------------------
if have 4; then
  log "PHASE 4: sample multi-layer NLA targets (layers=$NLA_TRACE_LAYERS)"
  nohup "$PY" scripts/sample_targets.py \
    --backbone "$BACKBONE" --dtype "$DTYPE" \
    --layers "$NLA_TRACE_LAYERS" --num-sequences "$NLA_NUM_SEQ" \
    --seq-len "$NLA_SEQ_LEN" --batch-size 8 \
    --out "$ART/targets_multiL.pt" \
    > "$LOGS/nla_targets.log" 2>&1 &
  echo "  launched target sampling (pid $!) -> tail -f $LOGS/nla_targets.log"
  wait $!
fi

# ---- 5. flatten to trace pairs ---------------------------------------------
if have 5; then
  log "PHASE 5: build trace pairs"
  "$PY" scripts/build_trace_pairs.py \
    --sample "$ART/targets_multiL.pt" --layers all \
    --position-stride "$NLA_POS_STRIDE" --min-prefix-len 2 \
    --max-pairs "$NLA_MAX_PAIRS" \
    --out "$ART/trace_pairs.jsonl" 2>&1 | tee "$LOGS/nla_pairs.log"
fi

# ---- 6. train the Activation Verbalizer (layer-conditioned) ----------------
if have 6; then
  log "PHASE 6: train AV with layer embedding"
  nohup "$PY" scripts/train_nla_act.py \
    --targets "$ART/trace_pairs.jsonl.targets.pt" \
    --pairs "$ART/trace_pairs.jsonl" \
    --backbone "$BACKBONE" --dtype "$DTYPE" --layer "$NLA_LAYER" \
    --num-prefix-tokens "$NLA_PREFIX_TOKENS" --use-layer-embed \
    --ce-weight 1.0 --act-weight 0.0 \
    --max-seq-len "$NLA_SEQ_LEN" --epochs "$NLA_EPOCHS" \
    --batch-size 16 --lr 3e-4 --val-every 200 --val-vectors 512 \
    --out "$ART/av_full_trace" \
    > "$LOGS/nla_av.log" 2>&1 &
  echo "  launched AV train (pid $!) -> tail -f $LOGS/nla_av.log"
  wait $!
fi

# ---- 7. persistent state codebook ------------------------------------------
if have 7; then
  log "PHASE 7: build $CODEBOOK_MODE state codebook"
  EXTRA=""
  [ "$CODEBOOK_MODE" = "vq" ] && EXTRA="--k $CODEBOOK_K" || EXTRA="--n-bits $CODEBOOK_NBITS"
  "$PY" scripts/build_state_codebook.py \
    --targets "$ART/trace_pairs.jsonl.targets.pt" --pairs "$ART/trace_pairs.jsonl" \
    --backbone "$BACKBONE" --mode "$CODEBOOK_MODE" $EXTRA \
    --out "$ART/state_codebook_${CODEBOOK_MODE}.pt" 2>&1 | tee "$LOGS/nla_book.log"
fi

# ---- 8. backup (box is ephemeral!) -----------------------------------------
if have 8; then
  log "PHASE 8: backup to HF"
  # Token: prefer $HF_TOKEN from env; else rely on a prior `hf auth login` store.
  HF="$VENV/bin/hf"
  [ -x "$HF" ] || HF="$VENV/bin/huggingface-cli"
  # Resolve namespace from the token when HF_USER is unset ("choose best").
  if [ -z "$HF_USER" ]; then
    HF_USER=$("$HF" auth whoami 2>/dev/null | head -1 | tr -d '[:space:]')
  fi
  if [ -z "$HF_USER" ] || [ "$HF_USER" = "Notloggedin" ]; then
    echo "  No HF auth. Set a token first (on the box, typed by YOU, not via chat):"
    echo "    $HF auth login --token <YOUR_HF_TOKEN>"
    echo "  then re-run: PHASES=8 bash scripts/train_gptoss20b_all.sh"
  else
    echo "  namespace: $HF_USER"
    PROBE="$ROOT/artifacts/regime_calibration/gptoss20b_phaseAB_v2_probe.json"
    # Adapter checkpoint + its held-out probe result.
    "$HF" upload "$HF_USER/srt-adapter-gptoss20b" \
      "$ADAPTER_OUT/best_adapter.pt" best_adapter.pt || true
    [ -f "$PROBE" ] && "$HF" upload "$HF_USER/srt-adapter-gptoss20b" \
      "$PROBE" phaseAB_probe.json || true
    # Activation Verbalizer.
    "$HF" upload "$HF_USER/srt-nla-av-gptoss20b" \
      "$ART/av_full_trace/best_av.pt" best_av.pt || true
    # NLA artifacts dataset — EXCLUDE the 35 GB regenerable raw targets
    # (targets_multiL.pt); keep the codebook, pairs, and flat AV-retrain targets.
    "$HF" upload-large-folder "$HF_USER/srt-nla-gptoss20b-artifacts" \
      "$ART" --repo-type dataset --exclude "targets_multiL.pt" || true
  fi
fi

log "DONE phases: $PHASES"
