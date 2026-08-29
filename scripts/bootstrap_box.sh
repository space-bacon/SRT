#!/bin/bash
# Bring a fresh vast.ai box up to working state in one command.
#
# Assumes nothing except ssh access and a CUDA image. Syncs the repo, installs
# what the encoders actually need, verifies the GPU can run a kernel, and
# optionally drags anything irreplaceable off an old box before it is reclaimed.
#
#   bash scripts/bootstrap_box.sh -p 12345 -h ssh5.vast.ai
#   bash scripts/bootstrap_box.sh -p 12345 -h ssh5.vast.ai --rescue 35745:ssh8.vast.ai
set -eu

PORT=""; HOST=""; RESCUE=""
while [ $# -gt 0 ]; do
  case $1 in
    -p) PORT=$2; shift 2;;
    -h) HOST=$2; shift 2;;
    --rescue) RESCUE=$2; shift 2;;   # oldport:oldhost
    *) echo "unknown arg $1"; exit 1;;
  esac
done
[ -n "$PORT" ] && [ -n "$HOST" ] || { echo "need -p PORT -h HOST"; exit 1; }
SSH="ssh -p $PORT root@$HOST"
REPO=$(cd "$(dirname "$0")/.." && pwd)

echo "=== 1/5 repo -> $HOST:/root/srt-adapter"
rsync -rlptD -e "ssh -p $PORT" \
  --exclude '.venv*' --exclude '__pycache__' --exclude '.git' \
  --exclude 'artifacts' --exclude 'checkpoints' --exclude '.mypy_cache' \
  "$REPO/" "root@$HOST:/root/srt-adapter/"

echo "=== 2/5 dependencies"
$SSH 'set -e
mkdir -p /root/logs /root/.hf_home
PY=$(command -v /venv/main/bin/python || command -v python3)
echo "  python: $PY"
$PY -m pip install -q --upgrade pip
# nibabel for CT volumes; the rest are usually present but pinning avoids surprises.
$PY -m pip install -q nibabel pillow numpy huggingface_hub datasets
$PY -c "import transformers, torch; print(f\"  transformers {transformers.__version__}  torch {torch.__version__}\")"'

echo "=== 3/5 GPU check (Blackwell needs cu128; cu121 wheels have no sm_120)"
$SSH 'PY=$(command -v /venv/main/bin/python || command -v python3)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<PYEOF
import torch
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}, devices {torch.cuda.device_count()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"  capability sm_{cap[0]}{cap[1]}")
    try:  # a real kernel, not just a device count
        x = torch.randn(1024, 1024, device="cuda")
        print(f"  matmul ok, result finite: {bool(torch.isfinite(x @ x).all())}")
    except Exception as e:
        print(f"  KERNEL FAILED: {type(e).__name__}: {e}")
        print("  fix: pip install --index-url https://download.pytorch.org/whl/cu128 torch")
else:
    print("  NO CUDA")
PYEOF'

echo "=== 4/5 disk"
$SSH 'df -h / | tail -1'

if [ -n "$RESCUE" ]; then
  OLD_PORT=${RESCUE%%:*}; OLD_HOST=${RESCUE##*:}
  echo "=== 5/5 rescuing irreplaceable state from $OLD_HOST"
  # These exist on no other machine. Pull them before the old box is reclaimed.
  for f in caps_l47 full_states_L20_L47.npz full_img_vecs.npy \
           qwen38_img_vecs.npy train30k_states.npz depth_states_all9.npz \
           l60_states.npz; do
    echo "  $f"
    $SSH "rsync -a -e 'ssh -p $OLD_PORT -o StrictHostKeyChecking=no' \
      root@$OLD_HOST:/root/$f /root/ 2>/dev/null || echo '    (absent, skipped)'"
  done
else
  echo "=== 5/5 no --rescue given, skipping migration"
fi

echo
echo "=== READY"
echo "  ssh -p $PORT root@$HOST"
echo "  export HF_HOME=/root/.hf_home PYTHONPATH=/root/srt-adapter"
