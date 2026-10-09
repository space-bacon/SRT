#!/usr/bin/env bash
# Single-GPU box: HF transformers stack, bf16 Qwen3.8-27B, ARC data.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace
log() { echo "{\"t\": $(date +%s), \"step\": \"$1\"}" >> /workspace/setup.jsonl; }
log start
apt-get update -qq && apt-get install -y -qq git tmux >/dev/null 2>&1
pip install -q uv
uv venv /workspace/venv --python 3.12 -q
source /workspace/venv/bin/activate
uv pip install -q torch --torch-backend=auto
uv pip install -q "transformers>=5.10.4,<5.18" accelerate pillow numpy huggingface_hub hf_transfer openai 2> /workspace/pip.err
log libs
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY' > /workspace/model_dl.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.8-27B", local_dir="/workspace/models/qwen3.8-27b", max_workers=8)
print("done")
PY
log model
[ -d ARC-AGI-2 ] || git clone --depth 1 -q https://github.com/arcprize/ARC-AGI-2.git
python /workspace/make_data.py /workspace/ARC-AGI-2 /workspace/data >> /workspace/setup.log 2>&1
log done
