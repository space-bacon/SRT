#!/usr/bin/env bash
# L4 test box: vLLM venv, ARC data, FP8 and INT4 Qwen3.8-27B checkpoints. Progress goes to /workspace/setup.jsonl.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /workspace
log() { echo "{\"t\": $(date +%s), \"step\": \"$1\"}" >> /workspace/setup.jsonl; }
log start
apt-get update -qq && apt-get install -y -qq tmux git rsync ninja-build >/dev/null 2>&1
pip install -q uv huggingface_hub hf_transfer
log tools
(
  export HF_HUB_ENABLE_HF_TRANSFER=1
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("RedHatAI/Qwen3.8-27B-INT4", local_dir="/workspace/models/qwen3.8-27b-int4")
print("int4 done", flush=True)
snapshot_download("Qwen/Qwen3.8-27B-FP8", local_dir="/workspace/models/qwen3.8-27b-fp8")
print("fp8 done", flush=True)
PY
  log models
) > /workspace/model_dl.log 2>&1 &
MP=$!
uv venv /workspace/venv --python 3.12 -q
source /workspace/venv/bin/activate
uv pip install -q vllm --torch-backend=auto 2> /workspace/vllm_install.err
log vllm_installed
uv pip install -q openai httpx numpy transformers 2>> /workspace/vllm_install.err
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'gpus')" >> /workspace/setup.log 2>&1
[ -d ARC-AGI-2 ] || git clone --depth 1 -q https://github.com/arcprize/ARC-AGI-2.git
python /workspace/make_data.py /workspace/ARC-AGI-2 /workspace/data >> /workspace/setup.log 2>&1
log data
wait $MP
log done
