# SRT-NLA v1 — HF push runbook

Run from the vast box that holds `best_av.pt` + the targets file (the new
box at `ssh -p 17091 root@ssh8.vast.ai` as of 2026-05-18). Do NOT run from
the Mac unless you have both artifacts mirrored locally — the targets file
alone is ~26 GB.

The cards + `config.json` live in this directory (`docs/hf/nla_v1/`), which
is tracked in git, so `git pull` brings everything the push script needs.

## 0. Prereqs (on the vast box)

```bash
cd /workspace/srt-adapter         # or wherever the repo lives
git fetch origin && git checkout nla && git pull --ff-only

pip install -U "huggingface_hub>=0.26"
huggingface-cli login             # paste a write token for RiverRider
```

Confirm both source files exist:

```bash
ls -lh artifacts/nla/ce_seq64_np16_30k/best_av.pt
ls -lh artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt
```

If the paths differ, just pass the correct ones via `--ckpt` / `--targets`.

## 1. Push the model repo (small, fast)

```bash
python scripts/push_nla_v1_to_hf.py model \
    --ckpt artifacts/nla/ce_seq64_np16_30k/best_av.pt
```

What lands on `https://huggingface.co/RiverRider/srt-nla-av-v1`:

- `README.md`  (= `release/nla_v1/MODEL_CARD.md`)
- `config.json` (NLAConfig fields for `ce_seq64_np16` lineage)
- `best_av.pt`  (~50 MB)
- `eval/centered_eval_30k_M200.json`
- `eval/oracle_ceiling_30k_v2.json`
- `eval/rerank_eval_ce_seq64_np16_v2.json`

## 2. Push the dataset repo (large, slow)

```bash
python scripts/push_nla_v1_to_hf.py dataset \
    --targets artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt
```

What lands on `https://huggingface.co/datasets/RiverRider/srt-nla-targets-v1`:

- `README.md`  (= `release/nla_v1/DATASET_CARD.md`)
- `targets_q7b_L20_seq64_30k_seed1.pt`  (~26 GB)

Expect 20–60 min depending on the vast box uplink. `huggingface_hub`
auto-resumes on transient failures.

## 3. Optional: keep private during preview

Add `--private` to either command; flip to public from the HF web UI once
the cards look right.

## 4. Post-push verification (from anywhere)

```bash
huggingface-cli download RiverRider/srt-nla-av-v1 config.json --quiet
python -c "
from huggingface_hub import hf_hub_download
from srt.nla import NLAConfig
cfg = NLAConfig.from_json(hf_hub_download('RiverRider/srt-nla-av-v1', 'config.json'))
print(cfg)
"
```

If that prints an `NLAConfig` with `num_prefix_tokens=16`, the model card
load snippet works end-to-end.

## 5. Optional sanity smoke (loads weights, no generation)

```bash
python - <<'PY'
import torch
from huggingface_hub import hf_hub_download
from srt.nla import NLAConfig
from srt.nla.av import ActivationVerbalizer  # adjust import to your AV class

cfg_path = hf_hub_download("RiverRider/srt-nla-av-v1", "config.json")
ckpt_path = hf_hub_download("RiverRider/srt-nla-av-v1", "best_av.pt")
cfg = NLAConfig.from_json(cfg_path)
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print("AV state keys:", list(state.keys())[:6], "...")
print("config:", cfg)
PY
```
