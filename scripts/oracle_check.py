"""Oracle diagnostic: do gold token ids reproduce the stored target activation?

For each (target, gold_ids) pair we run the frozen backbone on the *gold* ids
without any prefix, extract layer-L last-token activation, and compute
fve_nrm against the stored target. If targets were honestly saved and there
is no tokenizer round-trip corruption, fve_nrm should be ~1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--n", type=int, default=64, help="number of pairs to check")
    p.add_argument("--prepend-bos", action="store_true",
                   help="prepend BOS token to match sampler context")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(args.targets, map_location="cpu", weights_only=False)
    acts = obj["activations"]  # list of (T_i, d)
    seqs = obj["sequences"]

    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=torch.bfloat16).to(device).eval()

    # Read first N pair records.
    pairs = []
    with args.pairs.open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
            if len(pairs) >= args.n:
                break

    bos = tok.bos_token_id or model.config.bos_token_id or 0
    fve_list = []
    cos_list = []
    for r in pairs:
        idx = r["target_idx"]
        ids = list(r["gold_ids"])
        if args.prepend_bos:
            ids = [bos] + ids
        t = torch.tensor(ids, device=device).unsqueeze(0)
        attn = torch.ones_like(t)
        with torch.no_grad():
            out = model(input_ids=t, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[args.layer][0, -1].float().cpu()  # (d,)
        v_tgt = acts[idx][-1].float()  # (d,)
        # Compare
        hn = F.normalize(h, dim=-1)
        tn = F.normalize(v_tgt, dim=-1)
        fve = 1.0 - ((hn - tn) ** 2).sum().item() / 2.0
        cos = float((hn * tn).sum().item())
        fve_list.append(fve)
        cos_list.append(cos)

    import statistics as st
    print(f"n={len(fve_list)}  prepend_bos={args.prepend_bos}")
    print(f"  fve_nrm  mean={st.mean(fve_list):.4f}  median={st.median(fve_list):.4f}  min={min(fve_list):.4f}  max={max(fve_list):.4f}")
    print(f"  cos      mean={st.mean(cos_list):.4f}  median={st.median(cos_list):.4f}")


if __name__ == "__main__":
    main()
