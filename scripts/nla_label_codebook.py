"""Self-label the VQ state codebook: gpt-oss-20b names its own states.

For each of the 4096 codes, gather up to 3 member gold-prefix snippets
(states that quantize to that code), prompt the frozen backbone for a short
topic/function label, and write {code: label} JSON.
"""
from __future__ import annotations
import json, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BB = "openai/gpt-oss-20b"
ART = "artifacts/nla/gptoss20b"
BATCH = 48
MAX_MEMBERS = 3
SNIP = 110

tok = AutoTokenizer.from_pretrained(BB)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
bb = AutoModelForCausalLM.from_pretrained(BB, dtype=torch.bfloat16).cuda().eval()
for p in bb.parameters():
    p.requires_grad_(False)

obj = torch.load(f"{ART}/state_codebook_vq.pt", map_location="cpu", weights_only=False)
mu = obj["mu"].float().cuda()
cent = obj["centroids"].float().cuda()
canon = {int(e["code"]): (e["text"] or "") for e in obj["entries"]}

pairs = [json.loads(l) for l in open(f"{ART}/trace_pairs.jsonl")]
targets = torch.load(f"{ART}/trace_pairs.jsonl.targets.pt", map_location="cpu",
                     weights_only=False)["targets"].float()

# assign every pair's state to a code (batched cdist)
codes = torch.empty(targets.size(0), dtype=torch.long)
for s in range(0, targets.size(0), 4096):
    vc = targets[s:s + 4096].cuda() - mu
    codes[s:s + 4096] = torch.cdist(vc, cent).argmin(dim=-1).cpu()

members: dict[int, list[str]] = {}
for i, r in enumerate(pairs):
    c = int(codes[i])
    lst = members.setdefault(c, [])
    if len(lst) < MAX_MEMBERS:
        t = tok.decode(r["gold_ids"], skip_special_tokens=True).strip()
        if t:
            lst.append(t[-SNIP:])

def make_prompt(c: int) -> str:
    snips = members.get(c) or [canon.get(c, "")[:SNIP]]
    body = "\n".join(f"- {s}" for s in snips if s)
    return (
        "Snippets:\n"
        "- The recipe calls for two cups of flour and a pinch of salt\n"
        "- Bake at 350 degrees until the crust turns golden brown\n"
        "Label: baking and recipe instructions\n\n"
        "Snippets:\n"
        "- The defendant filed a motion to dismiss the charges\n"
        "- Under state law the landlord must give thirty days notice\n"
        "Label: legal procedures and rights\n\n"
        f"Snippets:\n{body}\n"
        "Label:")

all_codes = sorted(canon.keys())
labels: dict[int, str] = {}
for s in range(0, len(all_codes), BATCH):
    chunk = all_codes[s:s + BATCH]
    prompts = [make_prompt(c) for c in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=256).to("cuda")
    with torch.no_grad():
        out = bb.generate(**enc, max_new_tokens=12, do_sample=False,
                          pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.shape[1]:]
    for c, ids in zip(chunk, new):
        lab = tok.decode(ids, skip_special_tokens=True)
        lab = re.split(r"[\n.:;\"]", lab)[0].strip().strip("-* ")
        words = lab.split()
        labels[c] = " ".join(words[:7]) if words else ""
    if (s // BATCH) % 5 == 0:
        print(f"{s + len(chunk)}/{len(all_codes)}  e.g. {chunk[0]} -> {labels[chunk[0]]!r}", flush=True)

json.dump({str(k): v for k, v in labels.items()},
          open(f"{ART}/codebook_labels.json", "w"), indent=0)
n_ok = sum(1 for v in labels.values() if v)
print(f"wrote {ART}/codebook_labels.json  ({n_ok}/{len(labels)} non-empty)")
