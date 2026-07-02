"""Relabel the VQ state codebook with multi-member sampling (v2).

Fixes the label-artifact failure documented in the wave-2/4 red-team runs
(e.g. #3667 'mental health and well-being' is actually a broad
generic-declarative basin): high-traffic codes were labelled from their first
few (correlated) members. v2 samples members uniformly at random across the
whole membership, uses more members for broad basins, and writes member
counts so UIs can flag broad basins as approximate.

Output: artifacts/nla/gptoss20b/codebook_labels_v2.json
  {"labels": {code: label}, "counts": {code: n_members}, "method": "..."}
"""
from __future__ import annotations
import json, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BB = "openai/gpt-oss-20b"
ART = "artifacts/nla/gptoss20b"
BATCH = 32
SNIP = 110
SEED = 0

random.seed(SEED)

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

codes = torch.empty(targets.size(0), dtype=torch.long)
for s in range(0, targets.size(0), 4096):
    vc = targets[s:s + 4096].cuda() - mu
    codes[s:s + 4096] = torch.cdist(vc, cent).argmin(dim=-1).cpu()

# full membership index (row indices per code)
member_idx: dict[int, list[int]] = {}
for i in range(len(pairs)):
    member_idx.setdefault(int(codes[i]), []).append(i)

counts = {c: len(v) for c, v in member_idx.items()}
mean_count = sum(counts.values()) / max(1, len(counts))


def snippets_for(c: int) -> list[str]:
    idxs = member_idx.get(c, [])
    if not idxs:
        return [canon.get(c, "")[:SNIP]]
    if len(idxs) > 24:
        # broad basin: label from the CORE, not a uniform sample (uniform
        # samples of a diffuse cell are mutually incoherent -> empty labels).
        sub = random.sample(idxs, min(512, len(idxs)))
        vc = (targets[sub].cuda() - mu)
        d = torch.cdist(vc, cent[c:c + 1]).squeeze(-1)
        order = d.argsort()
        chosen = [sub[int(j)] for j in order[:8]]
    else:
        chosen = random.sample(idxs, min(5, len(idxs)))
    out = []
    for i in chosen:
        t = tok.decode(pairs[i]["gold_ids"], skip_special_tokens=True).strip()
        if t:
            out.append(t[-SNIP:])
    return out or [canon.get(c, "")[:SNIP]]


def make_prompt(c: int) -> str:
    body = "\n".join(f"- {s}" for s in snippets_for(c) if s)
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
              max_length=640).to("cuda")
    with torch.no_grad():
        out = bb.generate(**enc, max_new_tokens=12, do_sample=False,
                          pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.shape[1]:]
    for c, ids in zip(chunk, new):
        lab = tok.decode(ids, skip_special_tokens=True)
        lab = re.split(r"[\n.:;\"]", lab)[0].strip().strip("-* ")
        words = lab.split()
        labels[c] = " ".join(words[:7]) if words else ""
    if (s // BATCH) % 10 == 0:
        print(f"{s + len(chunk)}/{len(all_codes)}  e.g. {chunk[0]}"
              f" ({counts.get(chunk[0], 0)} members) -> {labels[chunk[0]]!r}",
              flush=True)

json.dump({"labels": {str(k): v for k, v in labels.items()},
           "counts": {str(k): counts.get(k, 0) for k in all_codes},
           "method": f"multi-member v3, centroid-core k=8 for broad basins / "
                     f"random k=5 small, seed={SEED}, structural few-shot"},
          open(f"{ART}/codebook_labels_v2.json", "w"), indent=0)
n_ok = sum(1 for v in labels.values() if v)
print(f"wrote {ART}/codebook_labels_v2.json  ({n_ok}/{len(labels)} non-empty)")

# sanity: print the previously-mislabeled basins
for c in (3667, 1807, 1672, 3249, 266, 2506, 2117, 2237):
    print(f"  #{c} ({counts.get(c, 0)} members): {labels.get(c, '')!r}")
