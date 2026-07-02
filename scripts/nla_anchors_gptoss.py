"""gpt-oss-20b L18 anchors: replay ceiling, NN-retrieval baseline, random floor.

Centered metric throughout (mu from the L18 target pool). Gives the honest
reference frame for judging the AV (standing rule: never report fve without
a centered metric + retrieval baseline).
"""
from __future__ import annotations
import json, random, statistics as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BB = "openai/gpt-oss-20b"
L = 18
ART = "artifacts/nla/gptoss20b"
M = 100          # query targets
POOL = 2000      # NN retrieval pool

tok = AutoTokenizer.from_pretrained(BB)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
bb = AutoModelForCausalLM.from_pretrained(BB, dtype=torch.bfloat16).cuda().eval()
for p in bb.parameters():
    p.requires_grad_(False)

pairs = [json.loads(l) for l in open(f"{ART}/trace_pairs.jsonl")]
l18 = [r for r in pairs if int(r["layer"]) == L]
targets = torch.load(f"{ART}/trace_pairs.jsonl.targets.pt", map_location="cpu",
                     weights_only=False)["targets"].float()

random.seed(0)
random.shuffle(l18)
queries, pool_rows = l18[:M], l18[M:M + POOL]
Vq = targets[torch.tensor([r["target_idx"] for r in queries])].cuda()
Vp = targets[torch.tensor([r["target_idx"] for r in pool_rows])].cuda()
mu = torch.cat([Vq, Vp]).mean(0)


def fve(a, b):
    return 0.5 * (1.0 + F.cosine_similarity(a, b, dim=-1))


@torch.no_grad()
def encode_texts(id_lists):
    """Last-token L18 hidden for each token-id list (prefix-free forward)."""
    out = []
    for i in range(0, len(id_lists), 16):
        chunk = id_lists[i:i + 16]
        T = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); attn[j, :len(x)] = 1
        o = bb(input_ids=ids.cuda(), attention_mask=attn.cuda(),
               output_hidden_states=True, use_cache=False)
        last = (attn.sum(1) - 1).clamp(min=0)
        out.append(o.hidden_states[L][torch.arange(len(chunk)), last].float())
    return torch.cat(out)

# 1. replay ceiling: re-encode each query's own gold prefix.
H_replay = encode_texts([r["gold_ids"] for r in queries])
rep_raw = fve(H_replay, Vq).mean().item()
rep_cen = fve(H_replay - mu, Vq - mu).mean().item()

# 2. NN retrieval: nearest pool vector (centered cos), re-encode ITS gold text.
qn = F.normalize(Vq - mu, dim=-1); pn = F.normalize(Vp - mu, dim=-1)
nn_idx = (qn @ pn.T).argmax(dim=1)
H_nn = encode_texts([pool_rows[int(i)]["gold_ids"] for i in nn_idx])
nn_raw = fve(H_nn, Vq).mean().item()
nn_cen = fve(H_nn - mu, Vq - mu).mean().item()

# 3. random floor: shuffled pool vectors as "reconstructions".
perm = torch.randperm(M)
fl_raw = fve(Vp[:M], Vq[perm]).mean().item()
fl_cen = fve(Vp[:M] - mu, (Vq - mu)[perm]).mean().item()

print(f"gpt-oss-20b L18 anchors (M={M}, pool={POOL}, ||mu||={mu.norm():.1f})")
print(f"  replay ceiling : raw={rep_raw:.3f}  centered={rep_cen:.3f}")
print(f"  NN retrieval   : raw={nn_raw:.3f}  centered={nn_cen:.3f}")
print(f"  random floor   : raw={fl_raw:.3f}  centered={fl_cen:.3f}")
json.dump({"replay_raw": rep_raw, "replay_cen": rep_cen, "nn_raw": nn_raw,
           "nn_cen": nn_cen, "floor_raw": fl_raw, "floor_cen": fl_cen,
           "mu_norm": float(mu.norm()), "M": M, "pool": POOL, "layer": L},
          open(f"{ART}/anchors_L18.json", "w"), indent=2)
print(f"wrote {ART}/anchors_L18.json")
