"""K-curve (best-of-K, K=1..64) for a gpt-oss-20b AV checkpoint vs anchors.

One generation pass of 64 samples per target; best-of-K for K=1,2,4,...,64
computed as prefix maxima over the same samples (matches rerank_eval.py).
"""
from __future__ import annotations
import json, os, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

BB = "openai/gpt-oss-20b"
L = 18
ART = "artifacts/nla/gptoss20b"
KMAX = 64
N_EVAL = 50
CHUNK = 160  # rollouts per generate call
CKPT = os.environ.get("AV_CKPT", f"{ART}/av_full_trace/best_av.pt")
INJECT_NORM = os.environ.get("AV_INJECT_NORM", "none")

tok = AutoTokenizer.from_pretrained(BB)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
bb = AutoModelForCausalLM.from_pretrained(BB, dtype=torch.bfloat16).cuda().eval()
for p in bb.parameters():
    p.requires_grad_(False)

cfg = NLAConfig(backbone_id=BB, backbone_dtype="bfloat16", extraction_layer=L,
                num_prefix_tokens=16, use_layer_embed=True, max_new_tokens=64,
                inject_norm=INJECT_NORM)
av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
sd = torch.load(CKPT, map_location="cuda", weights_only=False)
state = sd.get("trainable", sd) if isinstance(sd, dict) else sd
own = av.state_dict()
av.load_state_dict({k: v for k, v in state.items() if k in own and own[k].shape == v.shape},
                   strict=False)

pairs = [json.loads(l) for l in open(f"{ART}/trace_pairs.jsonl")]
l18 = [r["target_idx"] for r in pairs if int(r["layer"]) == L]
targets = torch.load(f"{ART}/trace_pairs.jsonl.targets.pt", map_location="cpu",
                     weights_only=False)["targets"].float()
mu = targets[torch.tensor(l18)].mean(0).cuda()
random.seed(0)
idx = random.sample(l18, N_EVAL)
Vq = targets[torch.tensor(idx)].cuda()


def fve(a, b):
    return 0.5 * (1.0 + F.cosine_similarity(a, b, dim=-1))


# one pass: KMAX samples per target
scores = torch.zeros(N_EVAL, KMAX)
v_rep = Vq.repeat_interleave(KMAX, 0)  # (N*K, d) contiguous per target
lay = torch.full((v_rep.size(0),), L, device="cuda")
texts: list[str] = []
for s in range(0, v_rep.size(0), CHUNK):
    g = av.generate(v_rep[s:s + CHUNK], max_new_tokens=64, do_sample=True,
                    temperature=1.0, layer=lay[s:s + CHUNK])
    texts.extend(tok.batch_decode(g, skip_special_tokens=True))
    print(f"gen {min(s + CHUNK, v_rep.size(0))}/{v_rep.size(0)}", flush=True)

for s in range(0, len(texts), CHUNK):
    batch = texts[s:s + CHUNK]
    live = [(j, t if t.strip() else " ") for j, t in enumerate(batch)]
    enc = tok([t for _, t in live], return_tensors="pt", padding=True,
              truncation=True, max_length=96).to("cuda")
    with torch.no_grad():
        o = bb(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
               output_hidden_states=True, use_cache=False)
    last = (enc.attention_mask.sum(1) - 1).clamp(min=0)
    h = o.hidden_states[L][torch.arange(len(live), device="cuda"), last].float()
    for b, (j, _) in enumerate(live):
        gidx = s + j
        ti, ki = gidx // KMAX, gidx % KMAX
        scores[ti, ki] = fve(h[b] - mu, Vq[ti] - mu).item()

print(f"\nckpt={CKPT} inject_norm={INJECT_NORM}  (centered fve, n={N_EVAL})")
print("anchors: floor=0.500  NN=0.744  replay=0.999")
out = {}
for K in (1, 2, 4, 8, 16, 32, 64):
    bok = scores[:, :K].max(dim=1).values.mean().item()
    out[K] = bok
    print(f"  K={K:>2}: {bok:.3f}")
json.dump({"ckpt": CKPT, "kcurve_cen": out},
          open(f"{ART}/kcurve.json", "w"), indent=2)
print(f"wrote {ART}/kcurve.json")
