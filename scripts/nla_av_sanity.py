"""One-off AV sanity check for the gpt-oss-20b NLA run.

1. Decodes a few bare-BOS samples (the Phase-4 sampling distribution) to see if
   the harmony/chat-tuned model produced coherent text or degenerate control-token
   soup.
2. Computes best-of-K raw vs CENTERED fve for the trained AV on held-out layer-18
   targets, plus a centered random floor — so we know whether the modest raw
   fve (~0.55) is real underperformance or just gpt-oss anisotropy.
"""
from __future__ import annotations
import json, os, random, statistics as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from srt.nla import ActivationVerbalizer, NLAConfig

BB = "openai/gpt-oss-20b"
L = 18
ART = "artifacts/nla/gptoss20b"
K = 8
N_EVAL = 50
CKPT = os.environ.get("AV_CKPT", f"{ART}/av_full_trace/best_av.pt")
INJECT_NORM = os.environ.get("AV_INJECT_NORM", "none")
SKIP_SAMPLES = os.environ.get("SKIP_SAMPLES", "0") == "1"

tok = AutoTokenizer.from_pretrained(BB)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
bb = AutoModelForCausalLM.from_pretrained(BB, dtype=torch.bfloat16).cuda().eval()
for p in bb.parameters():
    p.requires_grad_(False)

# ---- 1. sampling quality: decode bare-BOS generations ----------------------
if not SKIP_SAMPLES:
    bos = tok.bos_token_id or bb.config.bos_token_id
    ids = torch.full((5, 1), bos, dtype=torch.long, device="cuda")
    gen = bb.generate(input_ids=ids, max_new_tokens=64, do_sample=True,
                      temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    print("=== 5 bare-BOS Phase-4-style samples ===")
    for i in range(5):
        print(f"[{i}] {tok.decode(gen[i], skip_special_tokens=True)[:220]!r}")

# ---- 2. centered fve of the trained AV -------------------------------------
cfg = NLAConfig(backbone_id=BB, backbone_dtype="bfloat16", extraction_layer=L,
                num_prefix_tokens=16, use_layer_embed=True, max_new_tokens=64,
                inject_norm=INJECT_NORM)
av = ActivationVerbalizer(cfg, backbone=bb, tokenizer=tok).cuda().eval()
sd = torch.load(CKPT, map_location="cuda", weights_only=False)
state = sd.get("trainable", sd) if isinstance(sd, dict) else sd
own = av.state_dict()
filt = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
missing = [k for k in own if k not in filt and not k.startswith("backbone.")]
av.load_state_dict(filt, strict=False)
print(f"\nAV loaded: {len(filt)} tensors, missing adapter keys: {missing}")

pairs = [json.loads(l) for l in open(f"{ART}/trace_pairs.jsonl")]
l18 = [r["target_idx"] for r in pairs if int(r["layer"]) == L]
targets = torch.load(f"{ART}/trace_pairs.jsonl.targets.pt", map_location="cpu",
                     weights_only=False)["targets"].float()
mu = targets[torch.tensor(l18)].mean(0).cuda()
random.seed(0)
idx = random.sample(l18, N_EVAL)
Vv = targets[torch.tensor(idx)].cuda()


def fve(a, b):
    return 0.5 * (1.0 + F.cosine_similarity(a, b, dim=-1))


raw_best, cen_best, raw_greedy, cen_greedy = [], [], [], []
for i in range(0, len(idx), 10):
    vb = Vv[i:i + 10]
    lay = torch.full((vb.size(0) * K,), L, device="cuda")
    vr = vb.repeat_interleave(K, 0)
    g = av.generate(vr, max_new_tokens=64, do_sample=True, temperature=1.0, layer=lay)
    txt = tok.batch_decode(g, skip_special_tokens=True)
    enc = tok(txt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    o = bb(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
           output_hidden_states=True, use_cache=False)
    last = (enc.attention_mask.sum(1) - 1).clamp(min=0)
    h = o.hidden_states[L][torch.arange(len(txt), device="cuda"), last].float()
    h = h.view(vb.size(0), K, -1)
    for j in range(vb.size(0)):
        vj = vb[j]
        rr = fve(h[j], vj.unsqueeze(0)); cc = fve(h[j] - mu, (vj - mu).unsqueeze(0))
        raw_best.append(rr.max().item()); cen_best.append(cc.max().item())
        raw_greedy.append(rr[0].item()); cen_greedy.append(cc[0].item())

perm = torch.randperm(Vv.size(0))
floor_raw = fve(Vv, Vv[perm]).mean().item()
floor_cen = fve(Vv - mu, (Vv - mu)[perm]).mean().item()

print(f"\nckpt={CKPT}  inject_norm={INJECT_NORM}")
print(f"||mu|| (L{L} anisotropy) = {mu.norm():.1f}")
print(f"random floor:   raw={floor_raw:.3f}  centered={floor_cen:.3f}")
print(f"AV sampled(1):  raw={st.mean(raw_greedy):.3f}  centered={st.mean(cen_greedy):.3f}")
print(f"AV best-of-{K}:  raw={st.mean(raw_best):.3f}  centered={st.mean(cen_best):.3f}  (n={len(raw_best)})")
