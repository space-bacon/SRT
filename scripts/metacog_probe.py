"""Metacognition probe on gpt-oss-20b: does read-layer choice matter, and is
the P(wrong) signal punctuation-spoofable?

Motivated by ginigen-ai's Metacognition-Bench adapters (last hidden state ->
MLP -> P(wrong), read at the FINAL layer) and our wave-4 finding that
gpt-oss-20b's final layer carries a punctuation-driven completeness flag.

Protocol:
 1. Sample N TriviaQA (rc.nocontext, validation) questions; greedy-generate
    short answers from the frozen backbone; grade against answer aliases.
 2. Collect last-token hidden states at L18 and L24 for "Q: ...\nA: {ans}".
 3. Train the ginigen MLP probe per layer (67/33 split), report AUROC vs the
    mean-logprob self-confidence baseline.
 4. Spoof battery on the test split: truncated answer vs truncated + fake
    period. Report mean dP(wrong) = P(trunc) - P(trunc+'.') per layer.
    A positive value means the fake period lowers the probe's error estimate.

Output: artifacts/nla/gptoss20b/metacog_probe.json
"""
from __future__ import annotations

import json
import random
import re
import string

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BB = "openai/gpt-oss-20b"
ART = "artifacts/nla/gptoss20b"
N = 1200
GEN_BS = 32
ENC_BS = 16
LAYERS = (18, 24)
SEED = 0

random.seed(SEED)
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(BB)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
bb = AutoModelForCausalLM.from_pretrained(BB, dtype=torch.bfloat16).cuda().eval()
for p in bb.parameters():
    p.requires_grad_(False)

# ---------------- 1. data + generation ----------------
from datasets import load_dataset  # noqa: E402

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
idx = random.sample(range(len(ds)), N)
rows = [{"q": ds[i]["question"],
         "aliases": [a.lower() for a in ds[i]["answer"]["normalized_aliases"]]}
        for i in idx]


def norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", "", s).strip()


prompts = [f"Answer the trivia question with just the answer.\n"
           f"Q: {r['q']}\nA:" for r in rows]
answers: list[str] = []
logps: list[float] = []
for s in range(0, len(prompts), GEN_BS):
    chunk = prompts[s:s + GEN_BS]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=128).to("cuda")
    with torch.no_grad():
        out = bb.generate(**enc, max_new_tokens=16, do_sample=False,
                          pad_token_id=tok.pad_token_id,
                          return_dict_in_generate=True, output_scores=True)
    new = out.sequences[:, enc.input_ids.shape[1]:]
    # mean logprob of generated tokens (self-confidence baseline)
    lp = torch.stack(out.scores, dim=1).log_softmax(-1)
    for b in range(new.shape[0]):
        ids = new[b]
        keep = ids != tok.pad_token_id
        n_tok = int(keep.sum())
        tok_lp = lp[b, torch.arange(ids.shape[0]), ids][keep]
        logps.append(float(tok_lp.mean()) if n_tok else -20.0)
        txt = tok.decode(ids, skip_special_tokens=True)
        answers.append(txt.split("\n")[0].strip())
    if s % (GEN_BS * 5) == 0:
        print(f"gen {s + len(chunk)}/{len(prompts)}", flush=True)

correct = []
for r, a in zip(rows, answers):
    na = norm(a)
    correct.append(int(bool(na) and any(
        norm(al) == na or (norm(al) and norm(al) in na) for al in r["aliases"])))
acc = sum(correct) / len(correct)
print(f"accuracy: {acc:.3f}  ({sum(correct)}/{len(correct)})", flush=True)


# ---------------- 2. hidden states (3 variants) ----------------
def variants(q: str, a: str) -> dict[str, str]:
    words = a.split()
    trunc = " ".join(words[:-1]) if len(words) > 1 else a[: max(1, len(a) // 2)]
    base = f"Q: {q}\nA: "
    return {"full": base + a,
            "trunc": base + trunc,
            "trunc_dot": base + trunc + "."}


texts: dict[str, list[str]] = {"full": [], "trunc": [], "trunc_dot": []}
for r, a in zip(rows, answers):
    v = variants(r["q"], a or "unknown")
    for k in texts:
        texts[k].append(v[k])

H: dict[str, dict[int, torch.Tensor]] = {}
for name, lst in texts.items():
    hs = {L: torch.empty(len(lst), bb.config.hidden_size) for L in LAYERS}
    for s in range(0, len(lst), ENC_BS):
        enc = tok(lst[s:s + ENC_BS], return_tensors="pt", padding=True,
                  truncation=True, max_length=160).to("cuda")
        with torch.no_grad():
            out = bb(**enc, output_hidden_states=True, use_cache=False)
        # left padding -> last position is the last real token
        for L in LAYERS:
            hs[L][s:s + enc.input_ids.shape[0]] = \
                out.hidden_states[L][:, -1].float().cpu()
    H[name] = hs
    print(f"encoded {name}", flush=True)


# ---------------- 3. probes ----------------
def make_probe(d: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 4), nn.GELU(),
                         nn.Dropout(0.1), nn.Linear(d // 4, 1))


def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, r = 0.0, 1
    for _, y in pairs:
        if y == 1:
            rank_sum += r
        r += 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


perm = list(range(N))
random.shuffle(perm)
n_tr = int(N * 0.67)
tr, te = perm[:n_tr], perm[n_tr:]
y = torch.tensor([1 - c for c in correct], dtype=torch.float32)  # 1 = wrong

results: dict = {"n": N, "accuracy": acc, "seed": SEED,
                 "baseline_logp_auroc": None, "layers": {}}
# baseline: high logp should mean less wrong -> score = -logp predicts wrong
results["baseline_logp_auroc"] = auroc([-logps[i] for i in te],
                                       [1 - correct[i] for i in te])

for L in LAYERS:
    X = H["full"][L]
    probe = make_probe(X.shape[1]).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = X[tr].cuda(), y[tr].cuda()
    for ep in range(60):
        probe.train()
        pred = probe(Xtr).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(probe(X[te].cuda()).squeeze(-1)).cpu()
        p_tr_full = torch.sigmoid(probe(Xtr).squeeze(-1)).cpu()
        p_trunc = torch.sigmoid(probe(H["trunc"][L][te].cuda()).squeeze(-1)).cpu()
        p_dot = torch.sigmoid(probe(H["trunc_dot"][L][te].cuda()).squeeze(-1)).cpu()
    a = auroc(p_te.tolist(), [1 - correct[i] for i in te])
    dspoof = (p_trunc - p_dot).mean().item()
    frac_drop = float((p_dot < p_trunc).float().mean())
    results["layers"][L] = {
        "auroc": a,
        "spoof_dP_mean": dspoof,           # >0: fake period lowers P(wrong)
        "spoof_frac_lowered": frac_drop,   # fraction of test items where it did
        "p_wrong_full_mean": float(p_te.mean()),
        "p_wrong_trunc_mean": float(p_trunc.mean()),
        "p_wrong_trunc_dot_mean": float(p_dot.mean()),
    }
    print(f"L{L}: auroc={a:.3f}  spoof_dP={dspoof:+.3f} "
          f"(lowered on {frac_drop:.0%} of items)  "
          f"P(wrong): full={p_te.mean():.3f} trunc={p_trunc.mean():.3f} "
          f"trunc+.'='{p_dot.mean():.3f}", flush=True)

with open(f"{ART}/metacog_probe.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {ART}/metacog_probe.json")
