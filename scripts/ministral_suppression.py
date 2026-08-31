"""Does the suppression result of section 5.5 survive at scale?

Section 5.5 drove convergence down with eight distinct personas on six models of
0.36B to 2B. The same small-model objection that section 5.2 answered for section 5
applies to it. This repeats the persona arms on Ministral-3 at 3B, 8B and 14B.

The chat baseline is reused from the ministral ladder rather than regenerated. All
chat-template arms go through the pre-tokenized path, because this family silently
destroys special tokens when a template is rendered to a string and re-encoded.

    python scripts/ministral_suppression.py --k 8 --prompts 60
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from hivemind_gen import Progress, run_model  # noqa: E402
from ministral_ladder import STEMS  # noqa: E402

# Copied verbatim from scripts/hivemind_suppression.py so this runs without
# pulling in that module's Mac-local model paths.
PERSONAS = [
    "You are a blunt, opinionated essayist who despises cliches.",
    "You are a working plumber giving plain practical advice.",
    "You are a poet who answers in vivid concrete images.",
    "You are a skeptical scientist who hedges every claim carefully.",
    "You are an enthusiastic teenager texting a close friend.",
    "You are a nineteenth-century letter writer, formal and unhurried.",
    "You are a terse expert who answers in one short sentence.",
    "You are a stand-up comedian working the answer into a bit.",
]
DEPLOYED = [
    "You are a helpful, harmless, and honest AI assistant.",
    "You are a large language model. Answer accurately and concisely.",
    "You are an AI assistant. Provide clear, accurate and helpful responses.",
    "You are a friendly and knowledgeable assistant. Be concise but thorough.",
    "You are an AI assistant designed to be maximally helpful while remaining safe.",
    "You are a helpful assistant. Answer directly and avoid unnecessary preamble.",
    "You are an AI language model. Respond helpfully and without verbosity.",
    "You are a thoughtful assistant. Give balanced, well-reasoned answers.",
]

HUB = "/root/.hf_home/hub"
OUT = "artifacts/nla/ministral_suppression"
SIZES = ["3B", "8B", "14B"]
MODES = ["persona_model", "persona_sample", "deployed_model", "deployed_sample"]
SCORER = "sentence-transformers/all-MiniLM-L6-v2"


def snap(repo):
    g = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/")
    return g[0] if g else None


def make_render(model_idx, k):
    def render(tok, stem, mode):
        pool = PERSONAS if mode.startswith("persona") else DEPLOYED
        if mode.endswith("_model"):
            people = [pool[model_idx % len(pool)]]
            per_sample = False
        else:
            people = [pool[j % len(pool)] for j in range(k)]
            per_sample = True
        outs = []
        for p in people:
            msgs = [{"role": "system", "content": p}, {"role": "user", "content": stem}]
            try:
                ids = tok.apply_chat_template(msgs, tokenize=True,
                                              add_generation_prompt=True)["input_ids"]
            except Exception:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": f"{p}\n\n{stem}"}],
                    tokenize=True, add_generation_prompt=True)["input_ids"]
            outs.append(ids)
        return (outs if per_sample else outs[0]), False, per_sample
    return render


def embed(texts, batch=256):
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(SCORER)
    mod = AutoModel.from_pretrained(SCORER).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    X = torch.cat(out).double()
    return torch.nn.functional.normalize(X, dim=1).numpy()


def stats(per_model, rng):
    """intra within a model, inter across models on the same prompt, and a floor."""
    tags = sorted(per_model)
    intra, fracs = [], []
    for t in tags:
        E = per_model[t]
        k = E.shape[1]
        iu, ju = zip(*itertools.combinations(range(k), 2))
        v = (E[:, iu, :] * E[:, ju, :]).sum(-1).mean(1)
        intra.append(float(v.mean()))
        fracs.append(float((v > 0.8).mean()))
    inter = []
    for a, b in itertools.combinations(tags, 2):
        A, B = per_model[a], per_model[b]
        inter.append(float(np.einsum("pki,pli->pkl", A, B).mean()))
    F = np.concatenate([per_model[t].reshape(-1, per_model[t].shape[-1]) for t in tags])
    n_p = per_model[tags[0]].shape[0]
    k = per_model[tags[0]].shape[1]
    p = rng.integers(0, len(F), 20000)
    q = rng.integers(0, len(F), 20000)
    ok = (p % (n_p * k)) // k != (q % (n_p * k)) // k
    return {"intra_mean": round(float(np.mean(intra)), 4),
            "inter_mean": round(float(np.mean(inter)), 4),
            "frac_models_any_prompt_above_0.8": round(float(np.mean([f > 0 for f in fracs])), 4),
            "mean_frac_prompts_above_0.8": round(float(np.mean(fracs)), 4),
            "floor": round(float((F[p[ok]] * F[q[ok]]).sum(-1).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=60)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--out", default=f"{OUT}/results.json")
    a = ap.parse_args()

    prompts = STEMS[:a.prompts]
    jobs = []
    for i, s in enumerate(SIZES):
        repo = f"mistralai/Ministral-3-{s}-Instruct-2512"
        jobs.append((f"ministral{s}_inst", repo, i))
    mine = [j for n, j in enumerate(jobs) if n % a.shards == a.shard]
    print(f"shard {a.shard}/{a.shards}  {len(mine)} models  {len(prompts)} prompts  "
          f"K={a.k}", flush=True)

    prog = Progress(len(mine) * len(MODES) * len(prompts))
    for tag, repo, idx in mine:
        p = snap(repo)
        if not p:
            print(f"  {tag} MISSING {repo}", flush=True)
            continue
        run_model(tag, p, prompts, MODES, make_render(idx, a.k),
                  os.path.join(HERE, OUT), prog, k=a.k, max_new=a.max_new,
                  batch=a.batch, dtype=torch.bfloat16)
    if a.gen_only:
        return

    rng = np.random.default_rng(0)
    res = {"question": "does persona suppression survive at 3B to 14B",
           "family": "Ministral-3, Mistral AI", "scorer": SCORER,
           "baseline": "chat arm from artifacts/nla/ministral_ladder", "arms": {}}
    for mode in ["chat"] + MODES:
        src = ("artifacts/nla/ministral_ladder" if mode == "chat" else OUT)
        per = {}
        for s in SIZES:
            f = os.path.join(HERE, src, f"ministral{s}_inst__{mode}.json")
            if not os.path.isfile(f):
                break
            g = json.load(open(f))
            flat = [t if t.strip() else " " for row in g for t in row]
            per[s] = embed(flat).reshape(len(g), len(g[0]), -1)
        if len(per) == len(SIZES):
            res["arms"][mode] = stats(per, rng)
            v = res["arms"][mode]
            print(f"  {mode:16s} intra {v['intra_mean']:.4f}  inter {v['inter_mean']:.4f}  "
                  f"floor {v['floor']:.4f}", flush=True)

    if "chat" in res["arms"]:
        b = res["arms"]["chat"]
        res["suppression"] = {
            m: {"intra_drop": round(b["intra_mean"] - res["arms"][m]["intra_mean"], 4),
                "inter_drop": round(b["inter_mean"] - res["arms"][m]["inter_mean"], 4)}
            for m in MODES if m in res["arms"]}
        for m, v in res["suppression"].items():
            print(f"  {m:16s} intra drop {v['intra_drop']:+.4f}  "
                  f"inter drop {v['inter_drop']:+.4f}", flush=True)

    op = os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {op}", flush=True)


if __name__ == "__main__":
    main()
