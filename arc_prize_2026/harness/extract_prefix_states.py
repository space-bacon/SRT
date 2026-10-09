#!/usr/bin/env python
"""Hidden-state and confidence features at several prefix lengths of saved reasoning traces.

One forward pass over prompt + the first max(--positions) reasoning tokens gives every prefix position at once, because the model is
causal. For each position and each layer in --layers it stores the state at that token (float16), plus running confidence features
(mean token logprob and mean entropy proxy over the preceding window). Output npz is written through a temp file and renamed;
one JSON line per sample goes to <out>.jsonl.
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_llm_arc as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="direct", choices=["direct", "code"])
    ap.add_argument("--positions", default="2048,4096,8192,16384,24576,32768")
    ap.add_argument("--layers", default="24,32,40,48")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    positions = [int(x) for x in a.positions.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    lm = model.model.language_model
    store = {}
    for i in layers:
        lm.layers[i].register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(i, out[0] if isinstance(out, tuple) else out))
    head = model.lm_head

    data = json.load(open(a.challenges))
    rows = [json.loads(l) for l in open(a.run) if '"status": "ok"' in l]
    rows = [r for j, r in enumerate(sorted(rows, key=lambda r: (r["key"], r["sample"]))) if j % a.nshards == a.shard]
    log = open(a.out + ".jsonl", "a")
    keys, feats, conf = [], [], []
    for r in rows:
        tid, ti = r["key"].rsplit("_", 1)
        prompt = R.build_prompt(data[tid], int(ti), a.mode)
        with gzip.open(os.path.join(a.traces, f"{r['key']}__{r['sample']}.txt.gz"), "rt") as f:
            reasoning = f.read().strip()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        if not text.rstrip().endswith("<think>"):
            text += "<think>\n"
        ids_head = tok(text, add_special_tokens=False)["input_ids"]
        ids_r = tok(reasoning, add_special_tokens=False)["input_ids"]
        use = [p for p in positions if p <= len(ids_r)]
        if not use:
            log.write(json.dumps({"key": r["key"], "sample": r["sample"], "status": "too_short", "n": len(ids_r)}) + "\n"); log.flush()
            continue
        ids = ids_head + ids_r[: max(use)]
        t0 = time.time()
        with torch.no_grad():
            out = model.model(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False)
            h = out.last_hidden_state[0]
            lp = []
            for s in range(len(ids_head) - 1, len(ids) - 1, 4096):
                e = min(s + 4096, len(ids) - 1)
                logits = head(h[s:e]).float()
                logp = torch.log_softmax(logits, -1)
                tgt = torch.tensor(ids[s + 1:e + 1], device="cuda:0")
                ent = -(logp.exp() * logp).sum(-1)
                lp.append(torch.stack([logp.gather(1, tgt[:, None])[:, 0], ent], 1).cpu())
            lp = torch.cat(lp).numpy()
        v = np.zeros((len(positions), len(layers), h.shape[-1]), np.float16)
        c = np.full((len(positions), 4), np.nan, np.float32)
        for k, p in enumerate(positions):
            if p > len(ids_r):
                continue
            idx = len(ids_head) + p - 1
            for j, l in enumerate(layers):
                v[k, j] = store[l][0, idx].float().cpu().numpy()
            w = lp[max(0, p - 2048):p]
            c[k] = [lp[:p, 0].mean(), lp[:p, 1].mean(), w[:, 0].mean(), w[:, 1].mean()]
        store.clear()
        keys.append((r["key"], r["sample"]))
        feats.append(v)
        conf.append(c)
        log.write(json.dumps({"key": r["key"], "sample": r["sample"], "status": "ok", "n_reasoning_tokens": len(ids_r), "used": max(use), "seconds": round(time.time() - t0, 1)}) + "\n"); log.flush()
        if len(feats) % 20 == 0:
            tmp = a.out + ".tmp.npz"
            np.savez(tmp, keys=np.array(keys), feats=np.stack(feats), conf=np.stack(conf), positions=np.array(positions), layers=np.array(layers))
            os.replace(tmp, a.out)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, keys=np.array(keys), feats=np.stack(feats) if feats else np.zeros(0), conf=np.stack(conf) if conf else np.zeros(0), positions=np.array(positions), layers=np.array(layers))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
