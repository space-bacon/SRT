#!/usr/bin/env python
"""Teacher-forced hidden-state features for saved reasoning traces.

For every finished sample in the run JSONL it rebuilds prompt + reasoning + answer, runs one forward pass of the
bf16 model, and keeps the hidden state at two positions (last reasoning token, last answer token) for a few layers.
Output is one npz per shard, written through a temp file and renamed. One JSON line per sample goes to the log.
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--layers", default="16,24,32,40,48,56")
    ap.add_argument("--max-len", type=int, default=120000)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    lm = model.model.language_model
    store = {}

    def mk(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[i] = h
        return hook
    for i in layers:
        lm.layers[i].register_forward_hook(mk(i))

    data = json.load(open(a.challenges))
    rows = [json.loads(l) for l in open(a.run) if '"status": "ok"' in l]
    rows = [r for j, r in enumerate(sorted(rows, key=lambda r: (r["key"], r["sample"]))) if j % a.nshards == a.shard]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys, feats = [], []
    log = open(a.out + ".jsonl", "a")
    for r in rows:
        tid, ti = r["key"].rsplit("_", 1)
        prompt = R.build_prompt(data[tid], int(ti))
        with gzip.open(os.path.join(a.traces, f"{r['key']}__{r['sample']}.txt.gz"), "rt") as f:
            reasoning = f.read()
        head = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        if not head.rstrip().endswith("<think>"):
            head = head + "<think>\n"
        ids_head = tok(head, add_special_tokens=False)["input_ids"]
        ids_think = tok(reasoning.strip() + "\n", add_special_tokens=False)["input_ids"]
        ids_close = tok("</think>\n\n", add_special_tokens=False)["input_ids"]
        ids_ans = tok(r["content"], add_special_tokens=False)["input_ids"] or tok("\n", add_special_tokens=False)["input_ids"]
        ids = ids_head + ids_think + ids_close + ids_ans
        if len(ids) > a.max_len:
            log.write(json.dumps({"key": r["key"], "sample": r["sample"], "status": "too_long", "n": len(ids)}) + "\n"); log.flush()
            continue
        pos_think = len(ids_head) + len(ids_think) - 1
        pos_ans = len(ids) - 1
        t0 = time.time()
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device="cuda:0"), use_cache=False, logits_to_keep=1)
        v = np.stack([np.stack([store[i][0, pos_think].float().cpu().numpy(), store[i][0, pos_ans].float().cpu().numpy()]) for i in layers])
        store.clear()
        keys.append((r["key"], r["sample"]))
        feats.append(v.astype(np.float16))
        log.write(json.dumps({"key": r["key"], "sample": r["sample"], "status": "ok", "n_tokens": len(ids), "seconds": round(time.time() - t0, 1)}) + "\n"); log.flush()
        if len(feats) % 20 == 0:
            tmp = a.out + ".tmp.npz"
            np.savez(tmp, keys=np.array(keys), feats=np.stack(feats), layers=np.array(layers))
            os.replace(tmp, a.out)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, keys=np.array(keys), feats=np.stack(feats) if feats else np.zeros((0,)), layers=np.array(layers))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
