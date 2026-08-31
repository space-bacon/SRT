#!/usr/bin/env python3
"""Fast generation core: load each model once, batch the prompts.

Two inefficiencies in the earlier scripts, both free to fix.

The loop was inverted. Iterating `for arm: for model:` reloads every model once
per arm, so the five-arm template decomposition loaded six models five times each.
gpt-oss takes 2:45 to load, which is roughly fourteen minutes of pure loading for
one model. Here the model is loaded once and every arm runs against it.

Prompts were issued one at a time. Each `generate()` call handled a single prompt
with num_return_sequences=k, leaving the device mostly idle on small models. Here
B prompts go through together, so each call produces B*k sequences.

CACHE NAMESPACE, DELIBERATELY SEPARATE. Batched sampling consumes the RNG in a
different order than unbatched sampling, so results are statistically equivalent
but not bit-identical. Mixing arms generated here with arms cached by the older
scripts inside one comparison would be a silent contamination bug, so this writes
to its own directory and never reads theirs.

Left padding is required: the prompt is stripped with input_ids.shape[1], which is
only the true boundary for every row when padding sits on the left.
"""
import gc
import json
import os
import time

import torch

DEV = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


class Progress:
    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()

    def tick(self, n=1):
        self.done += n

    def fmt(self):
        el = time.time() - self.t0
        if not self.done:
            return f"[   0/{self.total}  ETA --:--]"
        rate = self.done / el
        rem = (self.total - self.done) / rate if rate else 0
        return (f"[{self.done:5d}/{self.total} {100 * self.done / self.total:5.1f}%  "
                f"elapsed {int(el // 60)}:{int(el % 60):02d}  "
                f"ETA {int(rem // 60)}:{int(rem % 60):02d}  {rate * 60:.0f}/min]")


def run_model(tag, mid, prompts, modes, render, out_dir, prog, k=8, max_new=48,
              top_p=0.9, temp=1.0, batch=8, dtype=torch.float32, device_map=None):
    """Load `mid` once, generate every arm in `modes`, return {mode: [[str]*k]*n}.

    render(tok, stem, mode) -> (text, add_special_tokens, per_sample)
    per_sample True means the k samples use k DIFFERENT prompts, which is how the
    varied-persona arms work.

    device_map="auto" shards across every visible GPU, which the 160-330 GB
    frontier models need since none of them fit on one 96 GB card.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {m: f"{out_dir}/{tag}__{m}.json" for m in modes}
    have = {}
    for m, p in paths.items():
        if os.path.isfile(p):
            r = json.load(open(p))
            if len(r) == len(prompts) and len(r[0]) == k:
                have[m] = r
                prog.tick(len(prompts))
    todo = [m for m in modes if m not in have]
    if not todo:
        print(f"  {tag:16s} all {len(modes)} arms cached   {prog.fmt()}", flush=True)
        return have

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        mod = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, device_map=device_map)
    except ValueError:
        # Ministral-3 and friends ship as multimodal ForConditionalGeneration configs.
        from transformers import AutoModelForImageTextToText
        mod = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype,
                                                          device_map=device_map)
    # A sharded model is already placed; moving it would collapse it onto one card.
    mod = (mod if device_map else mod.to(DEV)).eval()
    print(f"  {tag:16s} loaded {time.time() - t0:.0f}s, running {len(todo)} arm(s)"
          + (f", sharded over {torch.cuda.device_count()} gpu(s)" if device_map else ""),
          flush=True)

    torch.manual_seed(0)
    # With device_map the embeddings may not be on cuda:0, and inputs must meet them.
    in_dev = getattr(mod, "device", DEV) if device_map else DEV
    for mode in todo:
        rows = []
        with torch.no_grad():
            for s in range(0, len(prompts), batch):
                chunk = prompts[s:s + batch]
                texts, add_sp, per_sample = [], True, False
                for stem in chunk:
                    t, add_sp, per_sample = render(tok, stem, mode)
                    texts.extend(t if per_sample else [t])
                if texts and isinstance(texts[0], list):
                    # Pre-tokenized: some backends cannot round-trip specials through a string.
                    b = tok.pad({"input_ids": texts}, return_tensors="pt",
                                padding=True).to(in_dev)
                else:
                    b = tok(texts, return_tensors="pt", padding=True,
                            add_special_tokens=add_sp).to(in_dev)
                g = mod.generate(**b, max_new_tokens=max_new, do_sample=True,
                                 top_p=top_p, temperature=temp,
                                 num_return_sequences=1 if per_sample else k,
                                 pad_token_id=tok.pad_token_id)
                cut = b["input_ids"].shape[1]
                dec = [tok.decode(x[cut:], skip_special_tokens=True).strip() for x in g]
                for i in range(len(chunk)):
                    rows.append(dec[i * k:(i + 1) * k])
                prog.tick(len(chunk))
                if (s // batch) % 4 == 0:
                    print(f"  {tag:16s} {mode:14s} {min(s + batch, len(prompts)):4d}/"
                          f"{len(prompts)}  {prog.fmt()}", flush=True)
        json.dump(rows, open(paths[mode], "w"), indent=1)
        have[mode] = rows
        print(f"  {tag:16s} {mode:14s} DONE          {prog.fmt()}", flush=True)

    del mod
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return have
