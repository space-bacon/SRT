"""Regenerate coder arms on Apple Silicon with a budget that lets them finish.

The banked ladder arms were generated at max_new=192 and are 45% to 78%
truncated mid-fence, which is why Qwen2.5-Coder-32B scores 0.2439 there and
0.8986 in the ensemble. A truncated function cannot parse and cannot pass, so
those arms measure the token budget more than the model.

Everything here matches `coder_ladder.py` except the budget: same HumanEval
prompts, same chat template, same K, same top_p 0.9 and temperature 1.0.

    python scripts/coder_regen_mlx.py --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit --tag coder3B_inst
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prompts", type=int, default=164)
    ap.add_argument("--probs", default="data/humaneval.json")
    ap.add_argument("--out", default="artifacts/nla/coder_regen")
    a = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    he = json.load(open(a.probs))[: a.prompts]
    stems = [p["prompt"] for p in he]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.tag}__chat.json")

    rows = json.load(open(path)) if os.path.isfile(path) else []
    if len(rows) >= len(stems):
        print(f"{a.tag}: already complete ({len(rows)} problems)", flush=True)
        return

    model, tok = load(a.model)
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    print(f"{a.tag}: {len(stems) - len(rows)} problems left, K={a.k}, "
          f"max_new={a.max_new}", flush=True)

    t0 = time.time()
    for i in range(len(rows), len(stems)):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": stems[i]}], add_generation_prompt=True)
        out = batch_generate(model, tok, prompts=[ids] * a.k,
                             max_tokens=a.max_new, sampler=sampler, verbose=False)
        rows.append(list(out.texts))
        if (i + 1) % 5 == 0 or i + 1 == len(stems):
            json.dump(rows, open(path, "w"))
            done = i + 1 - 0
            rate = (time.time() - t0) / max(1, i + 1 - (len(rows) - len(rows)))
            eta = (len(stems) - done) * ((time.time() - t0) / max(1, done))
            print(f"  {done}/{len(stems)}  {time.time() - t0:.0f}s elapsed  "
                  f"eta {eta / 60:.0f}m", flush=True)

    json.dump(rows, open(path, "w"))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
