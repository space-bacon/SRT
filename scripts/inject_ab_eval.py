#!/usr/bin/env python3
"""Inject-ON vs inject-OFF A/B on GSM8K — the steering-effect test.

Loads a frozen backbone + SRT adapter once, then for each GSM8K problem runs
**greedy** generation twice: with the FiLM injectors enabled and disabled. Greedy
decoding makes `disable_injectors` the ONLY variable, so any accuracy difference
is purely the injection effect (no sampling noise, no seed sensitivity).

Scoring is objective exact-match against the GSM8K gold final number, so no judge
is needed. Reports per-condition accuracy plus the McNemar breakdown (where the
two conditions disagree), which is the honest measure of whether injection helps.

Usage (M2 / MPS):
    python scripts/inject_ab_eval.py --n 20                  # quick smoke
    python scripts/inject_ab_eval.py --n 200 --out artifacts/inject_ab/gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402

# Canonical 4-shot GSM8K chain-of-thought exemplars (kept short to bound prefill
# cost on MPS). Each ends with "The answer is N." so the model learns the format.
FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Answer: In April she sold 48 clips. In May she sold half as many, so 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes she earned 50 * 0.2 = 10 dollars. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Half of 100 is 100 / 2 = 50 dollars. Her grandparents gave twice the parents' 15, so 2 * 15 = 30 dollars. Now she has 50 + 15 + 30 = 95 dollars. She still needs 100 - 95 = 5 dollars. The answer is 5.

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: Each time he writes 3 * 2 = 6 pages. Twice a week that is 6 * 2 = 12 pages per week. In a year that is 12 * 52 = 624 pages. The answer is 624.

"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-repo", default="RiverRider/srt-adapter-v1.0")
    p.add_argument("--n", type=int, default=20, help="number of GSM8K test problems")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16", help="bfloat16 (Qwen-safe) or float32")
    p.add_argument("--gen-method", default="cached", choices=["cached", "forward"],
                   help="cached = fast KV-cache generate() (CUDA); forward = slow "
                        "O(T^2) non-cached path (correct on CPU/MPS where the "
                        "cached path is broken)")
    p.add_argument("--device", default=None, help="default: mps if available else cpu")
    p.add_argument("--out", default="artifacts/inject_ab/gsm8k.json")
    return p.parse_args()


def pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Last number in the text, normalised (strip $ and commas)."""
    # Prefer the number after "the answer is", else the last number seen.
    m = re.search(r"answer is\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    cand = m.group(1) if m else None
    if cand is None:
        nums = _NUM.findall(text)
        if not nums:
            return None
        cand = nums[-1]
    cand = cand.replace(",", "").replace("$", "")
    # normalise 5.0 -> 5
    try:
        f = float(cand)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def gold_number(answer_field: str) -> str:
    raw = answer_field.split("####")[-1].strip().replace(",", "").replace("$", "")
    try:
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return raw


def truncate_continuation(text: str) -> str:
    """Cut the greedy continuation at the start of the next exemplar."""
    for stop in ("\nQuestion:", "\nQ:", "\n\nQuestion"):
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"== inject A/B on GSM8K | device={device} dtype={args.dtype} "
          f"adapter={args.adapter_repo} n={args.n} ==")

    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(args.adapter_repo, "config.json")
    w_path = hf_hub_download(args.adapter_repo, "adapter.safetensors")
    cfg = SRTConfig.from_json(cfg_path)
    cfg.backbone_dtype = args.dtype
    print(f"   backbone={cfg.backbone_id}  inject@{cfg.rrm_inject_indices}")

    t0 = time.time()
    model = SRTAdapter(cfg)
    model.load_state_dict(load_safetensors(w_path, device="cpu"), strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_ids = {tok.eos_token_id}
    print(f"   model ready in {time.time() - t0:.0f}s")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    def gen(prompt: str, inject: bool) -> str:
        if args.gen_method == "cached":
            # Fast KV-cached path. Correct on CUDA (the showcase backend); known
            # broken on CPU/MPS where SDPA's is_causal path misbehaves for the
            # cached decode shape — use --gen-method forward there.
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(
                ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids,
                temperature=0.0, disable_injectors=(not inject),
            )
            new = out[0] if isinstance(out, torch.Tensor) else out
            return truncate_continuation(tok.decode(new, skip_special_tokens=True))
        # forward() fallback: parity-verified, O(T^2), early-stop at the answer.
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = ids.shape[1]
        cur = ids
        for _ in range(args.max_new_tokens):
            out = model(input_ids=cur, attention_mask=torch.ones_like(cur),
                        disable_injectors=(not inject))
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            if int(nxt.item()) in eos_ids:
                break
            if (cur.shape[1] - prompt_len) % 8 == 0:
                tail = tok.decode(cur[0, prompt_len:], skip_special_tokens=True)
                if re.search(r"answer is\s*\$?-?\d[\d,]*(?:\.\d+)?\s*\.", tail, re.IGNORECASE) \
                        or "\nQuestion:" in tail:
                    break
        cont = tok.decode(cur[0, prompt_len:], skip_special_tokens=True)
        return truncate_continuation(cont)

    # ── Startup sanity check: never run a full eval on degenerate output. ──
    sanity_prompt = FEWSHOT + ("Question: A robe takes 2 bolts of blue fiber and "
                               "half that much white fiber. How many bolts in total?\nAnswer:")
    sanity = gen(sanity_prompt, inject=False)
    ok = extract_final_number(sanity) is not None and "the other, I was" not in sanity
    print(f"   sanity (inject-off): {sanity[:80]!r}")
    if not ok:
        print("   !! SANITY FAILED: generation looks degenerate on this backend.")
        print(f"   !! gen-method={args.gen_method} device={device}. "
              "On CPU/MPS use --gen-method forward. Aborting.")
        sys.exit(2)
    print("   sanity OK\n")

    # McNemar cells: both right, on-only right, off-only right, both wrong.
    both = on_only = off_only = neither = 0
    on_correct = off_correct = 0
    rows = []
    t_start = time.time()
    for i in range(min(args.n, len(ds))):
        q = ds[i]["question"]
        gold = gold_number(ds[i]["answer"])
        prompt = FEWSHOT + f"Question: {q}\nAnswer:"

        cont_on = gen(prompt, inject=True)
        cont_off = gen(prompt, inject=False)
        a_on = extract_final_number(cont_on)
        a_off = extract_final_number(cont_off)
        ok_on = a_on == gold
        ok_off = a_off == gold
        on_correct += ok_on
        off_correct += ok_off
        if ok_on and ok_off:
            both += 1
        elif ok_on:
            on_only += 1
        elif ok_off:
            off_only += 1
        else:
            neither += 1
        rows.append({"i": i, "gold": gold, "on": a_on, "off": a_off,
                     "ok_on": ok_on, "ok_off": ok_off})
        elapsed = time.time() - t_start
        print(f"  [{i + 1}/{args.n}] gold={gold:>6}  on={str(a_on):>6}{'✓' if ok_on else '✗'}  "
              f"off={str(a_off):>6}{'✓' if ok_off else '✗'}  "
              f"| acc on={on_correct}/{i + 1} off={off_correct}/{i + 1}  "
              f"({elapsed / (i + 1):.0f}s/prob)")

    n = len(rows)
    result = {
        "adapter": args.adapter_repo, "backbone": cfg.backbone_id,
        "n": n, "device": device, "dtype": args.dtype, "decoding": "greedy",
        "acc_on": on_correct / n if n else 0.0,
        "acc_off": off_correct / n if n else 0.0,
        "delta": (on_correct - off_correct) / n if n else 0.0,
        "mcnemar": {"both_right": both, "on_only": on_only,
                    "off_only": off_only, "neither": neither},
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"INJECT A/B — GSM8K (greedy, n={n})")
    print("=" * 60)
    print(f"  accuracy inject-ON : {on_correct}/{n} = {result['acc_on']:.3f}")
    print(f"  accuracy inject-OFF: {off_correct}/{n} = {result['acc_off']:.3f}")
    print(f"  delta (on - off)   : {result['delta']:+.3f}")
    print(f"  disagreements      : on-only-right={on_only}  off-only-right={off_only}")
    print(f"  (both right={both}, neither={neither})")
    # McNemar exact-ish read: net of the discordant pairs.
    disc = on_only + off_only
    if disc:
        print(f"  among {disc} disagreements, inject-ON won {on_only} "
              f"({on_only / disc:.0%})")
    print("=" * 60)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
