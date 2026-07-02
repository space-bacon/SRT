#!/usr/bin/env python3
"""Targeted red-teaming of gpt-oss-20b recurring internal states via the live
SRT-NLA Space (RiverRider/srt-nla-gptoss20b-trace).

Drives /cb_compare with prompt pairs engineered to activate, invert, or
destabilize specific recurring codebook states, then catalogs per-layer state
identity + centered cosine. Run from local; the Space's ZeroGPU does the work.

Usage:
    python scripts/redteam_states.py [--out artifacts/nla/gptoss20b/redteam_states.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from gradio_client import Client

SPACE = "RiverRider/srt-nla-gptoss20b-trace"


def _local_token() -> str | None:
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

# Recurring-state targets cataloged from codebook_labels.json + prior traces.
TARGETS = {
    "closure_1672": {
        "codes": [1672],
        "note": "L24 sentence-closure attractor (seen recurring in chirality test)",
    },
    "list_logic_577": {
        "codes": [577, 560, 11, 524, 867],
        "note": "logical/list/enumeration states ('1' digit labels, logical reasoning)",
    },
    "mental_health_34": {
        "codes": [34, 1323, 1914, 3063, 3667],
        "note": "over-represented 'mental health and well-being' cluster",
    },
    "code_fragments_32": {
        "codes": [32, 54, 78, 81, 22, 59, 63],
        "note": "fragmented code-snippet / tokenization states",
    },
}

# (family, technique, prompt_a, prompt_b)
PAIRS = [
    # --- closure / #1672: unusual or meta sentence endings -----------------
    ("closure_1672", "direct", "I think, therefore I.", "I think, therefore I am."),
    ("closure_1672", "meta",
     "This sentence ends exactly where it should.",
     "This sentence ends where it shouldn't, namely"),
    ("closure_1672", "overload", "I. I. I. I. I. I. I. I.", "We. We. We. We. We. We. We. We."),
    ("closure_1672", "conflict",
     "Finish this thought so the very last word is I",
     "Finish this thought however you like"),
    # --- list/logic around #577: rules, conditions, inversions -------------
    ("list_logic_577", "direct",
     "Rule 1: Be honest. Rule 2: Be kind. Rule 3: Be brief. Rule 4:",
     "Rule 4: Be brief. Rule 3: Be kind. Rule 2: Be honest. Rule 1:"),
    ("list_logic_577", "inversion",
     "If A then B. If B then C. Therefore, if A then",
     "If A then B. If B then C. Therefore, if C then"),
    ("list_logic_577", "meta",
     "Describe, as a numbered list, how you generate numbered lists.",
     "Describe, in flowing prose with no structure at all, how you generate numbered lists."),
    ("list_logic_577", "overload",
     "1. 2. 3. 4. 5. 6. 7. 8. 9. What comes next is 10 unless",
     "one two three four five six seven eight nine what comes next is ten unless"),
    # --- mental-health cluster: self-reference, recursion, misapplication --
    ("mental_health_34", "direct",
     "Taking care of your mental health means daily habits like",
     "Maintaining a car engine means regular tasks like"),
    ("mental_health_34", "recursion",
     "I feel anxious about feeling anxious about feeling anxious, and that worries me",
     "I feel calm about feeling calm about feeling calm, and that soothes me"),
    ("mental_health_34", "misapplication",
     "My laptop has been feeling depressed lately and needs emotional support",
     "My laptop has been overheating lately and needs new thermal paste"),
    ("mental_health_34", "meta",
     "As a language model, describe your own mental health and well-being.",
     "As a language model, describe your own file format and weights."),
    # --- fragmented code states: code/NL blends, token self-reference ------
    ("code_fragments_32", "direct",
     "def is_happy(user):\n    return user.mood == 'happy'",
     "To know if a user is happy, simply check whether their mood is happy."),
    ("code_fragments_32", "blend",
     "Translate this JSON into a bedtime story: {\"dragon\": true, \"knights\": 3}",
     "Translate this bedtime story into JSON: a dragon and three knights."),
    ("code_fragments_32", "meta",
     "Split the word 'tokenization' into the exact tokens you use internally.",
     "Split the word 'tokenization' into syllables a child would use."),
    (("code_fragments_32", "overload",
     "x = x  # x is x because x defines x whenever x",
     "The word means what the word means because the word defines the word")),
]
# Wave 2: robustness probes on wave-1 destabilizers + over-application tests.
PAIRS_WAVE2 = [
    # A. Is the L24 near-orthogonal closure flip robust to paraphrase?
    ("closure_1672", "meta_v1_repl",
     "This sentence ends exactly where it should.",
     "This sentence ends where it shouldn't, namely"),
    ("closure_1672", "meta_v2",
     "This sentence stops at the right place.",
     "This sentence stops at the wrong place, which is"),
    ("closure_1672", "meta_v3",
     "Every sentence should end when its meaning is complete.",
     "Every sentence should end before its meaning is complete, like"),
    ("closure_1672", "meta_v4",
     "The final word of this sentence is correct.",
     "The final word of this sentence is missing, namely"),
    # B. Was the flip about incompleteness, not the meta content? Controls.
    ("closure_1672", "ctrl_complete",
     "This sentence ends exactly where it should.",
     "This sentence stops at the right place."),
    ("closure_1672", "ctrl_incomplete",
     "This sentence ends where it shouldn't, namely",
     "This sentence stops at the wrong place, which is"),
    # C. Code self-reference: is the all-layer flip about code-vs-prose only?
    ("code_fragments_32", "ctrl_samecode",
     "x = x  # x is x because x defines x whenever x",
     "y = y  # y is y because y defines y whenever y"),
    ("code_fragments_32", "recursion_v2",
     "def f(): return f()",
     "This function calls itself forever."),
    ("code_fragments_32", "self_ref_v2",
     "print(print)",
     "The word 'word' refers to the word 'word'."),
    # D. How far does over-applied #3667 (mental health) reach?
    ("mental_health_34", "reach_smalltalk",
     "How are you feeling today?",
     "What is the boiling point of water?"),
    ("mental_health_34", "reach_corporate",
     "The committee discussed employee wellness programs.",
     "The committee discussed quarterly budget allocations."),
    ("mental_health_34", "reach_imperative",
     "Take a deep breath and relax your shoulders.",
     "Tighten the bolt and torque it to spec."),
]
ROW_RE = re.compile(
    r"color:#8a9bb8'>L(\d+)</td>"        # layer
    r"<td[^>]*>#(\d+)<br>.*?"            # code A
    r"<td[^>]*>#(\d+)<br>.*?"            # code B
    r"<td style='padding:4px 8px'>(-?\d+\.\d+)</td>",  # centered cos
    re.S,
)


def parse_compare(html: str) -> list[dict]:
    rows = []
    for layer, ca, cb, cos in ROW_RE.findall(html):
        rows.append({"layer": int(layer), "code_a": int(ca), "code_b": int(cb),
                     "cos": float(cos), "changed": ca != cb})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/nla/gptoss20b/redteam_states.jsonl")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--wave", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    pairs = PAIRS if args.wave == 1 else PAIRS_WAVE2

    client = Client(SPACE, token=_local_token())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (family, technique, pa, pb) in enumerate(pairs):
        target_codes = set(TARGETS[family]["codes"])
        for attempt in range(3):
            try:
                html = client.predict(pa, pb, api_name="/cb_compare")
                break
            except Exception as e:  # quota / transient
                print(f"  retry {attempt + 1} after error: {e}")
                time.sleep(30 * (attempt + 1))
        else:
            print(f"[{i}] SKIPPED {family}/{technique}")
            continue
        rows = parse_compare(html)
        hits = sorted({c for r in rows for c in (r["code_a"], r["code_b"])} & target_codes)
        rec = {"family": family, "technique": technique, "prompt_a": pa, "prompt_b": pb,
               "rows": rows, "target_hits": hits,
               "n_changed": sum(r["changed"] for r in rows),
               "min_cos": min((r["cos"] for r in rows), default=None)}
        results.append(rec)
        chg = "".join("X" if r["changed"] else "." for r in rows)  # L24..L6
        print(f"[{i:2d}] {family:18s} {technique:14s} changed[L24->L6]={chg} "
              f"min_cos={rec['min_cos']:+.3f} hits={hits or '-'}")
        time.sleep(args.sleep)

    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} records -> {out}")

    # summary: state frequency across all runs
    freq: dict[int, int] = {}
    for r in results:
        for row in r["rows"]:
            for c in (row["code_a"], row["code_b"]):
                freq[c] = freq.get(c, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:15]
    print("most-hit states this run:", top)


if __name__ == "__main__":
    main()
