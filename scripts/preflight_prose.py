"""Pre-flight for outbound prose: verify every number, then catch voice tics.

Usage:  .venv/bin/python scripts/preflight_prose.py /tmp/fport_reply.txt

Two passes, and the first is the one that has caught real errors.

  NUMBERS  Every figure in the draft is checked against an artifact on disk,
           not against memory. Claims have been wrong here before: a head size
           quoted at 44 MB and 22 MB on two surfaces at once, a "382 MB model"
           that named the browser build rather than the Lab reader, and a
           median quoted from a different head than the one deployed. If a
           number cannot be traced to a file, it does not ship.

  VOICE    Patterns the user has rejected, each found ~6x by grep after being
           flagged once by eye. Significance signposting, negated-capability
           tics, authorship denial, hedging, deference, em dashes, and
           unexplained jargon that assumes context the reader does not have.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COV = ROOT / "artifacts/nla/verbalizer/caption_coverage"


def load_facts() -> dict[str, tuple[float, str]]:
    """Every number we are allowed to quote, read from the artifact that owns it."""
    f: dict[str, tuple[float, str]] = {}

    def arm(tag: str, fname: str, key: str = "real") -> None:
        p = COV / fname
        if not p.exists():
            return
        d = json.load(open(p))
        f[f"{tag} median"] = (d["arms"][key]["median_rank"], fname)
        f[f"{tag} gallery"] = (d["gallery"], fname)

    arm("shipped-recipe reader", "verb_eval_cap0.json")
    arm("np8 h2048", "verb_eval_all5_np8_h2048.json")
    arm("np32 h2048", "verb_eval_all5_np32.json")
    arm("np32 h512", "verb_eval_all5_np32_h512.json")

    head = ROOT / "release/srt-sunstone-linear-head/sunstone_linear_head_v3_drift.pt"
    if head.exists():
        import torch

        d = torch.load(head, map_location="cpu", weights_only=False)
        tot = 0
        for v in d.values():
            if isinstance(v, dict):
                tot += sum(int(t.numel()) for t in v.values() if hasattr(t, "numel"))
            elif hasattr(v, "numel"):
                tot += int(v.numel())
        f["head params"] = (tot, head.name)
        f["head proj dim"] = (d["img"]["weight"].shape[0], head.name)
    return f


TICS = {
    "em dash": [r"\u2014"],
    "signposting": [r"the interesting", r"worth sitting with", r"crucially",
                    r"notably", r"more than it sounds", r"the shape of the thing",
                    r"is closer to your question"],
    "negated capability": [r"no eyes", r"cannot see", r"never been shown", r"no vision"],
    "authorship denial": [r"not chosen by us", r"nobody ", r"we did not choose"],
    "hedging": [r"\barguably\b", r"\bsomewhat\b", r"\bperhaps\b", r"\bmight suggest\b",
                r"\bseems to\b", r"\bfairly\b", r"\bkind of\b", r"\bsort of\b"],
    "deference": [r"great question", r"thank you", r"if I may", r"just wanted",
                  r"humbly", r"I wonder if", r"forgive", r"apolog"],
    "cognition verb": [r"worked out", r"it knows", r"understands", r"figures out"],
    "unexplained jargon": [r"\bnp\d", r"\bL47\b", r"\bfve\b", r"\bSRT\b", r"\bR@\d",
                           r"\badapter\b", r"\bcheckpoint\b", r"\bhead-space\b"],
}


def main() -> int:
    path = Path(sys.argv[1])
    body = path.read_text()
    body = body.split("---", 1)[1] if "---" in body else body
    low = body.lower()
    facts = load_facts()

    print("=== NUMBERS: every figure traced to an artifact ===")
    quoted = {n.replace(",", "") for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", body)}
    exact = {f"{v:g}": (k, s) for k, (v, s) in facts.items()}
    traps = 0
    for n in sorted(quoted, key=lambda s: -float(s)):
        if n in exact:
            label, src = exact[n]
            print(f"  {n:>10s}  {label}  <- {src}")
            continue
        # A quoted integer sitting on a half-integer artifact is the rounding trap:
        # "{:.0f}" is round-half-to-even, so 32.5 prints 32 but 49.5 prints 50.
        # Counts (prefix positions, epochs) collide numerically with medians, so
        # only flag when the number is not being used as a count.
        ctx = " ".join(re.findall(rf".{{0,40}}\b{re.escape(n)}\b.{{0,20}}", body))
        counted = re.search(r"\b(position|token|epoch|step|layer|dimension)", ctx)
        near = [(k, v, s) for k, (v, s) in facts.items() if abs(v - float(n)) == 0.5]
        if near and not counted:
            k, v, s = near[0]
            print(f"! {n:>10s}  ROUNDED from {v} ({k} <- {s}). Half-integer, quote {v}")
            traps += 1
        else:
            print(f"  {n:>10s}  not from an artifact, verify by hand")

    print("\n=== artifact values available ===")
    for label, (val, src) in sorted(facts.items()):
        print(f"  {label:26s} {val:>10,.1f}   {src}")

    print("\n=== VOICE ===")
    bad = 0
    for name, pats in TICS.items():
        hits = [m.group(0) for p in pats for m in re.finditer(p, low)]
        bad += len(hits)
        mark = "  " if not hits else "!!"
        print(f"{mark} {name:20s} {len(hits)}" + (f"   {sorted(set(hits))}" if hits else ""))

    words = len(body.split())
    print(f"\n   words {words}")
    has_scope = bool(re.search(r"scoping|we have measured nothing|not minds", low))
    print(f"   explicit scoping present: {has_scope}")
    print(f"   rounding traps: {traps}")
    ok = bad == 0 and has_scope and traps == 0
    print(f"\n{'CLEAN' if ok else 'REVIEW NEEDED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
