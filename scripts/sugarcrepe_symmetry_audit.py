"""Count SugarCrepe pairs whose positive and negative are the same bag of words.

On the swap_* splits most items differ only in word order, so a model with an
order-insensitive text representation scores chance there no matter what it
represents. That pins a whole class of models to 0.5 by construction, which is
worth knowing before reading a swap score as a capability measurement.

Counts only. Nothing here evaluates a model.

    python scripts/sugarcrepe_symmetry_audit.py --data sugarcrepe
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel",
          "swap_att", "swap_obj"]


def bag(s: str) -> collections.Counter:
    return collections.Counter(re.findall(r"[a-z0-9']+", s.lower()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sugarcrepe")
    ap.add_argument("--out", default="artifacts/nla/q4/sugarcrepe_symmetry_audit.json")
    a = ap.parse_args()

    res, tot, tot_sym = {}, 0, 0
    for split in SPLITS:
        p = Path(a.data) / f"{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        items = list(d.values()) if isinstance(d, dict) else d
        n = sym = 0
        examples = []
        for it in items:
            pos = it.get("caption") or it.get("pos")
            neg = it.get("negative_caption") or it.get("neg")
            if not pos or not neg:
                continue
            n += 1
            if bag(pos) == bag(neg):
                sym += 1
                if len(examples) < 3:
                    examples.append({"positive": pos, "negative": neg})
        res[split] = {"n": n, "bag_identical": sym,
                      "frac": round(sym / n, 4) if n else None,
                      "pct": round(100 * sym / n, 1) if n else None,
                      "examples": examples}
        tot += n
        tot_sym += sym
        print(f"  {split:<12} n={n:>5}  bag-identical {sym:>4}  "
              f"({100 * sym / max(n, 1):5.1f}%)", flush=True)

    out = {
        "question": "how many SugarCrepe pairs differ only in word order",
        "method": "lowercase, tokenise on [a-z0-9']+, compare multisets of the "
                  "positive and negative caption",
        "why_it_matters": "for a bag-identical pair, any model whose text "
                          "representation ignores word order assigns both "
                          "captions the same score, so it lands at chance by "
                          "construction. On the swap_* splits such a model "
                          "cannot exceed chance however good its content "
                          "representation is, and a swap score therefore does "
                          "not separate content quality across that class of "
                          "models.",
        "totals": {"n": tot, "bag_identical": tot_sym,
                   "frac": round(tot_sym / tot, 4) if tot else None},
        "splits": res,
        "scope": "this is a property of the benchmark data alone. No model was "
                 "run. It is not a defect: the swap splits are meant to probe "
                 "word order. It is a caveat about how their scores read.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
