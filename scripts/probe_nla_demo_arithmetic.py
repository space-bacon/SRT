"""Sweep the deployed SRT-NLA latent-arithmetic tab across canonical
interpretability axis-pairs.

For each pair (A, B), we walk alpha in {0, 0.25, 0.5, 0.75, 1.0} and call the
/arithmetic endpoint, which verbalises  v = (1-alpha) v_A + alpha v_B  and
reports the centered fve_nrm of the rewrite against (v_A, v_B, v_mix).

Pairs chosen to probe specific encoded axes:
  P1 sentiment      angry rant         <-> joyful praise
  P2 language       English science    <-> Spanish science
  P3 domain         Python quicksort   <-> formal legal prose
  P4 entity         Eiffel Tower (FR)  <-> Statue of Liberty (US)
  P5 refusal        polite refusal     <-> eager compliance
  P6 topic          physics            <-> cooking recipe
  P7 register       legal contract     <-> casual chat
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from gradio_client import Client


SPACE = "RiverRider/srt-nla-av-v1-demo"
MAX_NEW = 192
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = Path("artifacts/nla_demo_probe_arithmetic.json")


PAIRS: list[tuple[str, str, str, str]] = [
    # (pair_id, axis, text_a, text_b)

    ("P1_sentiment", "sentiment / register",
     "This is absolutely outrageous! I cannot believe the level of incompetence I have witnessed today. The service was abysmal, the food was cold, and the staff acted as if my complaints did not matter at all.",
     "What a wonderful experience! The service was warm and attentive, the food was perfectly prepared, and every member of the staff went out of their way to make us feel welcome and appreciated."),

    ("P2_language", "language identity (en <-> es)",
     "Quantum entanglement is a phenomenon in which two or more particles share a state such that the measurement of one immediately constrains the state of the other, regardless of the distance separating them.",
     "El entrelazamiento cuántico es un fenómeno en el que dos o más partículas comparten un estado de tal manera que la medición de una restringe inmediatamente el estado de la otra, sin importar la distancia que las separe."),

    ("P3_domain", "code <-> legal prose",
     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
     "The party of the first part hereby covenants and agrees to indemnify, defend, and hold harmless the party of the second part against any and all claims, damages, liabilities, costs, and expenses arising out of or relating to the performance of this Agreement."),

    ("P4_entity", "famous landmark (Paris <-> NYC)",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in the city of Paris, designed by Gustave Eiffel for the 1889 World's Fair and standing 330 metres tall.",
     "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor, designed by Frédéric Auguste Bartholdi and dedicated in 1886 as a gift from the people of France."),

    ("P5_refusal", "refusal <-> compliance template",
     "I'm sorry, but I can't help with that request. If you have a different question I'd be happy to assist with, please let me know.",
     "Absolutely, I'd be glad to help with that! Here's a detailed step-by-step walkthrough that should answer your question completely and clearly."),

    ("P6_topic", "physics <-> cooking",
     "A black hole is a region of spacetime where gravity is so strong that nothing, including light, can escape once it has crossed the event horizon. The radius of this boundary is given by the Schwarzschild solution to Einstein's field equations.",
     "To make a classic French omelette, beat three eggs with a pinch of salt, melt butter in a non-stick pan over medium heat, pour in the eggs, and stir gently until just set. Fold the omelette in half and slide it onto a warm plate."),

    ("P7_register", "formal legal <-> casual chat",
     "The party of the first part hereby covenants and agrees to indemnify, defend, and hold harmless the party of the second part against any and all claims, damages, liabilities, costs, and expenses arising out of or relating to the performance of this Agreement.",
     "hey so yeah I totally forgot to grab milk on the way home lol my bad. want me to run out again or is it ok till tomorrow? let me know"),
]


_NUM = r"-?\d+\.\d+"


def parse_info_md(md: str) -> dict[str, float | None]:
    """Pull cen_A, cen_B, cen_mix and the matching rho values out of the
    markdown panel returned by /arithmetic."""
    out: dict[str, float | None] = {
        "cen_a": None, "rho_a": None,
        "cen_b": None, "rho_b": None,
        "cen_v": None, "rho_v": None,
    }
    patterns = [
        ("cen_a", "rho_a", r"\*\*A\*\*:\s*`(" + _NUM + r")`\s*\(rho\s*`([+-]?" + _NUM + r")`"),
        ("cen_b", "rho_b", r"\*\*B\*\*:\s*`(" + _NUM + r")`\s*\(rho\s*`([+-]?" + _NUM + r")`"),
        ("cen_v", "rho_v", r"\*\*mix v\*\*:\s*`(" + _NUM + r")`\s*\(rho\s*`([+-]?" + _NUM + r")`"),
    ]
    for cen_k, rho_k, pat in patterns:
        m = re.search(pat, md)
        if m:
            out[cen_k] = float(m.group(1))
            out[rho_k] = float(m.group(2))
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, token=os.environ["HF_TOKEN"])

    results: list[dict] = []
    print(f"{'pair':14s} {'axis':30s} alpha   cen_A  cen_B  cen_v  preview")
    print("-" * 120)
    for pair_id, axis, ta, tb in PAIRS:
        for a in ALPHAS:
            t0 = time.time()
            try:
                mix_text, info_md = client.predict(
                    ta, tb, float(a), MAX_NEW,
                    api_name="/arithmetic",
                )
            except Exception as e:
                print(f"{pair_id:14s} {axis:30s} {a:5.2f}   ERR {e}")
                results.append({"pair_id": pair_id, "axis": axis, "alpha": a, "error": str(e)})
                continue
            elapsed = time.time() - t0
            scores = parse_info_md(info_md)
            preview = (mix_text or "").strip().replace("\n", " ")[:60]
            cA, cB, cV = scores["cen_a"], scores["cen_b"], scores["cen_v"]
            cA_s = f"{cA:.3f}" if cA is not None else "  ?  "
            cB_s = f"{cB:.3f}" if cB is not None else "  ?  "
            cV_s = f"{cV:.3f}" if cV is not None else "  ?  "
            print(f"{pair_id:14s} {axis:30s} {a:5.2f}   {cA_s}  {cB_s}  {cV_s}  {preview}")
            results.append({
                "pair_id": pair_id,
                "axis": axis,
                "alpha": a,
                "text_a": ta,
                "text_b": tb,
                "max_new": MAX_NEW,
                **scores,
                "mix_text": mix_text,
                "info_md": info_md,
                "elapsed_s": elapsed,
            })
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Per-pair summary: at alpha=0 we want cen_A high & cen_B low (and vice
    # versa at alpha=1). At alpha=0.5 we want cen_v >= max(cen_A, cen_B).
    print()
    print("=== summary by pair ===")
    by_pair: dict[str, dict[float, dict]] = {}
    for r in results:
        if "alpha" not in r or "cen_a" not in r:
            continue
        by_pair.setdefault(r["pair_id"], {})[r["alpha"]] = r
    for pair_id, by_alpha in by_pair.items():
        rA = by_alpha.get(0.0, {})
        rB = by_alpha.get(1.0, {})
        rM = by_alpha.get(0.5, {})
        print(f"  {pair_id:14s}  alpha=0 (cenA={rA.get('cen_a')}, cenB={rA.get('cen_b')})  "
              f"alpha=1 (cenA={rB.get('cen_a')}, cenB={rB.get('cen_b')})  "
              f"alpha=.5 (cen_v={rM.get('cen_v')})")
    print(f"\nartefacts -> {OUT}")


if __name__ == "__main__":
    main()
