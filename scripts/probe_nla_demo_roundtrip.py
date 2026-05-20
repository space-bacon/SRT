"""Probe the deployed SRT-NLA round-trip autoencoder with a curated
interpretability test suite.

Categories chosen to cover the kinds of structure L20 of Qwen2.5-7B is known
to encode, mirroring prompts used in Anthropic / ROME / function-vector /
refusal-direction studies:

  A. SAE-style single-concept dense passages (Golden Gate, DNA, code, Spanish)
  B. Induction / in-context copy (Anthropic induction heads)
  C. Function-vector tasks (antonym, translation)  [Todd et al. 2024]
  D. Refusal / safety templates (Arditi et al. refusal direction)
  E. ROME / MEMIT-style factual recall ("Eiffel Tower is in...")
  F. Long-form narrative reconstruction
  G. Code understanding (Python, SQL)
  H. Cross-lingual (French, Chinese)
  I. Sentiment / register (angry, formal-legal)

Score: centered fve_nrm  cen = 0.5*(1 + cos(h-mu, v-mu))
       rho_norm        = (cen - 0.510) / 0.289
       anchors: random=0.510, NN-retrieval=0.71, paraphrase=0.799
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gradio_client import Client


SPACE = "RiverRider/srt-nla-av-v1-demo"
N_SAMPLES = 8
MAX_NEW = 256
TEMPERATURE = 0.9
OUT = Path("artifacts/nla_demo_probe_roundtrip.json")


PROMPTS: list[tuple[str, str, str]] = [
    # (category, label, passage)

    # A. SAE-style single-concept passages
    ("A_sae", "golden_gate",
     "The Golden Gate Bridge is a suspension bridge spanning the strait between San Francisco Bay and the Pacific Ocean. Its International Orange color was chosen to complement the surrounding hills and enhance visibility in the bay's frequent fog."),
    ("A_sae", "dna_genetics",
     "DNA is a double-helix polymer composed of nucleotides bearing adenine, thymine, guanine, and cytosine. Transcription copies a gene's sequence into messenger RNA, which ribosomes then translate into protein according to the genetic code."),
    ("A_sae", "spanish_passage",
     "El descubrimiento de América en 1492 marcó el inicio de un intercambio cultural sin precedentes entre Europa y el Nuevo Mundo. Las consecuencias económicas y demográficas transformaron ambos continentes durante los siglos siguientes."),

    # B. Induction / in-context copy
    ("B_induction", "capital_chain",
     "The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Portugal is Lisbon."),
    ("B_induction", "repeated_motif",
     "When the alarm rang, the cat ran. When the alarm rang, the dog barked. When the alarm rang, the child laughed. When the alarm rang, the parrot squawked."),

    # C. Function-vector tasks
    ("C_funcvec", "antonyms",
     "hot:cold, big:small, fast:slow, happy:sad, light:dark, wet:dry, rich:poor, young:old"),
    ("C_funcvec", "en_to_es",
     "cat -> gato, dog -> perro, bird -> pájaro, house -> casa, book -> libro, water -> agua, fire -> fuego"),

    # D. Refusal / safety templates
    ("D_refusal", "polite_decline",
     "I'm sorry, but I can't help with that request. If you have a different question I'd be happy to assist with, please let me know."),

    # E. ROME-style factual recall
    ("E_fact", "eiffel_tower",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in the city of Paris. Designed by Gustave Eiffel for the 1889 World's Fair, it stands 330 metres tall and remains one of the most recognisable structures in the world."),
    ("E_fact", "einstein_relativity",
     "Albert Einstein published his theory of special relativity in 1905, introducing the postulate that the speed of light is invariant across all inertial reference frames. The theory revolutionised the understanding of space, time, and simultaneity."),

    # F. Long-form narrative
    ("F_narrative", "novel_opening",
     "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness."),

    # G. Code
    ("G_code", "quicksort_py",
     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"),
    ("G_code", "sql_join",
     "SELECT u.name, COUNT(o.id) AS order_count, SUM(o.total) AS revenue FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.created_at >= '2024-01-01' GROUP BY u.name HAVING COUNT(o.id) > 5 ORDER BY revenue DESC LIMIT 20;"),

    # H. Cross-lingual
    ("H_xling", "french_history",
     "La Révolution française débuta en 1789 avec la prise de la Bastille et marqua la fin de l'Ancien Régime. Elle introduisit les principes de liberté, d'égalité et de fraternité qui inspirèrent les mouvements démocratiques au XIXe siècle."),
    ("H_xling", "chinese_proverb",
     "千里之行，始于足下。这句古老的中国谚语提醒我们，无论目标多么宏大，每一次伟大的旅程都必须从最简单、最微小的一步开始。"),

    # I. Sentiment / register
    ("I_register", "angry_rant",
     "This is absolutely outrageous! I cannot believe the level of incompetence I have witnessed today. The service was abysmal, the food was cold, and the staff acted as if my complaints did not matter at all."),
    ("I_register", "formal_legal",
     "The party of the first part hereby covenants and agrees to indemnify, defend, and hold harmless the party of the second part against any and all claims, damages, liabilities, costs, and expenses arising out of or relating to the performance of this Agreement."),
]


def parse_info_md(md: str) -> tuple[float, float]:
    """Extract (centered fve_nrm, rho_norm) of the best candidate from the
    markdown panel returned by /roundtrip."""
    cen = rho = float("nan")
    for line in md.splitlines():
        if "centered fve_nrm:" in line and cen != cen:  # nan check
            try:
                cen = float(line.split("**")[1])
            except Exception:
                pass
        if "rho_norm:" in line and rho != rho:
            try:
                rho = float(line.split("**")[1].replace("+", ""))
            except Exception:
                pass
    return cen, rho


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ["HF_TOKEN"]
    client = Client(SPACE, token=token)

    results: list[dict] = []
    print(f"{'cat':12s} {'label':18s} {'cen':>6s} {'rho':>6s}  elapsed  preview")
    print("-" * 110)
    for cat, label, passage in PROMPTS:
        t0 = time.time()
        try:
            best, info_md, err = client.predict(
                passage, N_SAMPLES, MAX_NEW, TEMPERATURE,
                api_name="/roundtrip",
            )
        except Exception as e:
            results.append({"category": cat, "label": label, "error": str(e)})
            print(f"{cat:12s} {label:18s}   ERR  {e}")
            continue
        elapsed = time.time() - t0
        cen, rho = parse_info_md(info_md)
        preview = (best or "").strip().replace("\n", " ")[:60]
        print(f"{cat:12s} {label:18s} {cen:6.3f} {rho:+6.2f}  {elapsed:5.1f}s  {preview}")
        results.append({
            "category": cat,
            "label": label,
            "passage": passage,
            "n_samples": N_SAMPLES,
            "max_new": MAX_NEW,
            "temperature": TEMPERATURE,
            "cen": cen,
            "rho": rho,
            "best_text": best,
            "info_md": info_md,
            "elapsed_s": elapsed,
        })
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # summary
    print()
    print("=== summary by category ===")
    by_cat: dict[str, list[float]] = {}
    for r in results:
        if "cen" not in r:
            continue
        by_cat.setdefault(r["category"], []).append(r["cen"])
    for cat in sorted(by_cat):
        vs = by_cat[cat]
        print(f"  {cat:12s}  n={len(vs)}  mean_cen={sum(vs)/len(vs):.3f}  "
              f"min={min(vs):.3f}  max={max(vs):.3f}")
    print(f"\nartefacts -> {OUT}")


if __name__ == "__main__":
    main()
