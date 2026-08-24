"""Does a head-space axis steer real retrieval, or is it a decorated random walk?

The runtime exposes steering as a vector addition on the projected query. That
is only worth shipping if the direction carries meaning, so this tests the
claim the way it would fail if it were false.

Design:

  index    real images, projected through the image head
  axis     built from CAPTION vectors, mean(class A) - mean(class B)
  query    image vectors, steered by the axis at a range of alphas
  measure  fraction of the top-k retrieved images whose own gold caption is
           class A
  control  matched-norm random axes, many seeds, same measurement

Two properties make this a test rather than a demonstration. The axis is
defined in the TEXT modality and applied to IMAGE queries, so a positive
result is a claim about a shared space and not about memorised nearest
neighbours. And the captions used to build the axis are held out from the
images being evaluated, so the axis cannot contain its own answer.

Class lift alone would not settle it. A large enough alpha makes the axis
outweigh the query, at which point every query returns the same class-A
images because the query has been erased rather than steered. So retention of
the unsteered neighbourhood is measured alongside lift, and the operating
point is the largest alpha that still keeps the query in charge.

Classes come from keyword rules over COCO captions, which are crude but
objective, and crucially are never seen by the axis at evaluation time.

    python scripts/headspace_axis_validation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]

# Keyword contrasts. Chosen to be visually grounded and mutually exclusive, so
# an image is class A, class B, or neither, and "neither" is discarded.
CONTRASTS = {
    "animal_vs_vehicle": (
        {"dog", "cat", "horse", "sheep", "cow", "bear", "zebra", "giraffe",
         "elephant", "bird", "puppy", "kitten", "cattle", "goat"},
        {"car", "truck", "bus", "train", "motorcycle", "bicycle", "airplane",
         "plane", "boat", "van", "taxi", "scooter", "tractor"},
    ),
    "food_vs_sport": (
        {"pizza", "sandwich", "cake", "banana", "broccoli", "carrot", "donut",
         "salad", "plate", "meal", "fruit", "vegetables", "bread", "soup"},
        {"skateboard", "surfboard", "tennis", "baseball", "skis", "snowboard",
         "frisbee", "racket", "skiing", "surfing", "skating", "soccer"},
    ),
    "indoor_vs_outdoor": (
        {"kitchen", "bathroom", "bedroom", "living", "couch", "sofa", "toilet",
         "sink", "desk", "refrigerator", "counter", "indoors", "room"},
        {"beach", "street", "field", "mountain", "park", "sky", "grass",
         "ocean", "forest", "road", "snow", "outdoors", "sidewalk"},
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path,
                   default=ROOT / "checkpoints/gemma4_readout/qwen3b_v6_head.pt")
    p.add_argument("--img", type=Path,
                   default=ROOT / "artifacts/nla/q4/qwen3b_caches/sc_img_states_qwen3b.npz")
    p.add_argument("--txt", type=Path,
                   default=ROOT / "artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0])
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--seeds", type=int, default=32, help="random control draws")
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts/nla/q4/headspace_axis_validation.json")
    return p.parse_args()


def unit(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def label(text: str, a: set[str], b: set[str]) -> int | None:
    w = set(text.lower().replace(".", " ").replace(",", " ").split())
    ha, hb = bool(w & a), bool(w & b)
    if ha == hb:
        return None
    return 0 if ha else 1


def main() -> None:
    args = parse_args()
    head = torch.load(args.head, map_location="cpu", weights_only=True)
    Wi, bi, mi = (head["img"]["weight"].float().numpy(),
                  head["img"]["bias"].float().numpy(),
                  head["mu_img"].float().numpy())
    Wt, bt, mt = (head["txt"]["weight"].float().numpy(),
                  head["txt"]["bias"].float().numpy(),
                  head["mu_txt"].float().numpy())

    iz = np.load(args.img, allow_pickle=True)
    img_state = {str(f): iz["states"][i].astype(np.float32)
                 for i, f in enumerate(iz["files"])}
    tz = np.load(args.txt, allow_pickle=True)
    ts, tn = tz["states"], tz["texts"]
    txt_state = {str(t): ts[i].astype(np.float32) for i, t in enumerate(tn)}

    pairs: dict[str, str] = {}
    for s in SPLITS:
        with open(ROOT / "sugarcrepe" / f"{s}.json") as fh:
            for it in json.load(fh).values():
                if it["filename"] in img_state and it["caption"] in txt_state:
                    pairs.setdefault(it["filename"], it["caption"])

    files = sorted(pairs)
    Zi = unit((np.stack([img_state[f] for f in files]) - mi) @ Wi.T + bi)

    results = {"head": args.head.name, "n_images": len(files),
               "topk": args.topk, "seeds": args.seeds, "contrasts": {}}

    for name, (kw_a, kw_b) in CONTRASTS.items():
        img_lab = np.array([label(pairs[f], kw_a, kw_b) if label(pairs[f], kw_a, kw_b) is not None
                            else -1 for f in files])
        n_a = int((img_lab == 0).sum())
        n_b = int((img_lab == 1).sum())
        if n_a < 20 or n_b < 20:
            results["contrasts"][name] = {"skipped": f"too few images ({n_a}/{n_b})"}
            continue

        # Axis captions: drawn from the caption pool, excluding every caption
        # that is the gold answer of an image we will evaluate.
        held_out = set(pairs.values())
        cap_a, cap_b = [], []
        for t in txt_state:
            if t in held_out:
                continue
            lb = label(t, kw_a, kw_b)
            if lb == 0:
                cap_a.append(t)
            elif lb == 1:
                cap_b.append(t)
        if len(cap_a) < 20 or len(cap_b) < 20:
            results["contrasts"][name] = {"skipped": f"too few axis captions ({len(cap_a)}/{len(cap_b)})"}
            continue

        Za = unit((np.stack([txt_state[t] for t in cap_a]) - mt) @ Wt.T + bt)
        Zb = unit((np.stack([txt_state[t] for t in cap_b]) - mt) @ Wt.T + bt)
        axis = Za.mean(0) - Zb.mean(0)
        axis /= np.linalg.norm(axis) + 1e-8

        # Queries: class-B images. If the axis means anything, pushing a
        # class-B query along it should pull class-A images into the results.
        q_idx = np.where(img_lab == 1)[0]
        Q = Zi[q_idx]

        def topk_of(direction: np.ndarray, alpha: float) -> np.ndarray:
            Qs = unit(Q + alpha * direction)
            S = Qs @ Zi.T
            S[np.arange(len(q_idx)), q_idx] = -np.inf  # never retrieve the query itself
            return np.argpartition(-S, args.topk, axis=1)[:, : args.topk]

        def purity(direction: np.ndarray, alpha: float) -> float:
            return float((img_lab[topk_of(direction, alpha)] == 0).mean())

        base_top = topk_of(axis, 0.0)
        base_sets = [set(r) for r in base_top]

        def retention(direction: np.ndarray, alpha: float) -> float:
            """Share of the unsteered neighbourhood still present after steering.

            Near 0 means the query was replaced by the axis, not steered by it.
            """
            top = topk_of(direction, alpha)
            return float(np.mean([len(base_sets[i] & set(top[i])) / args.topk
                                  for i in range(len(top))]))

        rng = np.random.default_rng(0)
        rows = []
        for alpha in args.alphas:
            real = purity(axis, alpha)
            keep = retention(axis, alpha)
            ctrl, ctrl_keep = [], []
            for _ in range(args.seeds):
                r = rng.standard_normal(axis.shape[0]).astype(np.float32)
                r /= np.linalg.norm(r)
                ctrl.append(purity(r, alpha))
                ctrl_keep.append(retention(r, alpha))
            ctrl = np.array(ctrl)
            sd = float(ctrl.std())
            rows.append({
                "alpha": alpha,
                "class_a_in_topk": round(real, 4),
                "random_mean": round(float(ctrl.mean()), 4),
                "random_sd": round(sd, 4),
                "z_vs_random": round((real - float(ctrl.mean())) / sd, 2) if sd > 1e-9 else None,
                "retention": round(keep, 4),
                "random_retention": round(float(np.mean(ctrl_keep)), 4),
            })
            print(f"{name:20s} a={alpha:<5} real={real:.4f} "
                  f"random={ctrl.mean():.4f}+-{sd:.4f} "
                  f"z={rows[-1]['z_vs_random']} "
                  f"retention={keep:.3f} (random {np.mean(ctrl_keep):.3f})")

        base = rows[0]["class_a_in_topk"]
        best = max(rows, key=lambda r: r["class_a_in_topk"])
        # Calibrated point: strongest steering that still keeps most of the
        # query's own neighbourhood. Reporting only `best` would be reporting
        # the alpha at which the query stopped mattering.
        usable = [r for r in rows if r["retention"] >= 0.5 and r["alpha"] > 0]
        operating = max(usable, key=lambda r: r["class_a_in_topk"]) if usable else None
        results["contrasts"][name] = {
            "n_class_a_images": n_a,
            "n_class_b_images": n_b,
            "n_axis_captions": [len(cap_a), len(cap_b)],
            "baseline_class_a_in_topk": base,
            "best_alpha": best["alpha"],
            "best_class_a_in_topk": best["class_a_in_topk"],
            "lift": round(best["class_a_in_topk"] - base, 4),
            "operating_point": operating,
            "dose_response": rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
