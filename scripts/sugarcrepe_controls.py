"""Every SugarCrepe null in one place, with named arms and a single scorer.

Rounds 12 and 13 of the public review produced controls (blind bigram prior,
mean-image ablation) that were banked as JSON with no code path. This script
is the retrofit, plus the round-14 arms.

Arms:
  real            real images, real pairing (the published number)
  shuffle_pair    real images, trained head, pairing deranged  [round-14 ask:
                  the only null that is not blind to the towers]
  mean_image      image replaced by the split's mean image     [round-13]
  foreign_mean    image replaced by ANOTHER split's mean image [ours]
  random_dir      image slot replaced by a random head-space direction [ours]
  blind_bigram    no image, no head, no encoder                [round-12]

The mean_image/foreign_mean/random_dir arms all reduce to ranking the two
captions by cosine against one fixed direction, so they measure caption
typicality, not word order. foreign_mean and random_dir separate the two.
Candidate position exchange cannot: --exchange verifies that it is an identity
for an independent-scoring cosine.

Usage:
    python scripts/sugarcrepe_controls.py --cell gemma-img_x_qwen-txt
    python scripts/sugarcrepe_controls.py --cell qwen-img_x_qwen-txt --exchange
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def normalise(text: str) -> str:
    """Strip the provenance formatting: trailing period, leading capital."""
    t = text.strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    return (t[:1].lower() + t[1:]) if t else t


def words(text: str, strip_punct: bool) -> list[str]:
    """Whitespace split keeps `water.` and `water` apart, which is how the
    trailing period got inside the shipped prior's final bigram."""
    return (re.findall(r"[a-z0-9']+", text.lower()) if strip_punct
            else text.lower().split())

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["add_att", "add_obj", "replace_att", "replace_obj",
          "replace_rel", "swap_att", "swap_obj"]
# The five splits where a blind length rule gains nothing (reviewer round 11).
CLEAN5 = ["replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]

CELLS: dict[str, dict] = {
    "gemma-img_x_qwen-txt": {
        "head": "checkpoints/gemma4_readout/mixed_v6_head.pt",
        "img": "checkpoints/gemma4_readout/sc_img_slots_2x2.npz",
        "img_slot": 4,  # global mean; the encoder asserts cos>0.99 vs the L47 state
        "txt": "artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz",
    },
    # Same cell, but reading the original one-at-a-time image cache instead of
    # the batched 2x2 encoder's global-mean slot. The two differ by ~18 item
    # flips on 7,511 (the reviewer's measured re-encode floor).
    "gemma-img_x_qwen-txt_origproto": {
        "head": "checkpoints/gemma4_readout/mixed_v6_head.pt",
        "img": "sugarcrepe/img_states.npz",
        "img_slot": None,
        "txt": "artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz",
    },
    "qwen-img_x_qwen-txt": {
        "head": "checkpoints/gemma4_readout/qwen3b_v6_head.pt",
        "img": "artifacts/nla/q4/qwen3b_caches/sc_img_states_qwen3b.npz",
        "img_slot": None,
        "txt": "artifacts/nla/q4/qwen3b_caches/sc_txt_states_qwen3b.npz",
    },
    "qwen-img_x_gemma-txt": {
        "head": "checkpoints/gemma4_readout/cell4_v6_head.pt",
        "img": "artifacts/nla/q4/qwen3b_caches/sc_img_states_qwen3b.npz",
        "img_slot": None,
        "txt": "sugarcrepe/txt_states.npz",
    },
    # Rebuilt head: the original gemma/gemma v6 head existed neither locally nor
    # in the published artifact repo. Regenerated from the published pair chunks
    # plus a seed-0 rebuild of the v6 negatives (counts matched to the item).
    "gemma-img_x_gemma-txt": {
        "head": "checkpoints/gemma4_readout/gemma_v6_head.pt",
        "img": "sugarcrepe/img_states.npz",
        "img_slot": None,
        "txt": "sugarcrepe/txt_states.npz",
    },
}


def load_items(work_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for s in SPLITS:
        with open(work_dir / f"{s}.json") as f:
            out[s] = list(json.load(f).values())
    return out


def load_cell(cfg: dict, txt_override: str | None = None) -> tuple[dict, dict, dict]:
    head = torch.load(ROOT / cfg["head"], map_location="cpu", weights_only=True)
    h = {
        "Wi": head["img"]["weight"].float().numpy(),
        "bi": head["img"]["bias"].float().numpy(),
        "mi": head["mu_img"].float().numpy(),
        "Wt": head["txt"]["weight"].float().numpy(),
        "bt": head["txt"]["bias"].float().numpy(),
        "mt": head["mu_txt"].float().numpy(),
    }
    iz = np.load(ROOT / cfg["img"])
    states = iz["states"]
    if cfg["img_slot"] is not None:
        states = states[:, cfg["img_slot"], :]
    img = {f: states[i].astype(np.float32) for i, f in enumerate(iz["files"])}
    tz = np.load(ROOT / (txt_override or cfg["txt"]))
    # Bind once: indexing an NpzFile re-reads the whole array every time.
    tstates, tnames = tz["states"], tz["texts"]
    txt = {t: tstates[i].astype(np.float32) for i, t in enumerate(tnames)}
    return h, img, txt


def proj_img(v: np.ndarray, h: dict) -> np.ndarray:
    return (v - h["mi"]) @ h["Wi"].T + h["bi"]


def proj_txt(v: np.ndarray, h: dict) -> np.ndarray:
    return (v - h["mt"]) @ h["Wt"].T + h["bt"]


def unit(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def gather(items: list[dict], img: dict, txt: dict, key=lambda s: s):
    """Rows of (image state, positive text state, negative text state)."""
    vi, vp, vn = [], [], []
    for it in items:
        a, p, n = (img.get(it["filename"]), txt.get(key(it["caption"])),
                   txt.get(key(it["negative_caption"])))
        if a is None or p is None or n is None:
            continue
        vi.append(a)
        vp.append(p)
        vn.append(n)
    return np.stack(vi), np.stack(vp), np.stack(vn)


def acc(zi: np.ndarray, zp: np.ndarray, zn: np.ndarray) -> float:
    """Ties count 0.5."""
    zi, zp, zn = unit(zi), unit(zp), unit(zn)
    sp, sn = (zi * zp).sum(-1), (zi * zn).sum(-1)
    return float((sp > sn).mean() + 0.5 * (sp == sn).mean())


def exchange_is_tautological(zi, zp, zn) -> bool:
    """Candidate position exchange (reviewer round 14) cannot discriminate here.

    Each candidate is scored independently against the image and the two scores
    are compared, so the scorer carries no positional term and exchanging the
    candidates maps acc -> 1 - acc identically. The test only bites on a scorer
    that sees both candidates jointly, e.g. a cross-encoder."""
    return abs(acc(zi, zp, zn) + acc(zi, zn, zp) - 1.0) < 1e-9


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def bigram_prior(items_by_split: dict[str, list[dict]],
                 strip_punct: bool = False) -> dict[str, float]:
    """Reviewer's round-12 prior: mean log(count+1) per bigram over the
    benchmark's own positive captions, leave-one-out by caption TEXT (not by
    item id; 3,166 positives recur across splits and leaking them is worth
    ~5 points on swap_obj)."""
    counts: Counter[tuple[str, str]] = Counter()
    per_caption: dict[str, Counter] = {}
    for items in items_by_split.values():
        for it in items:
            c = it["caption"]
            if c in per_caption:
                continue
            toks = words(c, strip_punct)
            bg = Counter(zip(toks, toks[1:]))
            per_caption[c] = bg
            counts.update(bg)

    def score(text: str, held_out: str) -> float:
        toks = words(text, strip_punct)
        bgs = list(zip(toks, toks[1:]))
        if not bgs:
            return 0.0
        drop = per_caption.get(held_out, Counter())
        return float(np.mean([math.log(max(counts[b] - drop[b], 0) + 1)
                              for b in bgs]))

    out = {}
    for s, items in items_by_split.items():
        hit = 0.0
        for it in items:
            c, nc = it["caption"], it["negative_caption"]
            sp, sn = score(c, c), score(nc, c)
            hit += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
        out[s] = hit / len(items)
    return out


def caption_stats(texts: list[str], corpus: Counter,
                  strip_punct: bool = False) -> np.ndarray:
    """[n_words, mean unigram log-freq] per caption, for the typicality
    regression."""
    rows = []
    for t in texts:
        toks = words(t, strip_punct)
        rows.append([len(toks),
                     float(np.mean([math.log(corpus[w] + 1) for w in toks]))
                     if toks else 0.0])
    return np.asarray(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", choices=sorted(CELLS), required=True)
    ap.add_argument("--work-dir", default="sugarcrepe")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--exchange", action="store_true",
                    help="verify that candidate position exchange is an identity "
                         "for this scorer (it is; see exchange_is_tautological)")
    ap.add_argument("--txt", default=None,
                    help="override the cell's text-state cache")
    ap.add_argument("--normalise", action="store_true",
                    help="key states by the normalised caption and strip "
                         "punctuation in the blind prior; needs a cache "
                         "encoded from normalised text (round 16)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = CELLS[args.cell]
    items = load_items(Path(args.work_dir))
    h, img, txt = load_cell(cfg, args.txt)
    rng = np.random.default_rng(0)
    key = normalise if args.normalise else (lambda s: s)

    prior = bigram_prior(items, args.normalise)
    corpus = Counter(w for its in items.values() for it in its
                     for w in words(it["caption"], args.normalise))

    rows = {s: gather(items[s], img, txt, key) for s in SPLITS}
    split_mean_img = {s: proj_img(rows[s][0].mean(0, keepdims=True), h)
                      for s in SPLITS}

    res: dict[str, dict] = {}
    for s in SPLITS:
        vi, vp, vn = rows[s]
        n = len(vi)
        zi, zp, zn = proj_img(vi, h), proj_txt(vp, h), proj_txt(vn, h)
        mean_img = split_mean_img[s]

        arms: dict[str, float] = {"real": acc(zi, zp, zn)}

        # Round-14 ask: real images, trained head, pairing deranged. Unlike the
        # blind nulls this takes a different value in every cell.
        sh = []
        for k in range(args.seeds):
            g = np.random.default_rng(1000 + k)
            while True:
                perm = g.permutation(n)
                if n < 2 or not (perm == np.arange(n)).any():
                    break
            sh.append(acc(zi[perm], zp, zn))
        arms["shuffle_pair"] = float(np.mean(sh))
        arms["shuffle_pair_sd"] = float(np.std(sh))

        arms["mean_image"] = acc(np.repeat(mean_img, n, 0), zp, zn)

        fm = [acc(np.repeat(split_mean_img[o], n, 0), zp, zn)
              for o in SPLITS if o != s]
        arms["foreign_mean"] = float(np.mean(fm))
        arms["foreign_mean_sd"] = float(np.std(fm))

        rd = [acc(np.repeat(rng.standard_normal((1, zi.shape[1])).astype(np.float32), n, 0),
                  zp, zn) for _ in range(args.seeds)]
        arms["random_dir"] = float(np.mean(rd))
        arms["random_dir_sd"] = float(np.std(rd))

        arms["blind_bigram"] = prior[s]

        if args.exchange:
            arms["exchange_tautological"] = bool(
                exchange_is_tautological(zi, zp, zn)
                and exchange_is_tautological(np.repeat(mean_img, n, 0), zp, zn))

        # Typicality regression: does the fixed-direction score reduce to
        # caption length + unigram frequency?
        g_dir = unit(mean_img)[0]
        s_pos = unit(zp) @ g_dir
        s_neg = unit(zn) @ g_dir
        X = np.concatenate([caption_stats([it["caption"] for it in items[s]][:n], corpus, args.normalise),
                            caption_stats([it["negative_caption"] for it in items[s]][:n], corpus, args.normalise)])
        y = np.concatenate([s_pos, s_neg])
        Xd = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ beta
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        arms["typicality_r2"] = 1.0 - ss_res / max(ss_tot, 1e-12)
        pp, pn = pred[:n], pred[n:]
        arms["typicality_acc"] = float((pp > pn).mean() + 0.5 * (pp == pn).mean())

        arms["n"] = n
        arms["ci95_real"] = ci95(arms["real"], n)
        res[s] = arms
        print(f"{s:13s} n={n:5d} real {arms['real']:.4f} "
              f"shuf {arms['shuffle_pair']:.4f} mean_img {arms['mean_image']:.4f} "
              f"foreign {arms['foreign_mean']:.4f} rand {arms['random_dir']:.4f} "
              f"blind {arms['blind_bigram']:.4f} typ {arms['typicality_acc']:.4f}",
              flush=True)

    macro = {k: round(float(np.mean([res[s][k] for s in CLEAN5])), 4)
             for k in ("real", "shuffle_pair", "mean_image", "foreign_mean",
                       "random_dir", "blind_bigram", "typicality_acc")}
    print("\nclean-5 macro:", json.dumps(macro))

    out = Path(args.out or ROOT / "artifacts" / "nla" / "q4"
               / f"controls_{'norm_' if args.normalise else ''}{args.cell}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"cell": args.cell, "normalised": args.normalise,
         "txt_cache": args.txt or CELLS[args.cell]["txt"],
         "clean5_macro": macro, "per_split": res}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
