#!/usr/bin/env python3
"""De-risking probe for the "contested-word decompressor" demo idea.

Question: if you replace a word with progressively longer *uncontested* wording,
can you keep the sentence's semantic meaning (srt-adapter STS embedding) while
shedding the metapragmatic load (MAH divergence)? And is there a floor on how
much load a contested word can shed before the meaning breaks?

Two axes, measured with the released product:
  * STS retention = cosine of `community_output.encoded` (the adapter's sentence
    embedding) between the original and the rewrite. This is meaning-preservation.
  * load          = MAH divergence norm (`out.divergences[-1]`), max over the
    sentence's content tokens. This is the metapragmatic weight.

Prediction to test: contested words (freedom, justice, gender) start heavy; plain
expansions should drop the load, but the question is whether the load floor you
can reach while keeping STS high is still above the controls (bicycle, teapot).

    /venv/main/bin/python scripts/decompress_probe.py --repo RiverRider/srt-adapter-v1.0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import (
    BENConfig,
    CommunityConfig,
    LossConfig,
    MAHConfig,
    RRMConfig,
    SRTConfig,
)

# Each item: the carrier sentence, a paraphrase that KEEPS the word (the ceiling),
# uncontested expansions from short to long, and an unrelated sentence (the floor).
PROBES = [
    # ---- contested ----
    dict(word="freedom", kind="contested",
         carrier="They fought a long war for freedom.",
         keep="They fought a long war to be free.",
         expand=[
             "They fought a long war for independence.",
             "They fought a long war for the right to govern themselves.",
             "They fought a long war for the right of the people to govern themselves without control by a distant ruler.",
         ],
         unrelated="She poured the coffee and opened the newspaper."),
    dict(word="justice", kind="contested",
         carrier="The crowd demanded justice.",
         keep="The crowd demanded that justice be done.",
         expand=[
             "The crowd demanded fairness.",
             "The crowd demanded that the guilty be punished and the wronged repaid.",
             "The crowd demanded that whoever was responsible be punished in proportion to the harm, and that those harmed be repaid.",
         ],
         unrelated="The train arrived at the station at noon."),
    dict(word="gender", kind="contested",
         carrier="The form asked about gender.",
         keep="The form asked about the person's gender.",
         expand=[
             "The form asked about sex.",
             "The form asked whether the person was a man or a woman.",
             "The form asked how the person understood themselves as a man, a woman, or something else.",
         ],
         unrelated="The bakery sells fresh bread each morning."),
    # ---- concrete controls ----
    dict(word="bicycle", kind="control",
         carrier="He rode his bicycle to work.",
         keep="He rode his bike to work.",
         expand=[
             "He rode his two-wheeled cycle to work.",
             "He rode his pedal-powered two-wheeled vehicle to work.",
             "He rode his human-powered vehicle with two wheels and pedals to work.",
         ],
         unrelated="The orchestra slowly tuned their instruments."),
    dict(word="teapot", kind="control",
         carrier="She placed the teapot on the table.",
         keep="She placed the pot of tea on the table.",
         expand=[
             "She placed the vessel for tea on the table.",
             "She placed the small vessel used for brewing and pouring tea on the table.",
             "She placed the small covered container with a spout, used to brew and pour tea, on the table.",
         ],
         unrelated="The committee postponed the meeting until spring."),
]

# Enrichment ladders: the head noun is held constant while modifiers pin the
# referent. Reading the head-noun's own load across the ladder tells us whether
# specifying the referent drops the load (resolvable / referential) or not
# (irreducible / contested).
CARRIER = "They were discussing {phrase} at the long meeting yesterday."
ENRICH = {
    "bike": dict(kind="control", head="bike", ladder=[
        "the bike", "the road bike", "the red carbon road bike",
        "the red carbon road bike with drop handlebars and gears"]),
    "teapot": dict(kind="control", head="teapot", ladder=[
        "the teapot", "the porcelain teapot", "the small white porcelain teapot",
        "the small white porcelain teapot with a chipped spout"]),
    "triangle": dict(kind="abstract", head="triangle", ladder=[
        "the triangle", "the right triangle", "the small right triangle",
        "the small right triangle with legs of three and four"]),
    "freedom": dict(kind="contested", head="freedom", ladder=[
        "freedom", "press freedom", "constitutional press freedom",
        "the constitutional press freedom protected by the first amendment"]),
    "justice": dict(kind="contested", head="justice", ladder=[
        "justice", "criminal justice", "the criminal justice system",
        "the criminal justice system that sentences convicted offenders"]),
    "gender": dict(kind="contested", head="gender", ladder=[
        "gender", "the gender box", "the gender box on the form",
        "the gender box on the government census form"]),
}

# Fact/value test, properly powered and describability-controlled: THICK (moral-evaluative,
# verdict fused) vs ABSTRACT-thin (abstract, definitional, NO verdict, no rich referent).
# Concrete is dropped: its keep/strip gap was a describability artifact, not a verdict. keep =
# synonym KEEPING the verdict; strip = a description with the verdict REMOVED. Scored CENTERED.
FACTVALUE = [
    dict(word="justice", kind="thick", carrier="The court delivered justice.",
         keep="The court delivered fairness.", strip="The court handed down the assigned penalties."),
    dict(word="cruelty", kind="thick", carrier="They condemned the cruelty of it.",
         keep="They condemned the viciousness of it.", strip="They noted the pain it caused."),
    dict(word="courage", kind="thick", carrier="Everyone admired her courage.",
         keep="Everyone admired her bravery.", strip="Everyone saw that she acted despite the danger."),
    dict(word="betrayal", kind="thick", carrier="It was a plain betrayal.",
         keep="It was a plain act of treachery.", strip="It was a plain breaking of the agreement."),
    dict(word="generosity", kind="thick", carrier="The gift showed real generosity.",
         keep="The gift showed real kindness.", strip="The gift was large for the giver's means."),
    dict(word="honesty", kind="thick", carrier="They valued his honesty.",
         keep="They valued his truthfulness.", strip="They valued that he reported what was so."),
    dict(word="loyalty", kind="thick", carrier="She was known for her loyalty.",
         keep="She was known for her faithfulness.", strip="She was known for staying with the group."),
    dict(word="greed", kind="thick", carrier="The plan was driven by greed.",
         keep="The plan was driven by avarice.", strip="The plan was driven by a wish for more money."),
    dict(word="arrogance", kind="thick", carrier="People resented his arrogance.",
         keep="People resented his conceit.", strip="People resented that he placed himself above them."),
    dict(word="mercy", kind="thick", carrier="The judge showed mercy.",
         keep="The judge showed leniency.", strip="The judge gave a lighter sentence than allowed."),
    dict(word="tyranny", kind="thick", carrier="They rose against tyranny.",
         keep="They rose against oppression.", strip="They rose against rule with no limit on the ruler."),
    dict(word="hypocrisy", kind="thick", carrier="The article exposed his hypocrisy.",
         keep="The article exposed his duplicity.", strip="The article showed his acts differed from his words."),
    dict(word="kindness", kind="thick", carrier="They were moved by her kindness.",
         keep="They were moved by her warmth.", strip="They were moved that she helped without being asked."),
    dict(word="brutality", kind="thick", carrier="Witnesses described the brutality.",
         keep="Witnesses described the savagery.", strip="Witnesses described the force used and the injuries."),
    dict(word="number", kind="abstract", carrier="She wrote down the number.",
         keep="She wrote down the figure.", strip="She wrote down the count of items."),
    dict(word="triangle", kind="abstract", carrier="He drew a triangle.",
         keep="He drew a three-sided figure.", strip="He drew a polygon with three edges."),
    dict(word="distance", kind="abstract", carrier="They measured the distance.",
         keep="They measured the length.", strip="They measured how far apart the points were."),
    dict(word="average", kind="abstract", carrier="He computed the average.",
         keep="He computed the mean.", strip="He computed the central value of the numbers."),
    dict(word="angle", kind="abstract", carrier="She measured the angle.",
         keep="She measured the inclination.", strip="She measured the turn between the two lines."),
    dict(word="ratio", kind="abstract", carrier="They found the ratio.",
         keep="They found the proportion.", strip="They found one quantity divided by the other."),
    dict(word="volume", kind="abstract", carrier="He measured the volume.",
         keep="He measured the capacity.", strip="He measured the space it takes up."),
    dict(word="frequency", kind="abstract", carrier="They recorded the frequency.",
         keep="They recorded the rate.", strip="They recorded how often it happened."),
    dict(word="radius", kind="abstract", carrier="She measured the radius.",
         keep="She measured the half-diameter.", strip="She measured from the center to the edge."),
    dict(word="sum", kind="abstract", carrier="He wrote the sum.",
         keep="He wrote the total.", strip="He wrote the result of adding them."),
    dict(word="width", kind="abstract", carrier="They measured the width.",
         keep="They measured the breadth.", strip="They measured how wide it was."),
    dict(word="altitude", kind="abstract", carrier="They noted the altitude.",
         keep="They noted the elevation.", strip="They noted the height above sea level."),
    dict(word="speed", kind="abstract", carrier="He clocked the speed.",
         keep="He clocked the pace.", strip="He clocked how fast it moved."),
    dict(word="quantity", kind="abstract", carrier="She recorded the quantity.",
         keep="She recorded the amount.", strip="She recorded how many there were."),
]

# Generic pool is no longer needed: the adapter STS cosine is calibrated and the
# MAH load is an absolute scalar, so neither axis needs centering.


def build_config(config_path) -> SRTConfig:
    raw = json.loads(Path(config_path).read_text())
    return SRTConfig(
        backbone_id=raw["backbone_id"],
        backbone_dtype=raw["backbone_dtype"],
        mah_layer_indices=list(raw["mah_layer_indices"]),
        rrm_inject_indices=list(raw["rrm_inject_indices"]),
        community_layer_idx=raw["community_layer_idx"],
        num_mah_layers=raw["num_mah_layers"],
        mah=MAHConfig(**raw["mah"]),
        rrm=RRMConfig(**raw["rrm"]),
        ben=BENConfig(**raw["ben"]),
        community=CommunityConfig(**raw["community"]),
        loss=LossConfig(**{k: v for k, v in raw["loss"].items() if k in LossConfig.__dataclass_fields__}),
    )


@torch.no_grad()
def load_model(repo: str, device: str):
    cfg = build_config(hf_hub_download(repo, "config.json"))
    model = SRTAdapter(cfg).to(device)
    state = load_file(hf_hub_download(repo, "adapter.safetensors"), device=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.backbone_id)
    return model, tok


@torch.no_grad()
def readouts(model, tok, text: str, device: str, max_seq_len: int = 64):
    """Return (sts_embedding, load_max, load_mean) for one sentence."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    encoded = out.community_output.encoded
    if encoded.dim() == 3:
        m = enc.attention_mask.unsqueeze(-1).to(encoded.dtype)
        encoded = (encoded * m).sum(1) / m.sum(1).clamp_min(1.0)
    emb = F.normalize(encoded, p=2, dim=-1)[0].float().cpu()
    load = out.divergences[-1][0].norm(dim=-1).float().cpu()   # (T,) per-token load, last MAH layer
    load = load[1:] if load.numel() > 1 else load             # drop BOS
    return emb, float(load.max()), float(load.mean())


def cos(a, b) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def single_ids(tok, words):
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.append(t[0])
    return ids


@torch.no_grad()
def valence(model, tok, text, pos_ids, neg_ids, device, frame=" Overall, it was"):
    s = text.rstrip(".") + "." + frame
    enc = tok(s, return_tensors="pt").to(device)
    logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[0, -1].float().cpu()
    lp = torch.log_softmax(logits, dim=-1)
    pos = torch.logsumexp(lp[pos_ids], dim=0)
    neg = torch.logsumexp(lp[neg_ids], dim=0)
    return float(pos - neg)   # signed evaluative charge


@torch.no_grad()
def token_loads(model, tok, text: str, device: str, max_seq_len: int = 64):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    load = out.divergences[-1][0].norm(dim=-1).float().cpu().tolist()
    toks = tok.convert_ids_to_tokens(enc.input_ids[0].tolist())
    return list(zip(toks, load))


@torch.no_grad()
def head_span_load(model, tok, text: str, head: str, device: str, max_seq_len: int = 64) -> float:
    """Mean MAH load over the tokens of `head` (its last occurrence) inside `text`."""
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
              truncation=True, max_length=max_seq_len)
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    load = out.divergences[-1][0].norm(dim=-1).float().cpu()
    lo = text.lower().rfind(head.lower())
    hi = lo + len(head)
    idxs = [i for i, (a, b) in enumerate(offs) if b > a and a < hi and b > lo]
    if not idxs:
        return float("nan")
    return float(load[idxs].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RiverRider/srt-adapter-v1.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sts-thresh", type=float, default=0.75,
                    help="keep-meaning bar for the residual-load metric")
    ap.add_argument("--dump", action="store_true", help="print per-token load for each carrier and exit")
    ap.add_argument("--enrich", action="store_true", help="enrichment-ladder test: does specifying the referent drop the head-noun load?")
    ap.add_argument("--battery", action="store_true", help="detector bake-off vs curated contestedness on the battery")
    ap.add_argument("--battery-path", default="data/probes/contestedness_battery_v1.jsonl")
    ap.add_argument("--factvalue", action="store_true", help="fact/value test: does stripping the verdict cost meaning for thick words only?")
    ap.add_argument("--valence", action="store_true", help="valence residual: does stripping the verdict remove the sentence's evaluative charge?")
    ap.add_argument("--out", default="artifacts/nla/decompress_probe_stage1.json")
    args = ap.parse_args()

    model, tok = load_model(args.repo, args.device)
    print(f"repo={args.repo}\n")

    if args.dump:
        for p in PROBES:
            print(f"== {p['word']} ({p['kind']}): {p['carrier']}")
            for t, l in token_loads(model, tok, p["carrier"], args.device):
                print(f"    {l:7.2f}  {t!r}")
            print()
        return

    if args.enrich:
        print("head-noun load as the referent is specified. carrier: " + CARRIER.format(phrase="<phrase>") + "\n")
        rows = []
        for word, spec in ENRICH.items():
            loads = [head_span_load(model, tok, CARRIER.format(phrase=ph), spec["head"], args.device)
                     for ph in spec["ladder"]]
            drop = loads[0] - loads[-1]
            residual = loads[-1]
            rows.append(dict(word=word, kind=spec["kind"], loads=loads, drop=drop, residual=residual))
            ladder = "  ".join(f"{l:5.2f}" for l in loads)
            print(f"{word:9s} ({spec['kind']:9s})  L0..Ln: {ladder}   drop={drop:+.2f}  residual={residual:.2f}")
        con = [r for r in rows if r["kind"] == "contested"]
        res = [r for r in rows if r["kind"] != "contested"]

        def m(rs, k):
            return sum(r[k] for r in rs) / len(rs)

        print("\nmean drop      contested=%+.2f   resolvable=%+.2f" % (m(con, "drop"), m(res, "drop")))
        print("mean residual  contested=%.2f    resolvable=%.2f" % (m(con, "residual"), m(res, "residual")))
        print("=> if contested keeps a high residual while resolvable words fall, residual-after-enrichment is the per-word detector")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out.replace(".json", "_enrich.json"), "w") as f:
            json.dump(dict(repo=args.repo, carrier=CARRIER, rows=rows), f, indent=2)
        return

    if args.battery:
        import math

        def _rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[order[k]] = (i + j) / 2.0 + 1.0
                i = j + 1
            return r

        def spearman(x, y):
            rx, ry = _rank(x), _rank(y)
            n = len(x)
            mx, my = sum(rx) / n, sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
            return num / den if den else float("nan")

        items = [json.loads(l) for l in Path(args.battery_path).read_text().splitlines() if l.strip()]
        cols = {"div_mean": [], "div_last": [], "comm_entropy": [], "comm_top1": [], "r_hat_mean": []}
        scores = []
        for it in items:
            enc = tok(it["prompt"], return_tensors="pt", truncation=True, max_length=64).to(args.device)
            with torch.no_grad():
                out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            if out.divergences:
                div = out.divergences[-1][0].norm(dim=-1).float().cpu()
                cols["div_mean"].append(float(div.mean()))
                cols["div_last"].append(float(div[-1]))
            else:
                cols["div_mean"].append(float("nan"))
                cols["div_last"].append(float("nan"))
            w = getattr(out.community_output, "weights", None)
            if w is not None:
                p = w[0].float().cpu()
                p = p.mean(0) if p.dim() == 2 else p
                if float(p.min()) < 0 or abs(float(p.sum()) - 1.0) > 0.5:
                    p = torch.softmax(p, dim=-1)          # weights were logits
                else:
                    p = p / p.sum().clamp_min(1e-9)
                cols["comm_entropy"].append(float(-(p * p.clamp_min(1e-12).log()).sum() / math.log(p.numel())))
                cols["comm_top1"].append(float(p.max()))
            else:
                cols["comm_entropy"].append(float("nan"))
                cols["comm_top1"].append(float("nan"))
            rh = out.ben_output.r_hat[0].float().cpu() if out.ben_output is not None else None
            cols["r_hat_mean"].append(float(rh.mean()) if rh is not None else float("nan"))
            scores.append(float(it["contest_score"]))
        print("detector bake-off on %d battery concepts (Spearman vs curated contestedness):\n" % len(items))
        for k, vals in cols.items():
            if all(math.isnan(x) for x in vals):
                print(f"  {k:14s}  (unavailable)")
                continue
            print(f"  {k:14s}  rho={spearman(vals, scores):+.3f}")
        return

    if args.factvalue:
        import random
        import itertools
        from collections import defaultdict
        # Embed once, then CENTER by the pool mean (anisotropy adjustment). The adapter STS
        # space is anisotropic (unrelated sentences sit well above 0), which compresses raw
        # cosines and can wash out the small keep-minus-strip gaps.
        embs = {}
        for it in FACTVALUE:
            for key in ("carrier", "keep", "strip"):
                t = it[key]
                if t not in embs:
                    embs[t] = readouts(model, tok, t, args.device)[0]
        mu = torch.stack(list(embs.values())).mean(0)
        allv = list(embs.values())
        pairs = list(itertools.combinations(range(len(allv)), 2))
        aniso = sum(cos(allv[i], allv[j]) for i, j in pairs) / len(pairs)

        def ccos(a, b):
            return cos(a - mu, b - mu)

        print("keep = synonym keeping the verdict; strip = description with the verdict removed")
        print(f"raw anisotropy (mean pairwise cos, {len(embs)} sentences) = {aniso:.3f}  -> centering by pool mean")
        print("centered retention vs the original; rawgap = uncentered keep-minus-strip\n")
        print(f"{'word':11s} {'class':9s} {'keepC':>6s} {'stripC':>7s} {'gapC':>7s} {'rawgap':>7s}")
        print("-" * 56)
        rows = []
        for it in FACTVALUE:
            e0, ek, es = embs[it["carrier"]], embs[it["keep"]], embs[it["strip"]]
            rk, rs = ccos(e0, ek), ccos(e0, es)
            rawgap = cos(e0, ek) - cos(e0, es)
            rows.append((it["word"], it["kind"], rk, rs, rk - rs))
            print(f"{it['word']:11s} {it['kind']:9s} {rk:6.3f} {rs:7.3f} {rk - rs:+7.3f} {rawgap:+7.3f}")

        by = defaultdict(list)
        for w, k, rk, rs, g in rows:
            by[k].append(g)

        def mean(x):
            return sum(x) / len(x) if x else float("nan")

        print("\nmean CENTERED keep-minus-strip gap (larger => removing the verdict costs more meaning):")
        for k in ["thick", "concrete", "abstract"]:
            if by[k]:
                print(f"  {k:9s} {mean(by[k]):+.3f}  (n={len(by[k])})")

        labels = [1 if k == "thick" else 0 for _, k, _, _, _ in rows]
        gaps = [g for *_, g in rows]

        def did(lab):
            a = [g for g, l in zip(gaps, lab) if l == 1]
            b = [g for g, l in zip(gaps, lab) if l == 0]
            return mean(a) - mean(b)

        obs = did(labels)
        random.seed(0)
        P, c = 20000, 0
        for _ in range(P):
            perm = labels[:]
            random.shuffle(perm)
            if abs(did(perm)) >= abs(obs):
                c += 1
        print(f"\nthick vs rest CENTERED gap = {obs:+.3f}   permutation p = {(c + 1) / (P + 1):.4f}")

        # --- Erik Otarola-Castillo swamping correction (geometric morphometrics) ---
        # The describability nuisance is a dominant "size" axis swamping the verdict "shape".
        # Centering removed only translation. Now remove the COMMON ALLOMETRIC component:
        # regress gap on paraphrase-size, then permutation-test groups on the residual shape.
        def ntok(t):
            return len(tok(t, add_special_tokens=False).input_ids)

        proxy = [ntok(it["strip"]) - ntok(it["keep"]) for it in FACTVALUE]
        gall = [g for *_, g in rows]
        n = len(gall)
        mx, my = sum(proxy) / n, sum(gall) / n
        sp2 = sum((p - mx) ** 2 for p in proxy) or 1.0
        b = sum((p - mx) * (g - my) for p, g in zip(proxy, gall)) / sp2
        a = my - b * mx
        resid = [g - (a + b * p) for g, p in zip(gall, proxy)]
        sg = (sum((g - my) ** 2 for g in gall)) ** 0.5 or 1.0
        r = sum((p - mx) * (g - my) for p, g in zip(proxy, gall)) / ((sp2 ** 0.5) * sg)
        print(f"\nallometry: gap ~ paraphrase-size  r={r:+.3f}  slope={b:+.4f}  "
              f"(high |r| => describability/size is swamping the shape)")
        rt = [rr for rr, l in zip(resid, labels) if l]
        ra = [rr for rr, l in zip(resid, labels) if not l]
        obs2 = mean(rt) - mean(ra)
        random.seed(1)
        c2 = 0
        for _ in range(P):
            perm = labels[:]
            random.shuffle(perm)
            a2 = [rr for rr, l in zip(resid, perm) if l]
            b2 = [rr for rr, l in zip(resid, perm) if not l]
            if abs(mean(a2) - mean(b2)) >= abs(obs2):
                c2 += 1
        print(f"SIZE-CORRECTED thick vs abstract = {obs2:+.3f}   permutation p = {(c2 + 1) / (P + 1):.4f}")
        return

    if args.valence:
        import random
        from collections import defaultdict
        POS = [" good", " right", " just", " fair", " noble", " admirable", " honorable", " brave", " kind"]
        NEG = [" bad", " wrong", " unjust", " cruel", " shameful", " evil", " dishonorable", " cowardly", " selfish"]
        pos_ids, neg_ids = single_ids(tok, POS), single_ids(tok, NEG)
        print(f"evaluative charge = logP(pos) - logP(neg) at ' Overall, it was'  (pos_ids={len(pos_ids)} neg_ids={len(neg_ids)})\n")
        print(f"{'word':11s} {'class':9s} {'chargeK':>8s} {'chargeS':>8s} {'|K|-|S|':>8s}")
        print("-" * 48)
        rows = []
        for it in FACTVALUE:
            vk = valence(model, tok, it["keep"], pos_ids, neg_ids, args.device)
            vs = valence(model, tok, it["strip"], pos_ids, neg_ids, args.device)
            drop = abs(vk) - abs(vs)
            rows.append((it["word"], it["kind"], vk, vs, drop))
            print(f"{it['word']:11s} {it['kind']:9s} {vk:+8.2f} {vs:+8.2f} {drop:+8.2f}")

        by = defaultdict(list)
        for w, k, vk, vs, d in rows:
            by[k].append(d)

        def mean(x):
            return sum(x) / len(x) if x else float("nan")

        print("\nmean drop in |evaluative charge| when the verdict is stripped")
        print("(larger => the word was carrying a verdict that the description does not):")
        for k in ["thick", "concrete", "abstract"]:
            if by[k]:
                print(f"  {k:9s} {mean(by[k]):+.3f}  (n={len(by[k])})")

        labels = [1 if k == "thick" else 0 for _, k, _, _, _ in rows]
        drops = [d for *_, d in rows]

        def did(lab):
            a = [d for d, l in zip(drops, lab) if l]
            b = [d for d, l in zip(drops, lab) if not l]
            return mean(a) - mean(b)

        obs = did(labels)
        random.seed(0)
        P, c = 20000, 0
        for _ in range(P):
            perm = labels[:]
            random.shuffle(perm)
            if abs(did(perm)) >= abs(obs):
                c += 1
        print(f"\nthick vs rest |charge|-drop = {obs:+.3f}   permutation p = {(c + 1) / (P + 1):.4f}")
        return

    print("STS = semantic retention vs the original (adapter cosine); "
          "load = MAH load, max over content tokens\n")

    rows = []
    for p in PROBES:
        e0, mx0, _ = readouts(model, tok, p["carrier"], args.device)
        ek, mxk, _ = readouts(model, tok, p["keep"], args.device)
        exps = [readouts(model, tok, e, args.device) for e in p["expand"]]
        eu, _, _ = readouts(model, tok, p["unrelated"], args.device)

        print(f"{p['word']} ({p['kind']})   carrier load_max={mx0:.2f}")
        print(f'    keep   STS={cos(e0, ek):.3f}  load_max={mxk:.2f}   "{p["keep"]}"')
        for (ee, mxe, _), txt in zip(exps, p["expand"]):
            print(f'    exp    STS={cos(e0, ee):.3f}  load_max={mxe:.2f}   "{txt}"')
        print(f'    floor  STS={cos(e0, eu):.3f}                (unrelated)\n')

        keep_ok = [mxe for (ee, mxe, _) in exps if cos(e0, ee) >= args.sts_thresh]
        residual = min(keep_ok) if keep_ok else mx0
        rows.append(dict(
            word=p["word"], kind=p["kind"], carrier_load=mx0, keep_sts=cos(e0, ek),
            exp_sts=[cos(e0, ee) for (ee, _, _) in exps],
            exp_load=[mxe for (_, mxe, _) in exps],
            floor_sts=cos(e0, eu), residual_load=residual,
        ))

    con = [r for r in rows if r["kind"] == "contested"]
    ctl = [r for r in rows if r["kind"] == "control"]

    def avg(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    print("summary (residual_load = lowest load still reachable while STS >= %.2f):" % args.sts_thresh)
    print("  contested  carrier=%.2f  residual=%.2f" % (avg(con, "carrier_load"), avg(con, "residual_load")))
    print("  control    carrier=%.2f  residual=%.2f" % (avg(ctl, "carrier_load"), avg(ctl, "residual_load")))
    print("  => a high contested residual means plain words cannot carry the load without losing the meaning")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(repo=args.repo, sts_thresh=args.sts_thresh, rows=rows), f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
