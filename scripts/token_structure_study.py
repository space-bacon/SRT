"""Token-structural-type signal analysis for the SRT adapter on gpt-oss-20b.

Companion / control to the word-category study. Instead of binning by the
*word* a one-word prompt contains, this reads the seven SRT signals at EVERY
token position of a real-text corpus (one forward pass, no generation) and bins
each position by the STRUCTURAL type of its token:

    word_start   leading-space alphabetic token (" running", " the")
    word_cont    subword continuation / morpheme (no leading space, alphabetic:
                 "ning", "ized", "tion")
    punct        punctuation only (".", ",", "!", "'s")
    number       digits
    space_nl     whitespace / newline only
    other        everything else (mixed, symbols, special)

This tests whether the SRT signals track token STRUCTURE (morphology /
punctuation) rather than word SEMANTICS — the tokenization confound behind the
word-category result. If token type explains large signal variance, part of the
word-category effect is tokenization-driven; if not, the word effect is genuinely
semantic.

Signal alignment (all indexed to "arrival at token t"):
    r_hat[t], regime_super[t], div_norm[t], chain[t]  — internal SRT state at t
    entropy[t], margin[t], nll[t]                     — from logits[t-1], the
        predictive distribution that produced token t
Position 0 (BOS) and padding are skipped.

Output: artifacts/nla/gptoss20b/token_structure_study.json

Usage:
    python scripts/token_structure_study.py \
        --adapter checkpoints/gptoss20b_phaseAB_v2/best_adapter.pt \
        --corpus data/val_200.jsonl --max-passages 200 --max-seq-len 128
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

SIGNALS = ["r_hat", "regime_super", "div_norm", "chain", "ent", "margin", "nll"]
TYPES = ["word_start", "word_cont", "punct", "number", "space_nl", "other"]
LAYERS = (6, 12, 18, 24)
_PUNCT = set(".,;:!?\"'`()[]{}<>-–—/\\|@#$%^&*_=+~")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--corpus", default="data/val_200.jsonl")
    p.add_argument("--backbone", default="openai/gpt-oss-20b")
    p.add_argument("--out", default="artifacts/nla/gptoss20b/token_structure_study.json")
    p.add_argument("--codebook-repo", default="RiverRider/srt-nla-gptoss20b-artifacts")
    p.add_argument("--max-passages", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-basin", action="store_true")
    return p.parse_args()


def token_type(s: str) -> str:
    """Classify a decoded single-token string by structural type."""
    if s == "":
        return "other"
    if s.strip() == "":
        return "space_nl"
    core = s.strip()
    leading_space = s[0] == " " or s[0].isspace()
    if all(c in _PUNCT for c in core):
        return "punct"
    if any(c.isdigit() for c in core) and all(c.isdigit() or c in ".,-" for c in core):
        return "number"
    if any(c.isalpha() for c in core):
        return "word_start" if leading_space else "word_cont"
    return "other"


def one_way_anova(groups: list[list[float]]) -> tuple[float, float, int]:
    groups = [g for g in groups if len(g) > 1]
    k = len(groups)
    if k < 2:
        return float("nan"), float("nan"), 0
    n = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / n
    ss_b = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    ss_w = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    df_b, df_w = k - 1, n - k
    if df_w <= 0 or ss_w <= 0:
        return float("inf"), 0.0, n
    F_ = (ss_b / df_b) / (ss_w / df_w)
    return F_, _f_sf(F_, df_b, df_w), n


def _f_sf(F_: float, d1: int, d2: int) -> float:
    if F_ <= 0:
        return 1.0
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F_))


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + num / (c if abs(c) > 1e-30 else 1e-30)
        f *= d * c
        if abs(1.0 - d * c) < 1e-8:
            break
    return front * (f - 1.0)


def mutual_information(cats: list[str], codes: list[int]) -> float:
    n = len(cats)
    if n == 0:
        return 0.0
    pc, pk, pjk = Counter(cats), Counter(codes), Counter(zip(cats, codes))
    mi = 0.0
    for (c, k), njk in pjk.items():
        pj = njk / n
        mi += pj * math.log(pj / ((pc[c] / n) * (pk[k] / n)) + 1e-30)
    return mi / math.log(2)


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    passages = []
    for line in open(args.corpus):
        line = line.strip()
        if not line:
            continue
        try:
            passages.append(json.loads(line)["text"])
        except Exception:
            passages.append(line)
        if len(passages) >= args.max_passages:
            break
    print(f"{len(passages)} passages", flush=True)

    cfg = SRTConfig(backbone_id=args.backbone, backbone_dtype="bfloat16")
    model = SRTAdapter(cfg).to("cuda", dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    model.load_adapter(args.adapter)
    model.eval()

    # decode-string cache for token types
    type_cache: dict[int, str] = {}

    def tid_type(tid: int) -> str:
        if tid not in type_cache:
            type_cache[tid] = token_type(tok.decode([tid]))
        return type_cache[tid]

    cb = mu_by = None
    if not args.no_basin:
        from huggingface_hub import hf_hub_download
        cbo = torch.load(hf_hub_download(args.codebook_repo, "state_codebook_vq.pt",
                                         repo_type="dataset"),
                         map_location="cuda", weights_only=False)
        cb_mu, cb_cent = cbo["mu"].float().cuda(), cbo["centroids"].float().cuda()

        def encode_batch(vs: torch.Tensor) -> list[int]:
            return torch.cdist(vs.float() - cb_mu, cb_cent).argmin(dim=-1).cpu().tolist()
        cb = encode_batch

    per_signal: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in SIGNALS}
    basin_types: list[str] = []
    basin_codes: dict[int, list[int]] = {L: [] for L in LAYERS}
    bb = None
    for attr in ("backbone", "model", "base_model", "lm", "frozen_backbone"):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "config") and hasattr(obj, "forward"):
            bb = obj
            break

    for start in range(0, len(passages), args.batch_size):
        chunk = passages[start:start + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_seq_len).to("cuda")
        ids, attn = enc.input_ids, enc.attention_mask
        with torch.no_grad():
            out = model(ids, attention_mask=attn, read_only=True)
            logits = out.logits.float()                         # (B,T,V)
            r_hat = out.ben_output.r_hat.float()                # (B,T)
            regime = F.softmax(out.ben_output.regime_logits.float(), dim=-1)[..., 1]
            div = out.divergences[-1].float().norm(dim=-1)      # (B,T)
            chain = (out.chain_residual_per_token.float()
                     if out.chain_residual_per_token is not None
                     else torch.zeros_like(r_hat))
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            ent_all = -(probs * logp).sum(-1)                   # (B,T) entropy at t (predict t+1)
            top2 = probs.topk(2, dim=-1).values
            margin_all = (top2[..., 0] - top2[..., 1])          # (B,T)
            if cb is not None and bb is not None:
                hs = bb(ids, attention_mask=attn, output_hidden_states=True,
                        use_cache=False).hidden_states

        B, T = ids.shape
        for b in range(B):
            for t in range(1, T):
                if attn[b, t] == 0:
                    break
                tid = int(ids[b, t])
                tt = tid_type(tid)
                # arrival-at-t alignment: uncertainty from position t-1
                per_signal["ent"][tt].append(float(ent_all[b, t - 1]))
                per_signal["margin"][tt].append(float(margin_all[b, t - 1]))
                per_signal["nll"][tt].append(float(-logp[b, t - 1, tid]))
                per_signal["r_hat"][tt].append(float(r_hat[b, t]))
                per_signal["regime_super"][tt].append(float(regime[b, t]))
                per_signal["div_norm"][tt].append(float(div[b, t]))
                per_signal["chain"][tt].append(float(chain[b, t]))
                if cb is not None and bb is not None:
                    basin_types.append(tt)
                    for L in LAYERS:
                        basin_codes[L].append(int(
                            torch.cdist((hs[L][b, t].float() - cb_mu).unsqueeze(0),
                                        cb_cent).argmin()))
        if (start // args.batch_size) % 4 == 0:
            print(f"  {start + len(chunk)}/{len(passages)} passages", flush=True)

    counts = {tt: len(per_signal["ent"][tt]) for tt in TYPES}
    present = [tt for tt in TYPES if counts[tt] > 1]

    def anova_table():
        out = {}
        for s in SIGNALS:
            groups = [per_signal[s][tt] for tt in present if per_signal[s][tt]]
            F_, p, n = one_way_anova(groups)
            out[s] = {"F": F_, "p": p, "n": n}
        return out

    means = {s: {tt: (sum(per_signal[s][tt]) / len(per_signal[s][tt])
                      if per_signal[s][tt] else None) for tt in TYPES}
             for s in SIGNALS}

    result = {
        "backbone": args.backbone, "adapter": args.adapter,
        "n_passages": len(passages), "token_types": present,
        "counts": counts, "anova": anova_table(), "means": means,
    }

    if cb is not None:
        basin = {}
        for L in LAYERS:
            mi = mutual_information(basin_types, basin_codes[L])
            per_type_top = {}
            for tt in present:
                cc = Counter(k for t_, k in zip(basin_types, basin_codes[L]) if t_ == tt)
                per_type_top[tt] = cc.most_common(3)
            basin[str(L)] = {"mi_bits": mi, "n_codes": len(set(basin_codes[L])),
                             "top_codes_per_type": per_type_top}
        result["basin_analysis"] = basin

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\ncounts:", {tt: counts[tt] for tt in present})
    print("\n=== ANOVA across token types ===")
    for s in SIGNALS:
        a = result["anova"][s]
        star = "***" if a["p"] < 1e-3 else ("**" if a["p"] < 1e-2 else ("*" if a["p"] < 5e-2 else "ns"))
        print(f"  {s:13s} F={a['F']:9.2f}  p={a['p']:.1e}  {star}  n={a['n']}")
    print("\n=== means ===")
    print("  " + "signal".ljust(13) + "".join(tt[:10].rjust(11) for tt in present))
    for s in SIGNALS:
        row = "  " + s.ljust(13)
        for tt in present:
            v = means[s][tt]
            row += (f"{v:11.3f}" if v is not None else " " * 11)
        print(row)
    if cb is not None:
        print("\n=== basin MI(token_type; code) per layer ===")
        for L in LAYERS:
            print(f"  L{L}: {result['basin_analysis'][str(L)]['mi_bits']:.3f} bits")


if __name__ == "__main__":
    main()
