"""SRT capability evaluation.

Measures four capabilities exposed by the SRT adapter that no other
embedding model provides, on a small fixed bench:

  Cap-A  Community purity (NMI vs Banking77 intent labels)
  Cap-B  MAH boundary detection F1 on synthetic concatenated sequences
  Cap-C  Paraphrase community-stability (mean cosine on STSB pairs >=4.5)
  Cap-D  BEN regime coverage (fraction subcritical/supercritical, diagnostic)

Usage:
    python scripts/capability_eval.py \
        --backbone Qwen/Qwen2.5-7B \
        --adapter checkpoints/.../best_adapter.pt \
        --output-dir artifacts/capability/<tag>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srt.adapter import SRTAdapter  # noqa: E402
from srt.config import SRTConfig  # noqa: E402

log = logging.getLogger("cap_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--n-cap-a", type=int, default=1000, help="Banking77 sentences")
    p.add_argument("--n-cap-b", type=int, default=200, help="concat sequences")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_device(req):
    if req:
        return torch.device(req)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args, device):
    cfg = SRTConfig()
    cfg.backbone_name = args.backbone
    cfg.dtype = args.dtype
    model = SRTAdapter(cfg).to(device)
    model.load_adapter(args.adapter)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def forward_batch(model, tok, texts, device, max_len):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len)
    return model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
    ), enc["attention_mask"].to(device)


# ─────────────────────────── Cap-A ────────────────────────────
def cap_a_community_purity(model, tok, device, args, rng):
    """NMI of K-means(community_encoded) vs Banking77 intent labels."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    n = min(args.n_cap_a, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    texts = [ds[int(i)]["text"] for i in idx]
    labels = np.array([ds[int(i)]["label"] for i in idx])
    K = int(labels.max()) + 1

    embs = []
    for i in range(0, n, args.batch_size):
        chunk = texts[i:i + args.batch_size]
        out, _ = forward_batch(model, tok, chunk, device, args.max_seq_len)
        v = F.normalize(out.community_output.encoded.float(), dim=-1)
        embs.append(v.cpu().numpy())
    X = np.concatenate(embs, axis=0)

    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(X)
    nmi = float(normalized_mutual_info_score(labels, km.labels_))
    log.info("Cap-A community NMI = %.4f (n=%d, K=%d)", nmi, n, K)
    return {"nmi": nmi, "n": n, "K": K}


# ─────────────────────────── Cap-B ────────────────────────────
def cap_b_boundary_f1(model, tok, device, args, rng):
    """Concatenate 5 random Banking77 sentences from different intents.
    Check whether peaks of last-layer divergence norm align with the 4
    boundary positions (token tolerance ±2)."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    by_intent: dict[int, list[str]] = {}
    for row in ds:
        by_intent.setdefault(row["label"], []).append(row["text"])
    intents = list(by_intent)

    n = args.n_cap_b
    tp = fp = fn = 0
    tol = 2
    SEP = " "

    for _ in range(n):
        chosen = rng.choice(intents, size=5, replace=False)
        sents = [rng.choice(by_intent[int(c)]) for c in chosen]

        # Tokenize each separately to find boundary token offsets
        per_sent_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in sents]
        # Cumulative end positions of each sentence in the joined sequence
        boundaries = []
        cum = 0
        for ids in per_sent_ids[:-1]:
            cum += len(ids)
            # account for separator token(s)
            sep_ids = tok(SEP, add_special_tokens=False)["input_ids"]
            cum += len(sep_ids)
            boundaries.append(cum)  # token index where boundary occurs

        joined = SEP.join(sents)
        out, mask = forward_batch(model, tok, [joined], device, args.max_seq_len)
        if not out.divergences:
            continue
        div = out.divergences[-1][0].float()  # (T, d_div)
        valid = mask[0].bool()
        norms = div.norm(dim=-1).cpu().numpy()
        norms = norms[: int(valid.sum())]
        if norms.size < 6:
            continue

        # Top-4 peak positions (excluding first 2 and last 2 to avoid edge)
        scoreable = norms.copy()
        scoreable[:2] = -np.inf
        scoreable[-2:] = -np.inf
        peaks = set(np.argsort(scoreable)[-4:].tolist())

        # Filter boundaries to those within sequence
        gold = [b for b in boundaries if b < len(norms)]
        gold_set = set(gold)

        matched = set()
        for p in peaks:
            for g in gold_set:
                if abs(p - g) <= tol and g not in matched:
                    matched.add(g)
                    break
        tp += len(matched)
        fp += len(peaks) - len(matched)
        fn += len(gold_set) - len(matched)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    log.info("Cap-B boundary F1 = %.4f (P=%.3f R=%.3f, n=%d)", f1, prec, rec, n)
    return {"f1": f1, "precision": prec, "recall": rec, "n": n, "tol": tol}


# ─────────────────────────── Cap-C ────────────────────────────
def cap_c_paraphrase_stability(model, tok, device, args):
    """Mean cosine of community_encoded between paraphrase pairs (gold>=4.5)."""
    from datasets import load_dataset
    ds = load_dataset("mteb/stsbenchmark-sts", split="test")
    pairs = [(r["sentence1"], r["sentence2"]) for r in ds if r["score"] >= 4.5]
    if not pairs:
        return {"mean_cos": None, "n": 0}

    cos_vals = []
    for i in range(0, len(pairs), args.batch_size):
        chunk = pairs[i:i + args.batch_size]
        flat = [s for p in chunk for s in p]
        out, _ = forward_batch(model, tok, flat, device, args.max_seq_len)
        v = F.normalize(out.community_output.encoded.float(), dim=-1)
        a, b = v[0::2], v[1::2]
        cos_vals.extend((a * b).sum(dim=-1).cpu().numpy().tolist())

    mean_cos = float(np.mean(cos_vals))
    std_cos = float(np.std(cos_vals))
    log.info("Cap-C paraphrase mean-cos = %.4f ± %.4f (n=%d)",
             mean_cos, std_cos, len(cos_vals))
    return {"mean_cos": mean_cos, "std_cos": std_cos, "n": len(cos_vals)}


# ─────────────────────────── Cap-D ────────────────────────────
def cap_d_ben_regime(model, tok, device, args, rng):
    """Fraction of positions classified subcritical vs supercritical."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    n = min(500, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    texts = [ds[int(i)]["text"] for i in idx]

    sub = sup = total = 0
    rhats = []
    for i in range(0, n, args.batch_size):
        chunk = texts[i:i + args.batch_size]
        out, mask = forward_batch(model, tok, chunk, device, args.max_seq_len)
        if out.ben_output is None:
            continue
        regime = out.ben_output.regime_logits.argmax(-1)  # (B, T) 0=sub, 1=sup
        r_hat = out.ben_output.r_hat
        if regime.dim() == 2:
            valid = mask.bool()
            r_vals = regime[valid].cpu().numpy()
            rhats.extend(r_hat[valid].float().cpu().numpy().tolist())
        else:
            r_vals = regime.cpu().numpy()
            rhats.extend(r_hat.float().cpu().numpy().tolist())
        sub += int((r_vals == 0).sum())
        sup += int((r_vals == 1).sum())
        total += int(r_vals.size)

    out = {
        "n_positions": total,
        "frac_subcritical": sub / max(total, 1),
        "frac_supercritical": sup / max(total, 1),
        "rhat_mean": float(np.mean(rhats)) if rhats else None,
        "rhat_std": float(np.std(rhats)) if rhats else None,
    }
    log.info("Cap-D regime: sub=%.3f sup=%.3f rhat_mean=%s",
             out["frac_subcritical"], out["frac_supercritical"], out["rhat_mean"])
    return out


def main():
    args = parse_args()
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args, device)

    results = {
        "adapter": args.adapter,
        "backbone": args.backbone,
        "seed": args.seed,
    }

    log.info("=== Cap-A community purity ===")
    results["cap_a"] = cap_a_community_purity(model, tok, device, args, rng)
    log.info("=== Cap-B MAH boundary F1 ===")
    results["cap_b"] = cap_b_boundary_f1(model, tok, device, args, rng)
    log.info("=== Cap-C paraphrase stability ===")
    results["cap_c"] = cap_c_paraphrase_stability(model, tok, device, args)
    log.info("=== Cap-D BEN regime coverage ===")
    try:
        results["cap_d"] = cap_d_ben_regime(model, tok, device, args, rng)
    except Exception as e:
        log.warning("Cap-D failed: %s", e)
        results["cap_d"] = {"error": str(e)}

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
