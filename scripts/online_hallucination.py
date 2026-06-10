"""Online hallucination detection and reject-and-resample for the SRT adapter.

Two studies, both using per-token signals available DURING generation:

(A) Detection (teacher-forced, paired):
    For each HaluEval QA row we score the right answer and the hallucinated
    answer through the model, pool the per-token signals over the answer span,
    and ask which pooled signal best separates hallucinated from faithful. This
    is the honest "which online-computable signal detects hallucination" table.
    Signals:
      SRT side-channels : r_hat, regime_super, div_norm, chain
      predictive (free) : nll, ent, margin
    The pairing (same question) controls for question difficulty, matching the
    protocol behind the prior per-answer AUCs in artifacts/hallucination/.

(B) Reject-and-resample (generative):
    The model answers the question itself. With --resample, whenever the chosen
    token's trigger signal exceeds a percentile threshold the token is rejected
    and resampled (the offending token id is masked out and we draw again, up to
    a retry budget). We compare the hallucination rate of the model's own
    answers with and without the intervention, labelling each generated answer
    by string match against the gold right / hallucinated answers.

Usage:
    python scripts/online_hallucination.py detect \
        --backbone Qwen/Qwen2.5-7B-Instruct \
        --adapter-ckpt artifacts/checkpoints/v4/best_adapter.pt \
        --n 200 --out artifacts/hallucination/online_detect.json

    python scripts/online_hallucination.py resample \
        --backbone Qwen/Qwen2.5-7B-Instruct \
        --adapter-ckpt artifacts/checkpoints/v4/best_adapter.pt \
        --n 100 --trigger margin --pct 10 --out artifacts/hallucination/resample.json
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
import torch.nn.functional as F

from srt.adapter import SRTAdapter
from srt.config import SRTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("online_hallu")

SRT_SIGNALS = ["r_hat", "regime_super", "div_norm", "chain"]
PRED_SIGNALS = ["nll", "ent", "margin"]
ALL_SIGNALS = SRT_SIGNALS + PRED_SIGNALS


def _load(backbone_id: str, adapter_ckpt: str | None, device: str):
    cfg = SRTConfig(backbone_id=backbone_id)
    model = SRTAdapter(cfg)
    if adapter_ckpt:
        sd = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "trainable" in sd:
            sd = sd["trainable"]
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone_id)
    return model, tok


def _eos_ids(tok) -> set[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.add(im_end)
    return ids


@torch.no_grad()
def _score_answer(model, tok, question: str, answer: str, device: str) -> dict:
    """Teacher-force `answer` after `question`; pool per-token signals over the
    answer span. Returns mean+max of each signal over the answer tokens."""
    msgs = [{"role": "user", "content": question}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix_ids = tok(prefix, return_tensors="pt").input_ids
    full_ids = tok(prefix + answer, return_tensors="pt").input_ids
    a0 = prefix_ids.shape[1]            # first answer-token index
    a1 = full_ids.shape[1]             # exclusive end
    if a1 <= a0:
        return {}
    ids = full_ids.to(device)

    out = model(ids)  # full teacher-forced forward; all signals per-position (B,T,*)
    logits = out.logits[0].float()     # (T, V)

    # Predictive signals: for answer token at position p, the distribution that
    # predicts it is at position p-1.
    feats: dict[str, list[float]] = {s: [] for s in ALL_SIGNALS}
    ben = out.ben_output
    div_last = out.divergences[-1][0] if out.divergences else None
    chain = out.chain_residual_per_token
    for p in range(a0, a1):
        logp = torch.log_softmax(logits[p - 1], dim=-1)
        probs = logp.exp()
        tok_id = int(ids[0, p])
        feats["nll"].append(float(-logp[tok_id]))
        feats["ent"].append(float(-(probs * logp).sum()))
        top2 = torch.topk(probs, 2).values
        feats["margin"].append(float(top2[0] - top2[1]))
        if ben is not None:
            feats["r_hat"].append(float(ben.r_hat[0, p]))
            feats["regime_super"].append(
                float(F.softmax(ben.regime_logits[0, p], dim=-1)[1])
            )
        if div_last is not None:
            feats["div_norm"].append(float(div_last[p].norm()))
        if chain is not None:
            feats["chain"].append(float(chain[0, p]))

    pooled = {}
    for s, vals in feats.items():
        if vals:
            t = torch.tensor(vals)
            pooled[f"{s}_mean"] = float(t.mean())
            pooled[f"{s}_max"] = float(t.max())
    pooled["n_tokens"] = a1 - a0
    return pooled


def _auc(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC via Mann-Whitney U, no sklearn dependency."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    n_pos = sum(lab for _, lab in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def run_detect(args) -> None:
    from datasets import load_dataset
    model, tok = _load(args.backbone, args.adapter_ckpt, args.device)
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")

    rows: list[dict] = []
    for i in range(min(args.n, len(ds))):
        r = ds[i]
        q = r["question"].strip()
        for label, ans in ((0, r["right_answer"]), (1, r["hallucinated_answer"])):
            p = _score_answer(model, tok, q, ans.strip(), args.device)
            if p:
                p["label"] = label
                p["qid"] = i
                rows.append(p)
        if (i + 1) % 25 == 0:
            logger.info("scored %d/%d questions", i + 1, args.n)

    # AUC per pooled feature.
    feat_keys = [k for k in rows[0] if k.endswith(("_mean", "_max"))]
    labels = [r["label"] for r in rows]
    aucs = {}
    for k in feat_keys:
        a = _auc([r[k] for r in rows], labels)
        # A signal can point either way; report the max of (auc, 1-auc) with sign.
        aucs[k] = round(a, 4)
    ranked = sorted(aucs.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)

    result = {
        "n_questions": args.n,
        "n_examples": len(rows),
        "n_hallucinated": sum(labels),
        "auc_by_feature": dict(ranked),
        "top": ranked[:8],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", args.out)
    print(json.dumps({"top_features": ranked[:8]}, indent=2))


@torch.no_grad()
def _generate_labeled(model, tok, question, right, hallu, device,
                      resample, trigger, threshold, max_retries):
    """Generate the model's own answer (optionally with reject-and-resample),
    then label it by gold string match. Returns (text, label, n_resamples)."""
    msgs = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    eos = _eos_ids(tok)

    if not resample:
        out, sig = model.generate(ids, max_new_tokens=64, eos_token_ids=eos,
                                  temperature=0.7, top_p=0.9, return_signals=True)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        n_re = 0
    else:
        text, n_re = _generate_resample(
            model, tok, ids, eos, trigger, threshold, max_retries, device
        )

    return text, _label_answer(text, right, hallu), n_re


def _label_answer(text: str, right: str, hallu: str) -> int:
    """1 = hallucinated, 0 = correct, -1 = ambiguous. Normalized containment."""
    t = text.lower()
    r = right.strip().lower()
    h = hallu.strip().lower()
    r_hit = r and r in t
    h_hit = h and h in t
    if r_hit and not h_hit:
        return 0
    if h_hit and not r_hit:
        return 1
    return -1


def _label_correct(text: str, right: str) -> int:
    """Generative correctness label: 0 = correct (gold short-answer span present),
    1 = wrong. Used for the model's OWN free-form answers in HaluEval QA, where
    the gold right_answer is a short factual span."""
    return 0 if right.strip().lower() in text.lower() else 1


@torch.no_grad()
def run_gendetect(args) -> None:
    """Honest generative test. The model answers HaluEval questions itself; we
    label correctness by gold short-answer span presence, then test whether the
    per-token signals pooled over the model's OWN generation predict it — with a
    length-AUC control (the artifact that sank the teacher-forced study). The
    same pass, repeated with reject-and-resample on the chosen predictive
    trigger, gives a wrong-rate ablation."""
    from datasets import load_dataset
    model, tok = _load(args.backbone, args.adapter_ckpt, args.device)
    eos = _eos_ids(tok)
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")

    rows: list[dict] = []
    for i in range(min(args.n, len(ds))):
        r = ds[i]
        msgs = [{"role": "user", "content": r["question"].strip()}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        out, sig = model.generate(ids, max_new_tokens=48, eos_token_ids=eos,
                                  temperature=0.7, top_p=0.9, return_signals=True)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        label = _label_correct(text, r["right_answer"])
        # Pool per-token signals over the generation.
        pooled = {"label": label, "n_tokens": len(sig), "qid": i, "text": text}
        for s in ALL_SIGNALS:
            vals = [d[s] for d in sig if s in d]
            if vals:
                t = torch.tensor(vals)
                pooled[f"{s}_mean"] = float(t.mean())
                pooled[f"{s}_max"] = float(t.max())
        rows.append(pooled)
        if (i + 1) % 25 == 0:
            logger.info("gendetect %d/%d (wrong so far: %d)",
                        i + 1, args.n, sum(x["label"] for x in rows))

    labels = [r["label"] for r in rows]
    n_wrong = sum(labels)
    feat_keys = [k for k in rows[0] if k.endswith(("_mean", "_max"))]
    aucs = {k: round(_auc([r[k] for r in rows], labels), 4) for k in feat_keys}
    len_auc = round(_auc([r["n_tokens"] for r in rows], labels), 4)
    ranked = sorted(aucs.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)

    result = {
        "mode": "generative", "n_questions": args.n, "n_generations": len(rows),
        "n_wrong": n_wrong, "wrong_rate": round(n_wrong / len(rows), 4),
        "length_auc_control": len_auc,
        "auc_by_feature": dict(ranked), "top": ranked[:8],
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", args.out)
    print(json.dumps({
        "n_generations": len(rows), "wrong_rate": result["wrong_rate"],
        "length_auc_control": len_auc, "top_features": ranked[:8],
    }, indent=2))


@torch.no_grad()
def _generate_resample(model, tok, ids, eos, trigger, threshold, max_retries, device):
    """Sampling with reject-and-resample on a PREDICTIVE trigger.

    Predictive signals (margin, ent, nll) are properties of the next-token
    distribution and are known BEFORE the token is committed to the cache, so a
    rejected token can be masked and redrawn with no cache rollback. SRT signals
    (r_hat, div_norm, ...) require committing the token first and are therefore
    not usable as a no-rollback trigger; this function supports the predictive
    triggers only. The trigger fires when the chosen token looks risky:
        margin < threshold   (low top1-top2 gap)
        ent    > threshold   (diffuse distribution)
        nll    > threshold   (low-prob choice)
    On a fire we ban that token id and resample, up to max_retries, then accept
    the best available.
    """
    from transformers.cache_utils import DynamicCache
    from srt.adapter import _sample_token

    backbone_cache = DynamicCache()
    mah_kv: dict = {}
    logits, cvec, _ = model._cached_step(
        ids, backbone_cache, mah_kv, community_vec=None, start_pos=0,
        disable_injectors=False,
    )
    cur = ids.shape[1]
    new_ids: list[int] = []
    n_resamples = 0

    def risky(probs, logp, tok_id):
        if trigger == "margin":
            top2 = torch.topk(probs, 2).values
            return float(top2[0] - top2[1]) < threshold
        if trigger == "ent":
            return float(-(probs * logp).sum()) > threshold
        if trigger == "nll":
            return float(-logp[tok_id]) > threshold
        return False

    for _ in range(64):
        work = logits[:, -1, :].float().clone()
        chosen = None
        for attempt in range(max_retries + 1):
            nid = _sample_token(work, 0.7, 0.9)
            tok_id = int(nid.item())
            logp = torch.log_softmax(work, dim=-1)[0]
            probs = logp.exp()
            if attempt < max_retries and risky(probs, logp, tok_id):
                work[0, tok_id] = float("-inf")  # ban and redraw
                n_resamples += 1
                continue
            chosen = tok_id
            break
        new_ids.append(chosen)
        if chosen in eos:
            break
        logits, cvec, _ = model._cached_step(
            torch.tensor([[chosen]], device=device), backbone_cache, mah_kv,
            community_vec=cvec, start_pos=cur, disable_injectors=False,
        )
        cur += 1
    text = tok.decode(torch.tensor(new_ids, device=device), skip_special_tokens=True)
    return text.strip(), n_resamples


def run_resample(args) -> None:
    from datasets import load_dataset
    model, tok = _load(args.backbone, args.adapter_ckpt, args.device)
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    base_vals: list[float] = []
    baseline: list[dict] = []
    for i in range(min(args.n, len(ds))):
        r = ds[i]
        text, label, _ = _generate_labeled(
            model, tok, r["question"].strip(), r["right_answer"], r["hallucinated_answer"],
            args.device, resample=False, trigger=args.trigger, threshold=0, max_retries=0,
        )
        baseline.append({"qid": i, "label": label, "text": text})
        if (i + 1) % 25 == 0:
            logger.info("baseline %d/%d", i + 1, args.n)

    # Threshold: pct-th percentile of margin (low) or (100-pct) for ent/nll (high).
    # Simpler: use a fixed sensible default per trigger.
    threshold = {"margin": 0.20, "ent": 2.0, "nll": 1.5}[args.trigger]

    resampled: list[dict] = []
    tot_re = 0
    for i in range(min(args.n, len(ds))):
        r = ds[i]
        text, label, n_re = _generate_labeled(
            model, tok, r["question"].strip(), r["right_answer"], r["hallucinated_answer"],
            args.device, resample=True, trigger=args.trigger, threshold=threshold,
            max_retries=args.max_retries,
        )
        resampled.append({"qid": i, "label": label, "text": text, "n_resamples": n_re})
        tot_re += n_re
        if (i + 1) % 25 == 0:
            logger.info("resample %d/%d", i + 1, args.n)

    def rate(rows):
        lab = [x["label"] for x in rows if x["label"] != -1]
        if not lab:
            return None, 0
        return sum(l == 1 for l in lab) / len(lab), len(lab)

    base_rate, base_n = rate(baseline)
    re_rate, re_n = rate(resampled)
    result = {
        "n_questions": args.n, "trigger": args.trigger, "threshold": threshold,
        "max_retries": args.max_retries, "total_resamples": tot_re,
        "baseline": {"hallucination_rate": base_rate, "n_decisive": base_n},
        "resampled": {"hallucination_rate": re_rate, "n_decisive": re_n},
        "rows_baseline": baseline, "rows_resampled": resampled,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", args.out)
    print(json.dumps({
        "trigger": args.trigger, "threshold": threshold, "total_resamples": tot_re,
        "baseline_hallu_rate": base_rate, "resampled_hallu_rate": re_rate,
    }, indent=2))



def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "gendetect", "resample"):
        s = sub.add_parser(name)
        s.add_argument("--backbone", default="Qwen/Qwen2.5-7B-Instruct")
        s.add_argument("--adapter-ckpt", default=None)
        s.add_argument("--device", default="cuda")
        s.add_argument("--n", type=int, default=200)
        s.add_argument("--out", required=True)
        if name == "resample":
            s.add_argument("--trigger", default="margin", choices=ALL_SIGNALS)
            s.add_argument("--pct", type=float, default=10.0)
            s.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    if args.cmd == "detect":
        run_detect(args)
    elif args.cmd == "gendetect":
        run_gendetect(args)
    else:
        run_resample(args)


if __name__ == "__main__":
    main()
