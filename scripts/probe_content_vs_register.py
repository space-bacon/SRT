"""Content × register dissociation probe for SRT-NLA.

Pre-registered hypothesis (verbatim, do not edit after running):

    H1: median rho_intra-content > median rho_intra-register by >= 0.05
        in centered-cosine units (anisotropy-corrected).

    Falsifier: H1 fails  =>  the round-trip metric rho_norm is dominated
    by surface register rather than propositional content; paper_nla.md
    section 6 must be softened from "measures meaning" to a weaker claim
    about whatever-it-actually-measures.

Design (2x2x... matrix, 4 contents x 2 registers = 8 passages):

    C1  statistical mechanics / entropy / 2nd law
    C2  conservative force / path-independence / potentials
    C3  Shannon channel capacity / mutual information
    C4  self-modelling agency / strange loop

    R1  crisp formal one-line (~25 words, technical declarative)
    R2  ornate ontological ~120-word (voice of inputs #4/#5)

For each of the 8 passages we:
    1. Tokenize and run the frozen backbone, take L20 last-token hidden h*
       -> target matrix V in R^{8 x d}.
    2. Decode K=64 sampled rollouts via the AV (no v at inference: just
       feed v to the prefix and generate text).
    3. Re-encode every candidate text via the SAME backbone forward
       (no prefix, deployment-realistic) to get h_last in R^{B*K x d}.
    4. Score the K*8 = 512 candidate hiddens against ALL 8 targets in
       centered-cosine, yielding M_raw of shape (8, K, 8).
    5. Aggregate to a passage-level pairwise matrix M[i,j] = mean_k
       centered_cos(h_i,k - mu, v_j - mu) and decompose into:
         - self           : M[i,i]                     (4*2 = 8 numbers)
         - intra-content  : M[i,j] for (i,j) sharing C, differing R    (4 pairs)
         - intra-register : M[i,j] for (i,j) sharing R, differing C    (12 pairs)
         - cross-both     : M[i,j] for (i,j) differing in both         (12 pairs)
       (pairs symmetrised by averaging M[i,j] and M[j,i]).

Pool (size ~2000) for anisotropy mean mu is taken from the standard
targets shard; this matches centered_eval.py.

Example (Blackwell, after rsync of repo + targets shard):

    python scripts/probe_content_vs_register.py \\
        --av-ckpt   artifacts/nla/ce_seq64_np16/best_av.pt \\
        --targets   artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt \\
        --backbone  Qwen/Qwen2.5-7B --layer 20 \\
        --num-prefix-tokens 16 --num-inject-slots 1 \\
        --K 64 --rollout-len 64 --pool-size 2000 \\
        --outdir    artifacts/nla/content_vs_register/

Outputs (in --outdir):
    metrics.json         summary numbers + H1 verdict
    pairwise.json        full 8x8 mean-centered-cos matrix
    verbalizations.jsonl one record per (passage, candidate)
    heatmap.png          8x8 heatmap of M[i,j]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srt.nla.config import NLAConfig
from srt.nla.verbalizer import ActivationVerbalizer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)
log = logging.getLogger("probe_cvr")


# -------------------- the 8 hand-drafted passages --------------------
# row order is (content_id, register_id); we keep an explicit list so the
# matrix layout in metrics.json is stable.

PASSAGES: list[dict] = [
    {
        "id": "C1_R1",
        "content": "C1_entropy",
        "register": "R1_crisp",
        "text": (
            "Entropy of a closed thermodynamic system, defined as the "
            "logarithm of the number of accessible microstates, is "
            "non-decreasing in time."
        ),
    },
    {
        "id": "C1_R2",
        "content": "C1_entropy",
        "register": "R2_ornate",
        "text": (
            "Consider the unfathomed register of microstates beneath a "
            "single macroscopic gesture: each configuration a distinct "
            "way the world could have been, all collapsed into the "
            "indifferent shorthand of pressure, volume, temperature. "
            "Entropy is the cardinality of this hidden multiplicity made "
            "legible \u2014 the logarithm by which we count alternatives "
            "we will never inspect. Closed against its surroundings, a "
            "system cannot decrease this count; the arrow of time is not "
            "a metaphor but a consequence of arithmetic, the inexorable "
            "broadening of the space of equivalent realisations. What we "
            "call irreversibility is only the asymmetry between the few "
            "and the many."
        ),
    },
    {
        "id": "C2_R1",
        "content": "C2_conservative_force",
        "register": "R1_crisp",
        "text": (
            "A force field is conservative if and only if its line "
            "integral between any two points is independent of the path "
            "taken."
        ),
    },
    {
        "id": "C2_R2",
        "content": "C2_conservative_force",
        "register": "R2_ornate",
        "text": (
            "Imagine carrying a stone from valley to summit by every "
            "conceivable route \u2014 the meandering, the steep, the "
            "spiral, the impossible. If, upon arrival, the work done "
            "against the field is the same regardless of which thread "
            "of trajectory you chose, then the field harbours a hidden "
            "scalar from which every force in it descends as a gradient "
            "\u2014 a potential whose mere existence renders history "
            "irrelevant. Such a field forgets the journey and remembers "
            "only the endpoints; it admits a potential, and that "
            "potential is the field's deepest signature. "
            "Path-independence is not an accident of geometry but the "
            "very mark of conservation: the silent insistence that what "
            "is owed depends not on how we travelled, only where."
        ),
    },
    {
        "id": "C3_R1",
        "content": "C3_channel_capacity",
        "register": "R1_crisp",
        "text": (
            "The maximum information transmission rate of a noisy "
            "channel equals the supremum of mutual information between "
            "input and output over all input distributions."
        ),
    },
    {
        "id": "C3_R2",
        "content": "C3_channel_capacity",
        "register": "R2_ornate",
        "text": (
            "Every channel is a place where signal meets the "
            "indifference of medium, and the question is how much of the "
            "speaker's intent survives the crossing. Shannon's measure "
            "does not answer how loud one must shout; it asks instead "
            "how cleverly one may choose what to say. Across all "
            "possible distributions of source \u2014 every conceivable "
            "rationing of the alphabet \u2014 there exists one that "
            "maximises the dependence between what is sent and what is "
            "received. That maximum, expressed in bits per use, is the "
            "channel's capacity: the irreducible bandwidth of meaning "
            "the noise will permit."
        ),
    },
    {
        "id": "C4_R1",
        "content": "C4_self_modelling",
        "register": "R1_crisp",
        "text": (
            "An agent that maintains an internal predictive model of "
            "itself acquires representations whose targets are the very "
            "representations doing the modelling."
        ),
    },
    {
        "id": "C4_R2",
        "content": "C4_self_modelling",
        "register": "R2_ornate",
        "text": (
            "When a system grows large enough to fold a model of itself "
            "inside itself, something strange occurs at the seam where "
            "the modelling tissue meets the modelled. The representations "
            "that estimate behaviour become, simultaneously, the "
            "substrate they are trying to estimate; prediction and "
            "predicted are no longer in clean opposition but locked in "
            "mutual definition. Each update revises both portrait and "
            "sitter at once. This is not paradox but architecture: "
            "agency, in any non-trivial sense, requires that the loop "
            "close \u2014 that the map and the territory share a single "
            "sheet of paper."
        ),
    },
]


# -------------------- math helpers (mirror centered_eval.py) --------------------

def fve_from_cos(c: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + c)


def cos_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(-1)


@torch.no_grad()
def encode_passages_l20_last(
    backbone, tok, passages: list[str], layer: int, device: str, max_len: int = 256
) -> torch.Tensor:
    """Tokenize each passage, run backbone, return last-token hidden at `layer`.

    Returns (N, d) float32 on device.
    """
    out = []
    for txt in passages:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = ids["input_ids"].to(device)
        attn = ids["attention_mask"].to(device)
        fwd = backbone(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        h = fwd.hidden_states[layer]                  # (1, T, d)
        last_idx = int(attn.sum(-1).item()) - 1
        out.append(h[0, last_idx, :].detach().to(torch.float32))
    return torch.stack(out, 0)                        # (N, d)


@torch.no_grad()
def extract_last_hidden(backbone, gen_ids, gen_attn, layer):
    fwd = backbone(
        input_ids=gen_ids,
        attention_mask=gen_attn,
        output_hidden_states=True,
        use_cache=False,
    )
    h = fwd.hidden_states[layer]
    last = gen_attn.sum(-1) - 1
    idx = last.clamp(min=0).long()
    rows = torch.arange(h.size(0), device=h.device)
    return h[rows, idx, :].detach().to(torch.float32)


@torch.no_grad()
def rollout(av, backbone, v_batch, K, T, temperature, layer, eos_id):
    B = v_batch.size(0)
    v_rep = v_batch.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    gen_ids = av.generate(
        v_rep,
        max_new_tokens=T,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-6),
        top_p=1.0,
    )
    eos_mask = (gen_ids == eos_id)
    first_eos = torch.where(
        eos_mask.any(-1),
        eos_mask.float().argmax(-1),
        torch.full((gen_ids.size(0),), gen_ids.size(1) - 1, device=gen_ids.device),
    )
    arange = torch.arange(gen_ids.size(1), device=gen_ids.device).unsqueeze(0)
    attn = (arange <= first_eos.unsqueeze(1)).long()
    h_last = extract_last_hidden(backbone, gen_ids, attn, layer)         # (B*K, d)
    return h_last.view(B, K, -1), gen_ids.view(B, K, -1), attn.view(B, K, -1)


# -------------------- decomposition helpers --------------------

def _pair_means(M: torch.Tensor, idx_pairs: list[tuple[int, int]]) -> list[float]:
    """Symmetrised mean of M[i,j], M[j,i] for each (i,j) pair."""
    return [float(0.5 * (M[i, j] + M[j, i])) for (i, j) in idx_pairs]


def _build_index_pairs(passages: list[dict]) -> dict:
    n = len(passages)
    intra_content, intra_register, cross_both = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            same_c = passages[i]["content"] == passages[j]["content"]
            same_r = passages[i]["register"] == passages[j]["register"]
            if same_c and not same_r:
                intra_content.append((i, j))
            elif same_r and not same_c:
                intra_register.append((i, j))
            elif (not same_c) and (not same_r):
                cross_both.append((i, j))
            # same_c and same_r is impossible by construction
    return {
        "intra_content": intra_content,
        "intra_register": intra_register,
        "cross_both": cross_both,
    }


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-ckpt", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path,
                    help="standard targets .pt shard, used only for anisotropy mean")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--num-prefix-tokens", type=int, default=16)
    ap.add_argument("--prefix-mode", choices=["static", "mlp"], default="static")
    ap.add_argument("--prefix-mlp-hidden", type=int, default=256)
    ap.add_argument("--num-inject-slots", type=int, default=1)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-vectors", type=int, default=4,
                    help="passages per AV.generate call (B*K rollouts at once)")
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h1-threshold", type=float, default=0.05,
                    help="pre-registered minimum centered-cos gap for H1")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- load backbone + AV ----------
    log.info("loading backbone %s", args.backbone)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id or eos_id
    if backbone.config.pad_token_id is None:
        backbone.config.pad_token_id = pad_id

    cfg = NLAConfig(
        backbone_id=args.backbone,
        num_prefix_tokens=args.num_prefix_tokens,
        extraction_layer=args.layer,
        prefix_mode=args.prefix_mode,
        prefix_mlp_hidden=args.prefix_mlp_hidden,
        num_inject_slots=args.num_inject_slots,
    )
    av = ActivationVerbalizer(cfg, backbone, tok).to(device).eval()

    log.info("warm-start from %s", args.av_ckpt)
    sd = torch.load(args.av_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("trainable", "av", "av_state_dict", "model", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    missing, unexpected = av.load_state_dict(sd, strict=False)
    log.info("warm-start: missing=%d unexpected=%d", len(missing), len(unexpected))

    # ---------- pool for anisotropy mean ----------
    log.info("loading pool %s", args.targets)
    tgt = torch.load(args.targets, map_location="cpu", weights_only=False)
    acts = tgt["activations"]
    pool_all = torch.stack([a[-1].float() for a in acts], 0)
    P = min(args.pool_size, pool_all.size(0))
    pool = pool_all[:P].to(device)
    mu = pool.mean(dim=0, keepdim=True)
    log.info("pool=%d  ||mu||=%.4f", P, float(mu.norm()))

    # ---------- encode the 8 passages ----------
    texts = [p["text"] for p in PASSAGES]
    log.info("encoding %d passages at L%d", len(texts), args.layer)
    V = encode_passages_l20_last(backbone, tok, texts, args.layer, device)   # (N, d)
    N = V.size(0)

    # ---------- decode K rollouts per passage ----------
    K = args.K
    T = args.rollout_len
    B = args.batch_vectors

    log.info("rolling out N=%d K=%d T=%d (=%d candidates)", N, K, T, N * K)
    h_sampled_chunks = []
    gen_ids_chunks = []
    gen_attn_chunks = []
    for i in range(0, N, B):
        v = V[i:i + B]
        h_chunk, ids_chunk, attn_chunk = rollout(
            av, backbone, v, K, T, args.temperature, args.layer, eos_id
        )
        h_sampled_chunks.append(h_chunk)
        gen_ids_chunks.append(ids_chunk)
        gen_attn_chunks.append(attn_chunk)
        log.info("  passages %d-%d done", i, i + v.size(0))
    h_sampled = torch.cat(h_sampled_chunks, 0)        # (N, K, d)
    gen_ids = torch.cat(gen_ids_chunks, 0)            # (N, K, T_max)
    gen_attn = torch.cat(gen_attn_chunks, 0)          # (N, K, T_max)

    # ---------- score every candidate against every target (centered cos) ----------
    h_c = h_sampled - mu                               # (N, K, d)
    V_c = V - mu                                       # (N, d)
    h_flat = h_c.reshape(N * K, -1)
    h_n = F.normalize(h_flat.float(), dim=-1)
    V_n = F.normalize(V_c.float(), dim=-1)
    cos_full = h_n @ V_n.t()                           # (N*K, N)
    cos_full = cos_full.view(N, K, N)                  # (i, k, j)

    # diagonal stats (target = source passage)
    diag = torch.stack([cos_full[i, :, i] for i in range(N)], 0)   # (N, K)
    self_mean_K = diag.mean(dim=1)                                  # (N,)
    self_max_K = diag.max(dim=1).values                              # best-of-K per passage
    self_min_K = diag.min(dim=1).values

    # passage-level pairwise matrix M[i,j] = mean_k cos(h_i,k - mu, v_j - mu)
    M = cos_full.mean(dim=1)                                         # (N, N)
    M_cpu = M.detach().cpu()

    # ---------- decompose by content/register structure ----------
    pair_idx = _build_index_pairs(PASSAGES)
    intra_content_vals = _pair_means(M_cpu, pair_idx["intra_content"])
    intra_register_vals = _pair_means(M_cpu, pair_idx["intra_register"])
    cross_both_vals = _pair_means(M_cpu, pair_idx["cross_both"])

    rho_intra_content = median(intra_content_vals)
    rho_intra_register = median(intra_register_vals)
    rho_cross_both = median(cross_both_vals)

    h1_gap = rho_intra_content - rho_intra_register
    h1_pass = bool(h1_gap >= args.h1_threshold)

    # ---------- decode candidate text + write verbalizations.jsonl ----------
    verb_path = args.outdir / "verbalizations.jsonl"
    with verb_path.open("w") as f:
        for i in range(N):
            for k in range(K):
                ids_ik = gen_ids[i, k]
                attn_ik = gen_attn[i, k]
                n_keep = int(attn_ik.sum().item())
                txt = tok.decode(ids_ik[:n_keep].tolist(), skip_special_tokens=True)
                rec = {
                    "passage_id": PASSAGES[i]["id"],
                    "content": PASSAGES[i]["content"],
                    "register": PASSAGES[i]["register"],
                    "k": k,
                    "self_centered_cos": float(diag[i, k]),
                    "centered_cos_vs_all": [float(cos_full[i, k, j]) for j in range(N)],
                    "text": txt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("wrote %s", verb_path)

    # ---------- write pairwise.json ----------
    pairwise = {
        "passages": [{"id": p["id"], "content": p["content"], "register": p["register"]} for p in PASSAGES],
        "M_mean_centered_cos": M_cpu.tolist(),
        "diag_self_mean_K": self_mean_K.detach().cpu().tolist(),
        "diag_self_max_K": self_max_K.detach().cpu().tolist(),
        "diag_self_min_K": self_min_K.detach().cpu().tolist(),
        "intra_content_pairs": [
            {"i": i, "j": j,
             "ids": [PASSAGES[i]["id"], PASSAGES[j]["id"]],
             "value": float(0.5 * (M_cpu[i, j] + M_cpu[j, i]))}
            for (i, j) in pair_idx["intra_content"]
        ],
        "intra_register_pairs": [
            {"i": i, "j": j,
             "ids": [PASSAGES[i]["id"], PASSAGES[j]["id"]],
             "value": float(0.5 * (M_cpu[i, j] + M_cpu[j, i]))}
            for (i, j) in pair_idx["intra_register"]
        ],
        "cross_both_pairs": [
            {"i": i, "j": j,
             "ids": [PASSAGES[i]["id"], PASSAGES[j]["id"]],
             "value": float(0.5 * (M_cpu[i, j] + M_cpu[j, i]))}
            for (i, j) in pair_idx["cross_both"]
        ],
    }
    (args.outdir / "pairwise.json").write_text(json.dumps(pairwise, indent=2))

    # ---------- metrics.json + verdict ----------
    metrics = {
        "backbone": args.backbone,
        "layer": args.layer,
        "av_ckpt": str(args.av_ckpt),
        "targets_pool": str(args.targets),
        "pool_size": int(P),
        "anisotropy_mu_norm": float(mu.norm()),
        "N": N,
        "K": K,
        "T": T,
        "temperature": args.temperature,

        # diagonal (self-decode quality, centered cos units)
        "self_centered_cos_mean_over_passages_mean_K":
            float(self_mean_K.mean()),
        "self_centered_cos_mean_over_passages_max_K":
            float(self_max_K.mean()),
        "per_passage_self_mean_K": {
            PASSAGES[i]["id"]: float(self_mean_K[i]) for i in range(N)
        },
        "per_passage_self_max_K": {
            PASSAGES[i]["id"]: float(self_max_K[i]) for i in range(N)
        },

        # decomposition (centered cos units)
        "rho_intra_content_median": rho_intra_content,
        "rho_intra_register_median": rho_intra_register,
        "rho_cross_both_median": rho_cross_both,
        "h1_threshold": args.h1_threshold,
        "h1_gap": h1_gap,
        "h1_pass": h1_pass,
        "h1_statement": (
            "median rho_intra-content > median rho_intra-register by "
            f">= {args.h1_threshold} centered-cos units"
        ),
        "verdict": (
            "H1_PASS: content dominates register"
            if h1_pass
            else "H1_FAIL: register-dominated; soften paper section 6"
        ),

        # anchored to published rho_norm scale (floor 0.020, ceiling 0.598
        # in centered-cos), reported alongside but NOT used for H1
        "rho_norm_anchors": {"floor_centered_cos": 0.020, "ceiling_centered_cos": 0.598},
        "rho_intra_content_norm":
            (rho_intra_content - 0.020) / (0.598 - 0.020),
        "rho_intra_register_norm":
            (rho_intra_register - 0.020) / (0.598 - 0.020),
    }
    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- heatmap.png (best-effort, skip if matplotlib unavailable) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [p["id"] for p in PASSAGES]
        M_np = np.asarray(M_cpu)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M_np, cmap="viridis", vmin=-0.05, vmax=0.65)
        ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(N)); ax.set_yticklabels(labels)
        ax.set_xlabel("target passage v_j")
        ax.set_ylabel("source passage v_i (rollout origin)")
        ax.set_title(
            "Mean centered-cos of K rollouts vs. all targets\n"
            f"H1 gap (intra-content - intra-register) = {h1_gap:+.3f}  "
            f"[threshold {args.h1_threshold:+.3f}]  -> "
            f"{'PASS' if h1_pass else 'FAIL'}"
        )
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{M_np[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if M_np[i, j] < 0.35 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(args.outdir / "heatmap.png", dpi=150)
        plt.close(fig)
        log.info("wrote %s", args.outdir / "heatmap.png")
    except Exception as e:
        log.warning("heatmap skipped: %s", e)

    # ---------- terminal report ----------
    log.info("======== content x register probe ========")
    log.info("self  mean-K  (avg over 8) = %.4f", metrics["self_centered_cos_mean_over_passages_mean_K"])
    log.info("self  max-K   (avg over 8) = %.4f", metrics["self_centered_cos_mean_over_passages_max_K"])
    log.info("rho_intra_content  median  = %.4f  (n=%d pairs)",
             rho_intra_content, len(intra_content_vals))
    log.info("rho_intra_register median  = %.4f  (n=%d pairs)",
             rho_intra_register, len(intra_register_vals))
    log.info("rho_cross_both     median  = %.4f  (n=%d pairs)",
             rho_cross_both, len(cross_both_vals))
    log.info("H1 gap (content - register) = %+.4f  [threshold %+.4f]",
             h1_gap, args.h1_threshold)
    log.info("verdict: %s", metrics["verdict"])
    log.info("wrote %s", args.outdir / "metrics.json")


if __name__ == "__main__":
    main()
