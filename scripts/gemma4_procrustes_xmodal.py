"""Cross-modal Procrustes alignment on gemma-4-31B-it L47 (Sunstone §11.6.4).

Question: after per-modality centering, is the remaining image↔text modality
gap a rigid ROTATION? Fit orthogonal Procrustes W (image→text) on paired
COCO val2017 image/caption L47 states and measure held-out retrieval.

Because W is orthogonal it cannot add information — any R@K gain over
centered-cosine means the information was present but misaligned; a null
means centered cosine already extracts everything a linear map can.

Protocol
  1. Download COCO val2017 (5k images, 5 captions each) into --work-dir.
  2. Encode N images (mean over image-token positions, hidden_states[L])
     and their captions (last-token, BOS-enforced, seq<=64 — matches
     sample_targets.py corpus convention and the §11.7 BOS gotcha).
  3. Split fit/eval by image. Center each modality by ITS FIT-SPLIT MEAN.
  4. Fit W = UV^T from SVD(X^T Y). Controls: shuffled-pairs fit (floor),
     train-size curve, PCA-subspace variants (n < d conditioning check).
  5. Eval on held-out images: image→text R@1/5/10 against the 5*n_eval
     caption pool (hit = any of the image's 5 captions), both directions,
     raw-centered vs projected vs shuffled-control.
  6. Optional boundary probes (--probe-images + --caption-targets): rank
     the synthetic white-heart caption etc. in the 10k pool before/after
     projection (§11.6.3 boundary: does the heart move up from 352?).

All heavy stages cache to --work-dir so the script is resumable.

    nohup python scripts/gemma4_procrustes_xmodal.py \
        --work-dir /root/procrustes \
        --caption-targets /root/artifacts/targets_caps_L47.pt \
        --probe-images artifacts/nla/gemma4/stereo/control.png \
        --out artifacts/nla/gemma4/procrustes/procrustes_xmodal.json \
        > /root/procrustes/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from srt.nla import load_frozen_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("procrustes")

# COCO's own hostname is a CNAME to an S3 bucket whose wildcard cert does not
# cover the dotted name (https fails with hostname mismatch). Path-style S3
# addressing keeps a valid TLS cert.
COCO_IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="google/gemma-4-31B-it")
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--quant4", action="store_true",
                   help="load the backbone in 4-bit NF4 (quantization-drift runs)")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="cache dir for COCO data + encoded states")
    p.add_argument("--n-images", type=int, default=5000,
                   help="total COCO val2017 images to encode (max 5000)")
    p.add_argument("--n-eval", type=int, default=1000,
                   help="held-out images for retrieval eval")
    p.add_argument("--fit-sizes", type=int, nargs="+", default=[1000, 2000, 4000],
                   help="train-size curve; last entry = headline fit")
    p.add_argument("--pca-dims", type=int, nargs="+", default=[256, 1024],
                   help="PCA-subspace Procrustes variants (n<d conditioning)")
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--caption-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--caption-targets", type=Path, default=None,
                   help="10k caption pool .pt (sample_targets format) for boundary probes")
    p.add_argument("--probe-images", nargs="*", type=Path, default=[],
                   help="boundary probe images (e.g. stereo/control.png)")
    p.add_argument("--probe-expected", nargs="*", default=[],
                   help="per probe image: substring of its exact caption in the "
                        "pool, for rank reporting (align with --probe-images)")
    p.add_argument("--processor-cache", default=None)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


# --------------------------------------------------------------------- data
def ensure_coco(work: Path) -> tuple[Path, list[dict]]:
    """Download/extract COCO val2017; return (image_dir, [{file, captions}])."""
    img_dir = work / "val2017"
    ann_json = work / "annotations" / "captions_val2017.json"
    for url, marker in ((COCO_IMAGES_URL, img_dir),
                        (COCO_ANN_URL, ann_json)):
        if marker.exists():
            continue
        zpath = work / Path(url).name
        if not zpath.exists():
            log.info("downloading %s ...", url)
            urllib.request.urlretrieve(url, zpath)  # noqa: S310 — fixed first-party URLs
        log.info("extracting %s ...", zpath.name)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(work)
    ann = json.loads(ann_json.read_text())
    caps_by_img: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        caps_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
    rows = []
    for im in ann["images"]:
        caps = caps_by_img.get(im["id"], [])
        if len(caps) >= 5:
            rows.append({"file": str(img_dir / im["file_name"]),
                         "captions": caps[:5]})
    rows.sort(key=lambda r: r["file"])  # deterministic order
    log.info("COCO val2017: %d images with >=5 captions", len(rows))
    return img_dir, rows


# ----------------------------------------------------------------- encoders
class Encoder:
    def __init__(self, args):
        from transformers import AutoProcessor

        proc_kw = {"cache_dir": args.processor_cache} if args.processor_cache else {}
        self.proc = AutoProcessor.from_pretrained(args.backbone, **proc_kw)
        self.backbone, self.tok = load_frozen_backbone(
            args.backbone, args.dtype, device="cuda",
            quant4=getattr(args, "quant4", False))
        self.image_token_id = getattr(self.backbone.config, "image_token_id", None)
        assert self.image_token_id is not None, "backbone config lacks image_token_id"
        self.layer = args.layer
        self.max_seq_len = args.max_seq_len

    @torch.no_grad()
    def image_v(self, img) -> torch.Tensor:
        """Mean L-layer state over image-token positions (Sunstone convention)."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe this image."},
        ]}]
        enc = self.proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")
        out = self.backbone(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["input_ids"][0] == self.image_token_id
        return out.hidden_states[self.layer][0][mask].float().mean(0).cpu()

    @torch.no_grad()
    def text_vs(self, texts: list[str]) -> torch.Tensor:
        """Last-token L-layer states, BOS enforced when the tokenizer has one
        (§11.7: bare re-encode drops gemma-4 replay 0.9986 → 0.615). Qwen-style
        tokenizers have no BOS; add_special_tokens handles their convention."""
        tok = self.tok
        bos = tok.bos_token_id
        ids_list = []
        for t in texts:
            ids = tok(t, truncation=True, max_length=self.max_seq_len,
                      add_special_tokens=True).input_ids
            if bos is not None and ids[0] != bos:
                ids = [bos] + ids[: self.max_seq_len - 1]
            ids_list.append(ids)
        T = max(len(i) for i in ids_list)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (
            bos if bos is not None else 0)
        input_ids = torch.full((len(ids_list), T), pad, dtype=torch.long)
        attn = torch.zeros((len(ids_list), T), dtype=torch.long)
        for j, ids in enumerate(ids_list):  # right-pad
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn[j, : len(ids)] = 1
        out = self.backbone(
            input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
            output_hidden_states=True, use_cache=False)
        h = out.hidden_states[self.layer]
        last = attn.sum(-1) - 1
        rows = torch.arange(h.size(0))
        return h[rows, last].float().cpu()


def encode_all(args, rows: list[dict], get_encoder) -> dict:
    """Encode images + captions with caching. Returns dict of tensors."""
    cache = args.work_dir / f"encoded_L{args.layer}_n{args.n_images}.pt"
    if cache.exists():
        log.info("loading cached encodings %s", cache)
        return torch.load(cache, map_location="cpu", weights_only=True)

    from PIL import Image

    enc = get_encoder()
    rows = rows[: args.n_images]

    img_vs, cap0_vs = [], []
    for i, r in enumerate(rows):
        img_vs.append(enc.image_v(Image.open(r["file"]).convert("RGB")))
        if (i + 1) % 100 == 0:
            log.info("images %d/%d", i + 1, len(rows))
    # caption 0 (the fit pair) for every image
    for j in range(0, len(rows), args.caption_batch):
        batch = [r["captions"][0] for r in rows[j : j + args.caption_batch]]
        cap0_vs.append(enc.text_vs(batch))
        if (j // args.caption_batch) % 20 == 0:
            log.info("cap0 %d/%d", j, len(rows))
    # all 5 captions for the eval tail
    eval_rows = rows[-args.n_eval:]
    cap5_vs = []
    flat5 = [c for r in eval_rows for c in r["captions"]]
    for j in range(0, len(flat5), args.caption_batch):
        cap5_vs.append(enc.text_vs(flat5[j : j + args.caption_batch]))
        if (j // args.caption_batch) % 20 == 0:
            log.info("cap5 %d/%d", j, len(flat5))

    obj = {
        "img": torch.stack(img_vs),                # (N, d)
        "cap0": torch.cat(cap0_vs),                # (N, d)
        "cap5": torch.cat(cap5_vs),                # (n_eval*5, d)
        "files": [r["file"] for r in rows],
        "captions5": [r["captions"] for r in eval_rows],
    }
    tmp = cache.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(cache)
    log.info("cached encodings -> %s", cache)
    return obj


# ---------------------------------------------------------------- procrustes
def fit_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes: argmin_W ||XW - Y||_F s.t. W'W=I. X,Y centered (n,d)."""
    M = (X.double().T @ Y.double())
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vh).float()


def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors of centered X: (d, k)."""
    _, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return Vh[:k].T.float()


def retrieval_eval(
    q: torch.Tensor,           # (nq, d*) projected+centered queries
    pool: torch.Tensor,        # (nc, d*) centered candidates
    hit_sets: list[set[int]],  # per query: candidate indices that count as hits
    ks=(1, 5, 10),
) -> dict:
    qn = F.normalize(q, dim=-1)
    pn = F.normalize(pool, dim=-1)
    sims = qn @ pn.T                                   # (nq, nc)
    order = sims.argsort(dim=-1, descending=True)
    out = {f"r@{k}": 0.0 for k in ks}
    ranks = []
    for i, hits in enumerate(hit_sets):
        row = order[i].tolist()
        first = next(r for r, c in enumerate(row) if c in hits)
        ranks.append(first + 1)
        for k in ks:
            if first < k:
                out[f"r@{k}"] += 1.0
    n = len(hit_sets)
    for k in ks:
        out[f"r@{k}"] = round(out[f"r@{k}"] / n, 4)
    out["median_rank"] = float(torch.tensor(ranks, dtype=torch.float).median())
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    _, rows = ensure_coco(args.work_dir)
    if len(rows) < args.n_images:
        args.n_images = len(rows)

    # One backbone instance shared by encoding + probes (2x 31B would OOM).
    _enc_holder: list[Encoder] = []

    def get_encoder() -> Encoder:
        if not _enc_holder:
            _enc_holder.append(Encoder(args))
        return _enc_holder[0]

    enc = encode_all(args, rows, get_encoder)

    img, cap0, cap5 = enc["img"], enc["cap0"], enc["cap5"]
    N, d = img.shape
    n_eval = len(enc["captions5"])
    n_fit_pool = N - n_eval
    log.info("N=%d d=%d fit_pool=%d eval=%d", N, d, n_fit_pool, n_eval)

    X_pool, Y_pool = img[:n_fit_pool], cap0[:n_fit_pool]
    X_eval = img[n_fit_pool:]
    pool5 = cap5                                        # (n_eval*5, d)
    hit_sets = [set(range(5 * i, 5 * i + 5)) for i in range(n_eval)]
    # text→image direction: query = caption 0 of eval imgs, pool = eval images
    t2i_q = cap0[n_fit_pool:]
    t2i_hits = [{i} for i in range(n_eval)]

    results: dict = {
        "backbone": args.backbone, "layer": args.layer, "d": d,
        "n_images": N, "n_eval": n_eval, "seed": args.seed,
        "runs": {},
    }

    headline_n = min(args.fit_sizes[-1], n_fit_pool)

    def run(name: str, n_fit: int, shuffle: bool = False,
            pca_k: int | None = None) -> None:
        Xf, Yf = X_pool[:n_fit].clone(), Y_pool[:n_fit].clone()
        if shuffle:
            perm = torch.randperm(n_fit)
            Yf = Yf[perm]
        mu_x, mu_y = Xf.mean(0), Yf.mean(0)
        Xc, Yc = Xf - mu_x, Yf - mu_y
        Px = Py = None
        if pca_k is not None:
            Px, Py = pca_basis(Xc, pca_k), pca_basis(Yc, pca_k)
            Xc, Yc = Xc @ Px, Yc @ Py
        W = fit_procrustes(Xc, Yc)
        fit_res = float(((Xc @ W - Yc).norm() / Yc.norm()))

        def proj_img(v):
            v = v - mu_x
            if Px is not None:
                v = v @ Px
            return v @ W

        def proj_txt(v):
            v = v - mu_y
            if Py is not None:
                v = v @ Py
            return v

        i2t = retrieval_eval(proj_img(X_eval), proj_txt(pool5), hit_sets)
        t2i = retrieval_eval(proj_txt(t2i_q), proj_img(X_eval), t2i_hits)
        # paired similarity (true image ↔ its caption0)
        pc = F.cosine_similarity(proj_img(X_eval), proj_txt(t2i_q), dim=-1)
        rec = {"n_fit": n_fit, "shuffle": shuffle, "pca_k": pca_k,
               "fit_residual": round(fit_res, 4),
               "i2t": i2t, "t2i": t2i,
               "paired_cen_fve": round(float(0.5 * (1 + pc.mean())), 4)}
        results["runs"][name] = rec
        log.info("%-24s i2t %s  t2i r@1=%.3f  pair_fve=%.3f  resid=%.3f",
                 name, i2t, t2i["r@1"], rec["paired_cen_fve"], fit_res)

    # 0) baseline: centered cosine, no W (identity "projection")
    mu_x0, mu_y0 = X_pool[:headline_n].mean(0), Y_pool[:headline_n].mean(0)
    base_i2t = retrieval_eval(X_eval - mu_x0, pool5 - mu_y0, hit_sets)
    base_t2i = retrieval_eval(t2i_q - mu_y0, X_eval - mu_x0, t2i_hits)
    pc0 = F.cosine_similarity(X_eval - mu_x0, t2i_q - mu_y0, dim=-1)
    results["runs"]["baseline_centered"] = {
        "i2t": base_i2t, "t2i": base_t2i,
        "paired_cen_fve": round(float(0.5 * (1 + pc0.mean())), 4)}
    log.info("%-24s i2t %s  t2i r@1=%.3f", "baseline_centered", base_i2t, base_t2i["r@1"])
    # modality-gap descriptives
    results["modality_gap"] = {
        "mu_cos": round(float(F.cosine_similarity(mu_x0, mu_y0, dim=0)), 4),
        "mu_img_norm": round(float(mu_x0.norm()), 2),
        "mu_txt_norm": round(float(mu_y0.norm()), 2),
    }

    # 1) train-size curve (full-d)
    for n_fit in args.fit_sizes:
        run(f"procrustes_n{n_fit}", min(n_fit, n_fit_pool))
    # 2) shuffled-pairs floor at headline size
    run(f"shuffled_n{headline_n}", headline_n, shuffle=True)
    # 3) PCA-subspace variants at headline size
    for k in args.pca_dims:
        run(f"procrustes_pca{k}_n{headline_n}", headline_n, pca_k=k)

    # 4) boundary probes against the 10k caption pool
    if args.caption_targets and args.caption_targets.exists() and args.probe_images:
        from PIL import Image

        obj = torch.load(args.caption_targets, map_location="cpu", weights_only=False)
        big_pool = torch.stack([a[-1] for a in obj["activations"]]).float()
        big_caps = obj["sequences"]
        e = get_encoder()
        mu_x, mu_y = X_pool[:headline_n].mean(0), Y_pool[:headline_n].mean(0)
        W = fit_procrustes(X_pool[:headline_n] - mu_x, Y_pool[:headline_n] - mu_y)
        pool_c = F.normalize(big_pool - big_pool.mean(0), dim=-1)
        probes = []
        for pi, path in enumerate(args.probe_images):
            v = e.image_v(Image.open(path).convert("RGB"))
            rec = {"image": str(path)}
            for tag, q in (("raw_centered", v - mu_x),
                           ("procrustes", (v - mu_x) @ W)):
                sims = pool_c @ F.normalize(q, dim=-1)
                order = sims.argsort(descending=True)
                rec[tag] = {
                    "top3": [[round(float(0.5 * (1 + sims[i])), 3),
                              big_caps[int(i)][:100]] for i in order[:3]],
                }
                if pi < len(args.probe_expected):
                    needle = args.probe_expected[pi].lower()
                    rank = next((r + 1 for r, i in enumerate(order.tolist())
                                 if needle in big_caps[int(i)].lower()), None)
                    rec[tag]["expected_rank"] = rank
            probes.append(rec)
            log.info("probe %s: raw top1=%s | proc top1=%s", Path(path).name,
                     rec["raw_centered"]["top3"][0], rec["procrustes"]["top3"][0])
        results["probes"] = probes

    args.out.write_text(json.dumps(results, indent=2))
    # persist the headline W for reuse (demo / follow-ups)
    mu_x, mu_y = X_pool[:headline_n].mean(0), Y_pool[:headline_n].mean(0)
    W = fit_procrustes(X_pool[:headline_n] - mu_x, Y_pool[:headline_n] - mu_y)
    torch.save({"W": W, "mu_img": mu_x, "mu_txt": mu_y,
                "layer": args.layer, "n_fit": headline_n, "seed": args.seed},
               args.out.with_suffix(".W.pt"))
    log.info("wrote %s (+ .W.pt)", args.out)


if __name__ == "__main__":
    main()
