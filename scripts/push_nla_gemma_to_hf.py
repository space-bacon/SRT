"""Push SRT-NLA Gemma-2-2B model + dataset to the Hugging Face Hub.

Third-backbone replication of `scripts/push_nla_v1_to_hf.py` /
`scripts/push_nla_llama_to_hf.py`. Same logic, different repos /
target filename / eval artifacts. Run after `huggingface-cli login`
(write token for RiverRider).

Usage:

    # 1) Model repo (RiverRider/srt-nla-av-gemma2-2b-v1)
    python scripts/push_nla_gemma_to_hf.py model \
        --ckpt artifacts/nla/gemma2_2B/sft/best_av.pt

    # 2) Dataset repo (RiverRider/srt-nla-targets-gemma2-2b-v1)
    python scripts/push_nla_gemma_to_hf.py dataset \
        --targets artifacts/nla/gemma2_2B/targets_L19_seq64_30k_seed1.pt

    # 3) Both, in one go
    python scripts/push_nla_gemma_to_hf.py all \
        --ckpt    artifacts/nla/gemma2_2B/sft/best_av.pt \
        --targets artifacts/nla/gemma2_2B/targets_L19_seq64_30k_seed1.pt

Cards / config sources default to docs/hf/nla_v1_gemma2_2b/ (tracked in git).

Notes
-----
* Repos are *created if missing*; existing files are *overwritten*. Pass
  `--private` to keep either repo private during preview.
* The dataset upload is ~17 GB; expect a long transfer. Resumes are
  handled by `huggingface_hub`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import create_repo, upload_file

MODEL_REPO = "RiverRider/srt-nla-av-gemma2-2b-v1"
DATASET_REPO = "RiverRider/srt-nla-targets-gemma2-2b-v1"
TARGETS_FILENAME = "targets_L19_seq64_30k_seed1.pt"


# --------------------------------------------------------------------- model

def push_model(ckpt: Path, release_dir: Path, private: bool) -> None:
    assert ckpt.is_file(), f"missing checkpoint: {ckpt}"
    card = release_dir / "MODEL_CARD.md"
    cfg = release_dir / "config.json"
    assert card.is_file(), f"missing model card: {card}"
    assert cfg.is_file(), f"missing config.json: {cfg}"

    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=private)

    upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="model card",
    )
    upload_file(
        path_or_fileobj=str(cfg),
        path_in_repo="config.json",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="config",
    )
    upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="best_av.pt",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="best_av.pt (Gemma-2-2B L19, 30k pairs, 3 epochs)",
    )

    eval_files = [
        "artifacts/nla/gemma2_2B/centered_eval_M200_K64.json",
        "artifacts/nla/gemma2_2B/rerank_eval_M200_K32.json",
        "artifacts/nla/gemma2_2B/oracle_ceiling_M200.json",
    ]
    for rel in eval_files:
        p = Path(rel)
        if p.is_file():
            upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"eval/{p.name}",
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"eval artifact {p.name}",
            )
    print(f"[ok] pushed model -> https://huggingface.co/{MODEL_REPO}")


# ------------------------------------------------------------------- dataset

def push_dataset(targets: Path, release_dir: Path, private: bool) -> None:
    assert targets.is_file(), f"missing targets file: {targets}"
    card = release_dir / "DATASET_CARD.md"
    assert card.is_file(), f"missing dataset card: {card}"

    create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=private)

    upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="dataset card",
    )
    upload_file(
        path_or_fileobj=str(targets),
        path_in_repo=TARGETS_FILENAME,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="full targets (30k, seed=1, Gemma-2-2B L19)",
    )
    print(f"[ok] pushed dataset -> https://huggingface.co/datasets/{DATASET_REPO}")


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["model", "dataset", "all"])
    ap.add_argument("--ckpt", type=Path, help="path to best_av.pt")
    ap.add_argument("--targets", type=Path, help=f"path to {TARGETS_FILENAME}")
    ap.add_argument(
        "--release-dir",
        type=Path,
        default=Path("docs/hf/nla_v1_gemma2_2b"),
        help="directory holding MODEL_CARD.md, DATASET_CARD.md, config.json",
    )
    ap.add_argument("--private", action="store_true", help="create repos as private")
    args = ap.parse_args()

    if args.mode in ("model", "all"):
        assert args.ckpt, "--ckpt is required for model/all"
        push_model(args.ckpt, args.release_dir, args.private)
    if args.mode in ("dataset", "all"):
        assert args.targets, "--targets is required for dataset/all"
        push_dataset(args.targets, args.release_dir, args.private)


if __name__ == "__main__":
    main()
