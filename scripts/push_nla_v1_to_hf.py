"""Push SRT-NLA v1 model + dataset to the Hugging Face Hub.

Usage (after `huggingface-cli login`):

    # 1) Model repo (RiverRider/srt-nla-av-v1)
    python scripts/push_nla_v1_to_hf.py model \
        --ckpt /workspace/srt-adapter/artifacts/nla/ce_seq64_np16_30k/best_av.pt

    # 2) Dataset repo (RiverRider/srt-nla-targets-v1)
    python scripts/push_nla_v1_to_hf.py dataset \
        --targets /workspace/srt-adapter/artifacts/nla/targets_q7b_L20_seq64_30k_seed1.pt

    # 3) Both, in one go
    python scripts/push_nla_v1_to_hf.py all \
        --ckpt   .../ce_seq64_np16_30k/best_av.pt \
        --targets .../targets_q7b_L20_seq64_30k_seed1.pt

Card / config sources default to docs/hf/nla_v1/ (tracked in git), so a
fresh \`git pull\` on the vast box is sufficient — no rsync needed.

Notes
-----
* Repos are *created if missing*; existing files are *overwritten*. Pass
  `--private` to keep either repo private during preview.
* The dataset upload is ~26 GB; expect a long transfer. Resumes are handled
  by `huggingface_hub`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

MODEL_REPO = "RiverRider/srt-nla-av-v1"
DATASET_REPO = "RiverRider/srt-nla-targets-v1"


# --------------------------------------------------------------------- model

def push_model(ckpt: Path, release_dir: Path, private: bool) -> None:
    assert ckpt.is_file(), f"missing checkpoint: {ckpt}"
    card = release_dir / "MODEL_CARD.md"
    cfg = release_dir / "config.json"
    assert card.is_file(), f"missing model card: {card}"
    assert cfg.is_file(), f"missing config.json: {cfg}"

    create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=private)

    # README.md is the rendered card on HF
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
        commit_message="best_av.pt (ce_seq64_np16, 30k pairs)",
    )

    # Optional: ship the small JSON eval artifacts alongside the weights.
    eval_files = [
        "artifacts/nla/centered_eval_30k_M200.json",
        "artifacts/nla/oracle_ceiling_30k_v2.json",
        "artifacts/nla/rerank_eval_ce_seq64_np16_v2.json",
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
        path_in_repo="targets_q7b_L20_seq64_30k_seed1.pt",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="full targets (30k, seed=1)",
    )
    print(f"[ok] pushed dataset -> https://huggingface.co/datasets/{DATASET_REPO}")


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["model", "dataset", "all"])
    ap.add_argument("--ckpt", type=Path, help="path to best_av.pt")
    ap.add_argument("--targets", type=Path, help="path to targets_q7b_L20_seq64_30k_seed1.pt")
    ap.add_argument(
        "--release-dir",
        type=Path,
        default=Path("docs/hf/nla_v1"),
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
