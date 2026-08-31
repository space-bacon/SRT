#!/usr/bin/env python3
"""Publish the hivemind evidence to the Hub.

Every figure in paper_hivemind.md resolves to a file in this upload. The point of
publishing is that a reader can rerun the analysis without the four Blackwell cards
and the 830 GB of weights the generations cost.

The card lives in docs/model_cards/srt-hivemind.md so verify_cards.py can gate it
alongside the others rather than it drifting as an inline string.

    python scripts/publish_hivemind.py --dry-run
    python scripts/publish_hivemind.py
"""
import argparse
import pathlib

REPO = "RiverRider/srt-hivemind"
CARD_PATH = pathlib.Path("docs/model_cards/srt-hivemind.md")
ROOT = pathlib.Path("artifacts/nla")

# (local dir, remote prefix, glob). Ordered as the card's Layout section reads.
SETS = [
    ("atlas", "atlas", "*.json"),
    ("atlas/states", "atlas/states", "*.npz"),
    ("atlas/samples", "atlas/samples", "*.json"),
    ("atlas/samples_instruct", "atlas/samples_instruct", "*.json"),
    ("coder_ladder", "coder_ladder", "*.json"),
    ("ministral_ladder", "ministral_ladder", "*.json"),
    ("ministral_ladder/corrupt_chat_pre_fix", "ministral_ladder/corrupt_chat_pre_fix", "*.json"),
    ("ministral_suppression", "ministral_suppression", "*.json"),
    ("crosslab_suppression", "crosslab_suppression", "*.json"),
    ("mbpp_ladder", "mbpp_ladder", "*.json"),
    ("ensemble", "ensemble", "*.json"),
    ("ensemble/excluded", "ensemble/excluded", "*.json"),
    ("verifier", "verifier", "*.json"),
    ("code_select", "code_select", "*.json"),
    ("holonomy", "holonomy", "*.json"),
]

SCRIPTS = [
    "hivemind_gen.py", "openweight_transport_atlas.py", "hivemind_sampled_protocol.py",
    "hivemind_posttraining_isolation.py", "hivemind_template_decomp.py",
    "hivemind_suppression.py", "hivemind_mechanism_link.py",
    "coder_ladder.py", "coder_ladder_analyze.py", "ministral_ladder.py",
    "ministral_suppression.py", "crosslab_suppression.py", "mbpp_ladder.py",
    "fetch_mbpp.py", "ensemble_ceiling.py",
    "code_select.py", "visible_tests.py", "verifier_select.py",
    "exec_guided_select.py", "consensus_select.py", "chat_consensus.py",
    "holonomy_select.py",
]


def plan():
    items = []
    card = CARD_PATH
    if not card.is_file():
        raise SystemExit(f"missing card: {card}")
    items.append((card, "README.md"))
    for src, dst, pat in SETS:
        d = ROOT / src
        if not d.is_dir():
            continue
        for p in sorted(d.glob(pat)):
            if p.is_file():
                items.append((p, f"{dst}/{p.name}"))
    for s in SCRIPTS:
        p = pathlib.Path("scripts") / s
        if p.is_file():
            items.append((p, f"scripts/{s}"))
    paper = pathlib.Path("paper_hivemind.md")
    if paper.is_file():
        items.append((paper, "paper_hivemind.md"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()

    items = plan()
    total = sum(p.stat().st_size for p, _ in items)
    print(f"{len(items)} files, {total / 1e6:.0f} MB -> {a.repo}")
    if a.dry_run:
        by = {}
        for p, r in items:
            by.setdefault(r.split("/")[0], []).append(p)
        for k, v in sorted(by.items()):
            print(f"  {k:34s} {len(v):4d} files  {sum(x.stat().st_size for x in v)/1e6:8.1f} MB")
        return

    from huggingface_hub import HfApi, get_token
    api = HfApi(token=get_token())
    api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
    for p, remote in items:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=a.repo, repo_type="dataset")
        print(f"  {remote}")
    print(f"\nhttps://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
