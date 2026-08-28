#!/usr/bin/env python3
"""Strip sensory metaphors and negated-capability framing from the live HF cards.

Models do not have eyes, do not see and do not speak. They receive tensors and
emit tokens. Say what a component receives and what it produces.

Patches are applied to the card as it currently exists on the Hub rather than by
pushing a local copy, because several live cards have drifted from the repo and
pushing would clobber frontmatter. Each pattern must match exactly once or the
repo is skipped and reported.

    python scripts/fix_card_voice.py            # report only
    python scripts/fix_card_voice.py --apply
"""
import argparse
import re

from huggingface_hub import HfApi, get_token, hf_hub_download

# (repo, type, [(pattern, replacement), ...])
PATCHES = [
    ("RiverRider/walk-the-space", "space", [
        (r"Click anywhere in a 27B's understanding and hear it",
         "Click anywhere in a 27B's image space and read the point"),
        (r"a frozen \*\*Qwen3-0\.6B\*\*, which has no vision path and has never\s+seen a photograph, says what is at that spot",
         "a frozen **Qwen3-0.6B**, a text-only model whose entire input is that\none point, writes what is at that spot"),
        (r"printed what it said",
         "printed what it produced"),
        (r"\*\*Arbitrary points speak, not just stored ones\.\*\*",
         "**Any point resolves, not only the stored ones.**"),
        (r"which\s+a frozen Qwen3-0\.6B speaks from",
         "which\na frozen Qwen3-0.6B decodes"),
    ]),
    ("RiverRider/srt-sunstone", "space", [
        (r"it only knows the labels it was shown\. This read-out is different in kind:\s*\n\*\*it never saw an image in training\.\*\* It was trained only on text \u2014 to separate",
         "it only handles the labels in its training set. This read-out is different\nin kind: **its training data was text alone**, used to separate"),
    ]),
    ("RiverRider/srt-browser-head-118k", "model", [
        (r"survives the runtime change that R@1 cannot see\.",
         "survives the runtime change that R@1 does not register."),
    ]),
    ("RiverRider/srt-browser-demo", "space", [
        (r"A 0\.6B model in your tab searches what a 27B model saw",
         "A 0.6B model in your tab searches a 27B model's index"),
    ]),
    ("RiverRider/0.6b-reads-27b", "space", [
        (r"A 0\.6B model in your tab searches what a 27B model saw",
         "A 0.6B model in your tab searches a 27B model's index"),
    ]),
    ("RiverRider/srt-depth-probe-artifacts", "dataset", [
        (r"one model saw all three languages",
         "one model was trained on all three languages"),
    ]),
    ("RiverRider/srt-nla-av-gemma4", "model", [
        (r"- \*\*Sees and says\.\*\*", "- **Image to caption.**"),
        (r"- \*\*Says what it actually sees\.\*\* Shown a random-dot autostereogram",
         "- **Reports the texture that is present.** Given a random-dot autostereogram"),
        (r'"An abstract mosaic of tiny colored squares" \u2014 a faithful sentence-level\s+report of the texture the flat encoder truly perceives',
         '"An abstract mosaic of tiny colored squares", a faithful sentence-level\n  report of what a flat encoder actually receives'),
    ]),
    ("RiverRider/srt-verbalizer-v1", "model", [
        (r"The small model has no vision path\. It never sees the image\. It only ever\s+receives the vector\.",
         "The small model is text-only. Its entire input is the vector."),
        (r"> Not mind reading\. Reading aloud\. The record was always legible; this puts it\s*\n> in English\.",
         "> The record was always legible. This puts it in English."),
    ]),
    ("RiverRider/srt-nla-gemma4-artifacts", "dataset", [
        (r"say what it\s+sees and what it means \u2014 in full sentences",
         "turn its L47 states\ninto full sentences"),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=get_token())
    for repo, rtype, patches in PATCHES:
        txt = open(hf_hub_download(repo, "README.md", repo_type=rtype),
                   errors="ignore").read()
        new, applied, missed = txt, [], []
        for pat, rep in patches:
            out, n = re.subn(pat, rep, new)
            if n == 1:
                new, _ = out, applied.append(pat[:48])
            else:
                missed.append(f"{n}x {pat[:48]}")
        print(f"\n{rtype}/{repo}")
        for p in applied:
            print(f"    ok      {p}")
        for m in missed:
            print(f"    MISSED  {m}")
        if a.apply and new != txt:
            api.upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md",
                            repo_id=repo, repo_type=rtype,
                            commit_message="card voice: no sensory metaphors")
            print("    uploaded")


if __name__ == "__main__":
    main()
