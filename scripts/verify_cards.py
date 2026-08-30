#!/usr/bin/env python3
"""Check every number in a publisher's card against the artifacts, before upload.

Two wrong figures reached public cards in one session. 0.8192 was carried over
from a superseded 35k pilot, and the view-only baseline was quoted as 0.5896
when the reproducible value is 0.5883. Both survived because preflight_prose.py
was only ever pointed at prose drafts, never at the card text the publishers
actually upload.

The second one is the more instructive failure. 0.5827 and 0.5896 were both
real outputs, from the same code on two machines, because auroc() promised tied
ranks were averaged and did not average them. A binary feature is all ties, so
the baseline silently became a property of the sort order. Agreeing with an
artifact is not the same as being right, and the first fix here replaced a
wrong number with another wrong number for exactly that reason.

This imports each publisher, pulls its CARD string, and runs the same number
check over it. A card is prose that makes claims, so it gets the same gate.

    python scripts/verify_cards.py
"""
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile

# publisher -> artifacts whose numbers that card is allowed to quote
CARDS = {
    "scripts/publish_cxr_probe.py": [
        "artifacts/nla/cxr14_probe_full112k.json",
        "artifacts/nla/cxr14_pool_sweep.json"],
    "scripts/publish_cxr_model.py": [
        "artifacts/nla/cxr14_probe_full112k.json",
        "artifacts/nla/cxr14_pool_sweep.json"],
    "scripts/publish_cxr_space.py": [
        "artifacts/nla/cxr14_probe_full112k.json"],
    "scripts/publish_omni_dataset.py": [
        "artifacts/nla/omni/xvendor4.json",
        "artifacts/nla/omni/triadic_composition_roco.json",
        "artifacts/nla/omni/head_swap_roco.json",
        "artifacts/nla/omni/geometry_compare_roco.json"],
}


def card_text(path):
    """Load the module without running main(), and return whatever it uploads."""
    spec = importlib.util.spec_from_file_location("pub", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    parts = [getattr(mod, n) for n in ("CARD", "README", "MODEL_CARD")
             if isinstance(getattr(mod, n, None), str)]
    return "\n\n".join(parts)


def main():
    bad = 0
    for pub, arts in CARDS.items():
        if not pathlib.Path(pub).is_file():
            continue
        arts = [a for a in arts if pathlib.Path(a).is_file()]
        try:
            text = card_text(pub)
        except Exception as e:
            print(f"{pub}: could not load ({type(e).__name__}: {e})")
            bad += 1
            continue
        if not text.strip():
            print(f"{pub}: no CARD/README string found")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(text)
            tmp = f.name
        out = subprocess.run(
            [sys.executable, "scripts/preflight_prose.py", tmp, *arts],
            capture_output=True, text=True).stdout
        unsourced = re.findall(r"^\?\s+(\S+)\s+not in any", out, re.M)
        n = len(unsourced)
        flag = "OK " if n == 0 else "!! "
        print(f"{flag}{pub}  ({len(arts)} artifacts)  unsourced: {n}")
        for u in unsourced:
            print(f"     {u}")
        bad += n
    print(f"\n{'clean' if bad == 0 else str(bad) + ' number(s) need a source or a fix'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
