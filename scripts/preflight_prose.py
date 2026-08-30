"""Pre-flight for outbound prose: verify every number, then catch voice tics.

Usage:  .venv/bin/python scripts/preflight_prose.py /tmp/fport_reply.txt

Two passes, and the first is the one that has caught real errors.

  NUMBERS  Every figure in the draft is checked against an artifact on disk,
           not against memory. Claims have been wrong here before: a head size
           quoted at 44 MB and 22 MB on two surfaces at once, a "382 MB model"
           that named the browser build rather than the Lab reader, and a
           median quoted from a different head than the one deployed. If a
           number cannot be traced to a file, it does not ship.

  VOICE    Patterns the user has rejected, each found ~6x by grep after being
           flagged once by eye. Significance signposting, negated-capability
           tics, authorship denial, hedging, deference, em dashes, and
           unexplained jargon that assumes context the reader does not have.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COV = ROOT / "artifacts/nla/verbalizer/caption_coverage"


def load_facts() -> dict[str, tuple[float, str]]:
    """Every number we are allowed to quote, read from the artifact that owns it."""
    f: dict[str, tuple[float, str]] = {}

    def arm(tag: str, fname: str, key: str = "real") -> None:
        p = COV / fname
        if not p.exists():
            return
        d = json.load(open(p))
        f[f"{tag} median"] = (d["arms"][key]["median_rank"], fname)
        f[f"{tag} gallery"] = (d["gallery"], fname)

    arm("shipped-recipe reader", "verb_eval_cap0.json")
    arm("np8 h2048", "verb_eval_all5_np8_h2048.json")
    arm("np32 h2048", "verb_eval_all5_np32.json")
    arm("np32 h512", "verb_eval_all5_np32_h512.json")

    head = ROOT / "release/srt-sunstone-linear-head/sunstone_linear_head_v3_drift.pt"
    if head.exists():
        import torch

        d = torch.load(head, map_location="cpu", weights_only=False)
        tot = 0
        for v in d.values():
            if isinstance(v, dict):
                tot += sum(int(t.numel()) for t in v.values() if hasattr(t, "numel"))
            elif hasattr(v, "numel"):
                tot += int(v.numel())
        f["head params"] = (tot, head.name)
        f["head proj dim"] = (d["img"]["weight"].shape[0], head.name)
    return f


# Program-level claims live in prose, not JSON, so the check is whether the figure
# actually appears in a paper. A number I remember but cannot find is not a number.
# Extra sources can be passed as trailing argv, for findings that live only in a memo.
SOURCES = ["paper_srt_program.md", "paper_nla.md", "paper_program.md", "README.md"]
EXTRA: list[Path] = []


def _json_numbers(p: Path) -> list[float]:
    """Every number in a result file, so a rounded quote can be checked by value."""
    out: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            out.append(float(o))
        elif isinstance(o, dict):
            for k, v in o.items():
                # Pool sizes and layer numbers live in the keys, not the values.
                try:
                    out.append(float(k))
                except (TypeError, ValueError):
                    pass
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    try:
        walk(json.load(open(p)))
    except Exception:
        pass
    return out


def in_papers(n: str) -> list[str]:
    hits = []
    # Papers write 1,384 while a draft writes 1384, so try both groupings.
    forms = {n}
    if "." not in n and len(n) > 3:
        forms.add(f"{int(n):,}")
    pat = "|".join(rf"(?<![\d.,]){re.escape(f)}(?![\d])" for f in forms)
    for name in SOURCES:
        p = ROOT / name
        if p.exists() and re.search(pat, p.read_text()):
            hits.append(name)
    dec = len(n.split(".")[1]) if "." in n else 0
    for p in EXTRA:
        if not p.exists():
            continue
        if p.suffix == ".json":
            # A result file holds 0.6664648, the draft quotes 0.666. Match by value.
            if any(round(v, dec) == float(n) for v in _json_numbers(p)):
                hits.append(p.name)
            # Notes and supersedes fields are prose, so a figure can live only in a string.
            elif re.search(pat, p.read_text()):
                hits.append(p.name)
        elif re.search(pat, p.read_text()):
            hits.append(p.name)
    return hits


_ALL_ARTIFACTS: list[Path] | None = None


def elsewhere_in_repo(n: str, cited: set[str]) -> list[str]:
    """Which artifact DOES hold this number, if the cited ones do not.

    Twice on 2026-08-29 a figure from a superseded run reached a public card:
    0.8192 was a view-only baseline from a 35k pilot that a 112k run had
    already replaced. "Not in any artifact" was the wrong message, because it
    was in an artifact, just not the one being cited. Naming the file turns a
    shrug into a diagnosis.
    """
    global _ALL_ARTIFACTS
    if _ALL_ARTIFACTS is None:
        _ALL_ARTIFACTS = sorted(
            p for p in (ROOT / "artifacts").rglob("*.json")
            if p.stat().st_size < 4_000_000)
    dec = len(n.split(".")[1]) if "." in n else 0

    def kinship(p: Path) -> int:
        """Longest shared filename prefix with anything cited.

        A pilot and the run that replaced it are named alike, so ranking by
        name similarity puts the actual suspect first instead of whichever
        unrelated artifact happens to contain the same float.
        """
        best = 0
        for c in cited:
            i = 0
            while i < min(len(c), len(p.name)) and c[i] == p.name[i]:
                i += 1
            best = max(best, i)
        return best

    hits = []
    for p in sorted(_ALL_ARTIFACTS, key=kinship, reverse=True):
        if p.name in cited:
            continue
        try:
            if any(round(v, dec) == float(n) for v in _json_numbers(p)):
                hits.append(str(p.relative_to(ROOT)))
        except Exception:
            continue
        if len(hits) >= 3:
            break
    return hits


TICS = {
    "em dash": [r"\u2014"],
    "signposting": [r"the interesting", r"worth sitting with", r"crucially",
                    r"notably", r"more than it sounds", r"the shape of the thing",
                    r"is closer to your question"],
    "negated capability": [r"no eyes", r"cannot see", r"never been shown", r"no vision"],
    "authorship denial": [r"not chosen by us", r"nobody ", r"we did not choose"],
    "hedging": [r"\barguably\b", r"\bsomewhat\b", r"\bperhaps\b", r"\bmight suggest\b",
                r"\bseems to\b", r"\bfairly\b", r"\bkind of\b", r"\bsort of\b"],
    "deference": [r"great question", r"thank you", r"if I may", r"just wanted",
                  r"humbly", r"I wonder if", r"forgive", r"apolog"],
    "cognition verb": [r"worked out", r"it knows", r"understands", r"figures out"],
    # Body metaphors. "no eyes" was caught above and then replaced with "speaks",
    # which is the same error wearing a different coat: models do not have
    # sensory organs and do not speak, they receive tensors and emit tokens.
    "body metaphor": [r"\bno eyes\b", r"\beyes\b", r"\bspeaks?\b", r"\bspoke\b",
                      r"\bsees\b", r"\bsaw\b", r"\bseeing\b", r"\bblind\b",
                      r"\blistens?\b", r"\bhears?\b", r"\bmind\b", r"\bthinks?\b",
                      r"\bwatches\b", r"\btells you\b", r"\bsays\b"],
    # Offering an experiment we could just run. If the box can do it, run it and
    # send the number instead of handing the correspondent homework.
    "deferred experiment": [r"is a run\b", r"is one job", r"harness (now )?exists",
                           r"we have the harness", r"would give the \w+ directly",
                           r"would settle (it|this)", r"is the experiment",
                           r"the shape to test it against", r"worth running",
                           r"someone should", r"is testable with", r"left as an exercise",
                           r"if you want(ed)? to test"],
    # "X is not A. It is B." Reads as rhetoric doing work the evidence should do.
    "antithesis": [r"is not [^.]{0,70}\.\s*it is\b", r"does not [^.]{0,70}\.\s*it \w+s\b",
                   r"\bnot [^.,]{0,45},\s*(but|it'?s|it is)\b", r"\brather than\b",
                   r"\bso,? not\b", r"\binstead of\b.{0,40}\bit\b"],
    # Constructions that sound like a distinction while asserting nothing.
    # "Both answers are yes, and they are not the same yes" shipped in a paper
    # abstract on 2026-08-30: it names no difference, so the reader learns
    # nothing and the sentence cannot be checked against a result. State the
    # two findings instead. "not the same AS <thing>" is exempt because it
    # names the comparand and is therefore checkable.
    "hollow distinction": [r"not the same (?!as\b)\w+\b", r"\bthe same \w+ twice\b",
                           r"\bin more than one sense\b", r"\btwo different kinds of\b",
                           r"\bboth (?:answers?|are) [^.]{0,40}\band (?:they|both)\b",
                           r"\bis and is not\b", r"\bmore than one way\b(?![^.]{0,40}\d)"],
    "unexplained jargon": [r"\bnp\d", r"\bL47\b", r"\bfve\b", r"\bSRT\b", r"\bR@\d",
                           r"\badapter\b", r"\bcheckpoint\b", r"\bhead-space\b"],
    # One domain is not "every time". Published as a general recipe on
    # 2026-08-29 off a single gallery, then failed to replicate on the other
    # two we already held states for. If a claim reaches for a universal,
    # either the replication is in hand or the word comes out.
    "overgeneralised": [r"every time", r"\balways\b", r"without exception",
                        r"in every case", r"never fails", r"\buniversally\b",
                        r"holds everywhere", r"in all cases", r"\bany domain\b",
                        r"across the board"],
}

# A number quoted next to a comparison verb is a claim that two systems were
# measured the same way. CheXNet's 0.8414 sat in our own code as a head-to-head
# for a day before anyone checked that it is a different test split. If a draft
# compares, it has to say on what.
COMPARE = r"(beats|against|versus|\bvs\.?\b|ahead of|outperforms|compared (?:to|with))"
# Naming the actual split file is naming the split, so test_list.txt counts.
MATCHED = r"(split|matched|same (?:test|set|protocol|holdout)|holdout|config|" \
          r"protocol|our harness|different test set|not comparable|" \
          r"test_list|official (?:split|test|list))"

# A score with no metric named near it. An abstract paragraph on 2026-08-30 read
# "scores 0.7774 against 0.7451", three numbers deep before it said what was
# being scored or on what task, which asks the reader to hold figures they
# cannot yet interpret. Any score in a claim needs its metric within reach.
SCORE = r"(?<![\d.])0\.\d{3,4}(?![\d])"
# Metric NAMES only. Relational words (margin, gain, cost, ratio, floor) were in
# this list first and made it useless: they appear in every comparison, so the
# offending paragraph passed its own check on the word "margin".
METRIC = r"(auroc|\bauc\b|r@\d|\brecall\b|\bprecision\b|\bf1\b|accuracy|" \
         r"spearman|pearson|cosine|\bfve\b|\bmap\b|average precision|" \
         r"sensitivity|specificity|dice|iou|perplexity|\bbleu\b|rouge)"

# A finding announced before the reader is told what was done. The abstract of
# this paper opened "a probe moves between backbones at no cost" three sentences
# before it said what a probe was, what moving one meant, or why it should be
# hard, which asks the reader to accept a result they cannot yet picture. State
# the procedure and the reason it might fail, then the outcome.
RESULT_CLAIM = r"\b(at no cost|costs? nothing|it works\b|survives?|outperforms?|" \
               r"beats?|improves? on|still improves|holds up|transports?\b)"
METHOD_STMT = r"\b(we fit|we train|we measure|we compute|we encode|we read|" \
              r"we apply|we score|we probe|we swap|we hold|fitted on|" \
              r"estimated on|pooled over|is fitted|are fitted|by fitting)"
OPENING_WORDS = 250


def main() -> int:
    path = Path(sys.argv[1])
    EXTRA.extend(Path(a) for a in sys.argv[2:])
    body = path.read_text()
    body = body.split("---", 1)[1] if "---" in body else body
    # Drafts are hard wrapped, so a multi-word pattern like "we have measured
    # nothing" straddles a newline and a literal match silently misses it.
    low = re.sub(r"\s+", " ", body.lower())
    facts = load_facts()

    print("=== NUMBERS: every figure traced to an artifact ===")
    quoted = {n.replace(",", "") for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", body)}
    exact = {f"{v:g}": (k, s) for k, (v, s) in facts.items()}
    traps = 0
    unsourced = 0
    for n in sorted(quoted, key=lambda s: -float(s)):
        # "section 8.7" cites someone's paper. It is not a measurement of ours.
        if re.search(rf"(?:section|§|sec\.)\s*{re.escape(n)}\b", low):
            continue
        if n in exact:
            label, src = exact[n]
            print(f"  {n:>10s}  {label}  <- {src}")
            continue
        # A quoted integer sitting on a half-integer artifact is the rounding trap:
        # "{:.0f}" is round-half-to-even, so 32.5 prints 32 but 49.5 prints 50.
        # Counts (prefix positions, epochs) collide numerically with medians, so
        # only flag when the number is not being used as a count.
        ctx = " ".join(re.findall(rf".{{0,40}}\b{re.escape(n)}\b.{{0,20}}", body))
        counted = re.search(r"\b(position|token|epoch|step|layer|dimension)", ctx)
        # A number inside a model name is not a measurement. "ResNet-50" was
        # flagged as rounded from a 49.5 median belonging to another paper.
        named = re.search(rf"[a-z]-{re.escape(n)}\b", low)
        near = [(k, v, s) for k, (v, s) in facts.items() if abs(v - float(n)) == 0.5]
        if near and not counted and not named:
            k, v, s = near[0]
            print(f"! {n:>10s}  ROUNDED from {v} ({k} <- {s}). Half-integer, quote {v}")
            traps += 1
        else:
            papers = in_papers(n)
            if papers:
                print(f"  {n:>10s}  appears in {', '.join(papers)}")
            else:
                unsourced += 1
                cited = {p.name for p in EXTRA}
                other = elsewhere_in_repo(n, cited)
                if other:
                    print(f"? {n:>10s}  NOT in the cited artifacts, but IS in "
                          f"{', '.join(other)} -- superseded run?")
                else:
                    print(f"? {n:>10s}  not in any artifact or paper, verify by hand")

    print("\n=== artifact values available ===")
    for label, (val, src) in sorted(facts.items()):
        print(f"  {label:26s} {val:>10,.1f}   {src}")

    print("\n=== VOICE ===")
    bad = 0
    for name, pats in TICS.items():
        hits = [m.group(0) for p in pats for m in re.finditer(p, low)]
        bad += len(hits)
        mark = "  " if not hits else "!!"
        print(f"{mark} {name:20s} {len(hits)}" + (f"   {sorted(set(hits))}" if hits else ""))

    words = len(body.split())
    print(f"\n=== COMPARISONS: does every head-to-head say on what ===")
    unmatched = 0
    # Prose only. A negative-results ledger is a table of hypothesis strings, and
    # matching "beats" inside a row header flagged five rows that state no claim
    # of ours. Same exclusion the SCORES check uses.
    prose_only = re.sub(r"\s+", " ", "\n".join(
        l for l in body.lower().split("\n") if not l.lstrip().startswith("|")))
    for m in re.finditer(COMPARE, prose_only):
        # Wide enough that a headline claim can be qualified a sentence or two
        # later, which is how prose actually reads, and still narrow enough that
        # a split named in a different section does not launder the claim.
        window = prose_only[max(0, m.start() - 400): m.end() + 400]
        if not re.search(r"\d", window):
            continue
        if not re.search(MATCHED, window):
            unmatched += 1
            print(f"! {m.group(0):<16s} "
                  f"...{prose_only[max(0, m.start() - 60):m.end() + 60]}...")
    if not unmatched:
        print("  every comparison names its split or protocol")

    print(f"\n=== SCORES: is the metric named before the first figure ===")
    bare = 0
    # Per section, not per number. Flagging every figure produced thirty hits on
    # a clean draft, and a check that fires on everything gets ignored. The rule
    # is narrower: a reader meeting a section's FIRST score should already have
    # been told what is being measured.
    low_nl = body.lower()
    sections = re.split(r"\n(?=#{1,3} )", low_nl)
    for sec in sections:
        head = sec.split("\n", 1)[0].strip("# ").strip() or "(opening)"
        prose = "\n".join(l for l in sec.split("\n") if not l.lstrip().startswith("|"))
        first = re.search(SCORE, prose)
        if not first:
            continue
        # The metric may be named anywhere before the figure, or in the same
        # breath just after it.
        lead = prose[:first.start()] + prose[first.start():first.end() + 160]
        if not re.search(METRIC, lead):
            bare += 1
            n = len(re.findall(SCORE, prose))
            snippet = re.sub(r"\s+", " ", prose[max(0, first.start() - 90):first.end() + 40])
            print(f"! {head[:38]:<40s} {n} score(s), first unlabelled")
            print(f"    ...{snippet}...")
    if not bare:
        print("  every section names its metric before its first figure")

    print(f"\n=== OPENING: is the method stated before the finding ===")
    opening = " ".join(low.split()[:OPENING_WORDS])
    res_m = re.search(RESULT_CLAIM, opening)
    met_m = re.search(METHOD_STMT, opening)
    if res_m and (met_m is None or met_m.start() > res_m.start()):
        where = "no method statement at all" if met_m is None else \
                f"method arrives later, at '{met_m.group(0)}'"
        print(f"! '{res_m.group(0)}' claims a result first, {where}")
        print(f"    ...{opening[max(0, res_m.start() - 90):res_m.end() + 60]}...")
    else:
        print("  the opening says what was done before it says how it came out")

    print(f"\n   words {words}")
    # Scoping can be phrased many ways. What matters is that the draft names a
    # limit or a banked negative somewhere, not that it uses the word "scoping".
    scope_pats = [r"scoping", r"we have measured nothing", r"not minds",
                  r"have not touched", r"have never touched", r"not going to pretend",
                  r"banked it as a null", r"banked as a null", r"cut against us",
                  r"is not universal", r"stands untested", r"cannot support it",
                  r"where we are weak", r"came back null", r"per word it fails",
                  r"is untested", r"remains untested", r"we have not measured",
                  r"two limits", r"will not paper over", r"without settling",
                  r"scope note", r"does not say", r"my error",
                  r"we have not", r"one backbone at one layer", r"did not replicate"]
    has_scope = any(re.search(p, low) for p in scope_pats)
    print(f"   explicit scoping present: {has_scope}")
    print(f"   rounding traps: {traps}")
    print(f"   numbers with no source: {unsourced}")
    print(f"   comparisons with no split named: {unmatched}")
    ok = (bad == 0 and has_scope and traps == 0 and unsourced == 0
          and unmatched == 0)
    print(f"\n{'CLEAN' if ok else 'REVIEW NEEDED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
